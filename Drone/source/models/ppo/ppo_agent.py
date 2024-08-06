import os
import sys
import json
import logging
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from stable_baselines3.common.vec_env import DummyVecEnv
import airsim

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.append(project_root)

from Drone.source.envs.airsim_env import AirSimEnv
from Drone.source.envs.drone_controller import DroneController
from Drone.source.models.nn.policy_network import AdvancedPolicyNetwork
from Drone.source.models.nn.critic_network import AdvancedCriticNetwork
from Drone.source.models.ppo.ppo_utils import compute_gae, normalize
from Drone.source.models.nn.common_layers import ICM
from Drone.source.models.nn.Predictive_model import PredictiveModel
from Drone.source.utilities.custom_logger import CustomLogger
from Drone.source.learning.curriculum_learning import CurriculumLearning
from Drone.source.learning.hierarchical_rl import HighLevelPolicy, LowLevelPolicy, HierarchicalRLAgent
from Drone.source.learning.multi_agent_cooperation import MultiAgentCooperation

class PPOAgent:
    class Memory:
        def __init__(self):
            self.actions = []
            self.states = []
            self.visuals = []
            self.log_probs = []
            self.rewards = []
            self.dones = []
            self.goals = []

        def reset(self):
            self.actions.clear()
            self.states.clear()
            self.visuals.clear()
            self.log_probs.clear()
            self.rewards.clear()
            self.dones.clear()
            self.goals.clear()

        def add(self, action, state, visual, log_prob, reward, done, goal):
            self.actions.append(action)
            self.states.append(state)
            self.visuals.append(visual)
            self.log_probs.append(log_prob)
            self.rewards.append(reward)
            self.dones.append(done)
            self.goals.append(goal)

        def get_tensors(self, device):
            states = torch.tensor(np.vstack(self.states), dtype=torch.float32, device=device)
            actions = torch.tensor(np.vstack(self.actions), dtype=torch.float32, device=device)
            rewards = torch.tensor(np.vstack(self.rewards), dtype=torch.float32, device=device)
            dones = torch.tensor(np.vstack(self.dones), dtype=torch.bool, device=device)
            visuals = torch.tensor(np.vstack(self.visuals), dtype=torch.float32, device=device)
            log_probs = torch.tensor(np.vstack(self.log_probs), dtype=torch.float32, device=device)
            goals = torch.tensor(np.vstack(self.goals), dtype=torch.float32, device=device) if self.goals else None

            return states, actions, visuals, log_probs, rewards, dones, goals

    def __init__(self, config, logger=None, drone_controller=None):
        self.config = config
        self.drone_controller = drone_controller
        self.logger = logger or CustomLogger("PPOAgent", log_dir="./logs")
        
        self.setup_device()
        self.setup_networks()
        self.setup_training_components()
        self.logger.info("PPOAgent initialized successfully.")

    def setup_device(self):
        device_config = self.config['ppo'].get('device', 'auto')
        self.device = torch.device('cuda' if torch.cuda.is_available() and device_config == 'auto' else 'cpu')
        self.logger.info(f"Using device: {self.device}")

    def setup_networks(self):
        policy_config = {
            'image_channels': self.config['icm']['image_channels'],
            'image_height': self.config['icm']['image_height'],
            'image_width': self.config['icm']['image_width'],
            'cnn': self.config['icm']['cnn'],
            'use_batch_norm': self.config['policy_network'].get('use_batch_norm', True),
            'use_dropout': self.config['policy_network'].get('use_dropout', True),
            'dropout_rate': self.config['policy_network'].get('dropout_rate', 0.2),
            'use_attention': self.config['policy_network'].get('use_attention', True),
            'num_action_heads': self.config['policy_network'].get('num_action_heads', 1),
        }

        self.policy_network = AdvancedPolicyNetwork(
            self.config['environment']['state_dim'],
            self.config['environment']['action_dim'],
            continuous=self.config['ppo']['continuous'],
            hidden_sizes=self.config['policy_network']['hidden_layers'],
            config=policy_config
        ).to(self.device)

        critic_config = {
            'image_channels': self.config['icm']['image_channels'],
            'image_height': self.config['icm']['image_height'],
            'image_width': self.config['icm']['image_width'],
            'cnn': self.config['icm']['cnn'],
            'use_batch_norm': self.config['critic_network'].get('use_batch_norm', True),
            'use_dropout': self.config['critic_network'].get('use_dropout', True),
            'dropout_rate': self.config['critic_network'].get('dropout_rate', 0.2),
            'use_attention': self.config['critic_network'].get('use_attention', True),
        }

        self.critic_network = AdvancedCriticNetwork(
            self.config['environment']['state_dim'],
            hidden_sizes=self.config['critic_network']['hidden_layers'],
            config=critic_config
        ).to(self.device)

        self.icm = ICM(self.config['icm']).to(self.device)

        self.predictive_model = PredictiveModel(
            self.config['environment']['state_dim'] + self.config['environment']['action_dim'],
            self.config['environment']['state_dim'],
            self.config['predictive_model']['hidden_layers'],
            self.config['icm']['cnn']
        ).to(self.device)

        self.parameters = list(self.policy_network.parameters()) + \
                          list(self.critic_network.parameters()) + \
                          list(self.icm.parameters()) + \
                          list(self.predictive_model.parameters())

    def setup_training_components(self):
        self.optimizer = optim.Adam(self.parameters, lr=self.config['ppo']['learning_rate'])
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)
        self.scaler = GradScaler() if torch.cuda.is_available() else None
        self.gamma = self.config['ppo']['gamma']
        self.tau = self.config['ppo']['gae_lambda']
        self.epsilon = self.config['ppo']['clip_range']
        self.k_epochs = self.config['ppo']['n_epochs']
        self.clip_grad = self.config['ppo']['max_grad_norm']
        self.entropy_coef = self.config['ppo']['ent_coef']
        self.vf_coef = self.config['ppo']['vf_coef']
        self.writer = SummaryWriter(self.config['ppo']['tensorboard_log'])
        self.memory = self.Memory()
        self.icm_weight = self.config.get('icm_weight', 0.01)
        self.total_steps = 0

        if self.config['hrl']['use_hierarchical']:
            self.setup_hierarchical_rl()

        if self.config['curriculum_learning']['use_curriculum']:
            self.setup_curriculum_learning()

        if self.config['multi_agent']['use_multi_agent']:
            self.setup_multi_agent_cooperation()

    def setup_hierarchical_rl(self):
        self.high_level_policy = HighLevelPolicy(
            self.config['hrl']['high_level_policy']['input_size'],
            self.config['hrl']['sub_goal_dim'],
            self.config['hrl']['high_level_policy']['hidden_layers']
        ).to(self.device)
        self.low_level_policy = LowLevelPolicy(
            self.config['environment']['state_dim'] + self.config['hrl']['sub_goal_dim'],
            self.config['environment']['action_dim'],
            self.config['policy_network']['hidden_layers']
        ).to(self.device)
        self.hrl_agent = HierarchicalRLAgent(self.high_level_policy, self.low_level_policy)

    def setup_curriculum_learning(self):
        self.curriculum = CurriculumLearning(self.config)

    def setup_multi_agent_cooperation(self):
        self.multi_agent_cooperation = MultiAgentCooperation(
            num_agents=self.config['multi_agent']['num_agents'],
            state_dim=self.config['environment']['state_dim'],
            action_dim=self.config['environment']['action_dim'],
            hidden_layers=self.config['policy_network']['hidden_layers']
        )

    def select_action(self, observation):
        try:
            state = torch.FloatTensor(observation['state']).unsqueeze(0).to(self.device)
            visual = torch.FloatTensor(observation['visual']).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action, log_prob = self.policy_network.get_action(state, visual)

            return action.cpu().numpy().flatten(), log_prob.cpu().numpy()
        except Exception as e:
            self.logger.error(f"Error in select_action: {e}")
            raise

    def update(self):
        if len(self.memory.actions) < self.config['ppo']['batch_size']:
            return

        actions, states, visuals, log_probs, rewards, dones, goals = self.memory.get_tensors(self.device)

        with torch.cuda.amp.autocast(enabled=self.device.type == 'cuda'):
            next_value = self.critic_network(states[-1:], visuals[-1:]).detach()
            returns, advantages = compute_gae(next_value, rewards, dones, self.critic_network(states, visuals), self.gamma, self.tau)

            if self.config['advanced_training_techniques']['normalize_advantages']:
                advantages = normalize(advantages)

            for _ in range(self.k_epochs):
                new_log_probs, state_values, entropy = self.policy_network.evaluate_actions(states, visuals, actions)
                ratios = torch.exp(new_log_probs - log_probs)

                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(state_values, returns)
                intrinsic_rewards = self.icm.intrinsic_reward(states, states[1:], actions, visuals, visuals[1:])
                
                predicted_next_states = self.predictive_model(states, actions, visuals)
                predictive_loss = nn.functional.mse_loss(predicted_next_states, states[1:])
                
                total_loss = (
                    policy_loss 
                    + self.vf_coef * value_loss 
                    - self.entropy_coef * entropy 
                    + self.icm_weight * intrinsic_rewards.mean()
                    + predictive_loss
                )

                self.optimizer.zero_grad()
                if self.scaler:
                    self.scaler.scale(total_loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_value_(self.parameters, self.clip_grad)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    total_loss.backward()
                    torch.nn.utils.clip_grad_value_(self.parameters, self.clip_grad)
                    self.optimizer.step()

                self.writer.add_scalar('Loss/Policy', policy_loss.item(), self.total_steps)
                self.writer.add_scalar('Loss/Value', value_loss.item(), self.total_steps)
                self.writer.add_scalar('Loss/Total', total_loss.item(), self.total_steps)
                self.writer.add_scalar('Entropy', entropy.item(), self.total_steps)
                self.writer.add_scalar('Loss/Predictive', predictive_loss.item(), self.total_steps)
                self.writer.add_scalar('Loss/Intrinsic', intrinsic_rewards.mean().item(), self.total_steps)

                self.total_steps += 1

        self.lr_scheduler.step()
        self.memory.reset()

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'version': '1.0',
            'policy_state_dict': self.policy_network.state_dict(),
            'critic_state_dict': self.critic_network.state_dict(),
            'icm_state_dict': self.icm.state_dict(),
            'predictive_model_state_dict': self.predictive_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'high_level_policy_state_dict': self.high_level_policy.state_dict() if hasattr(self, 'high_level_policy') else None,
            'low_level_policy_state_dict': self.low_level_policy.state_dict() if hasattr(self, 'low_level_policy') else None
        }, path)
        self.logger.info(f'Model saved at {path}')

    def load_model(self, path):
        try:
            checkpoint = torch.load(path)
            self.policy_network.load_state_dict(checkpoint['policy_state_dict'])
            self.critic_network.load_state_dict(checkpoint['critic_state_dict'])
            self.icm.load_state_dict(checkpoint['icm_state_dict'])
            self.predictive_model.load_state_dict(checkpoint['predictive_model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            if hasattr(self, 'high_level_policy') and checkpoint['high_level_policy_state_dict']:
                self.high_level_policy.load_state_dict(checkpoint['high_level_policy_state_dict'])
            if hasattr(self, 'low_level_policy') and checkpoint['low_level_policy_state_dict']:
                self.low_level_policy.load_state_dict(checkpoint['low_level_policy_state_dict'])
            self.logger.info('Model loaded successfully')
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {e}")

    def train(self, env, total_timesteps, save_path):
        episode_rewards = []
        episode_lengths = []
        episode_reward = 0
        episode_length = 0

        try:
            observation = env.reset()
            for timestep in range(total_timesteps):
                action, log_prob = self.select_action(observation)
                next_observation, reward, done, _ = env.step(action)
                
                state = observation['state']
                visual = observation['visual']
                
                # Ensure state and visual are properly formatted
                if isinstance(state, np.ndarray) and state.ndim == 3:
                    state = state.squeeze(0)
                if isinstance(visual, np.ndarray) and visual.ndim == 5:
                    visual = visual.squeeze(0)
                
                self.memory.add(action, state, visual, log_prob, reward, done, None)
                
                episode_reward += reward
                episode_length += 1
                
                if done or episode_length >= self.config['ppo']['n_steps']:
                    self.update()
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(episode_length)
                    self.logger.info(f"Episode {len(episode_rewards)}: Reward = {episode_reward}, Length = {episode_length}")
                    
                    observation = env.reset()
                    episode_reward = 0
                    episode_length = 0
                else:
                    observation = next_observation

                if (timestep + 1) % self.config['ppo']['save_freq'] == 0:
                    self.save_model(save_path)
                    self.logger.info(f"Model saved at timestep {timestep + 1}")

            self.save_model(save_path)
            self.logger.info(f"Final model saved at timestep {total_timesteps}")

        except Exception as e:
            self.logger.error(f"Error during training: {e}")
            raise

        return episode_rewards, episode_lengths

    def evaluate(self, env, num_episodes):
        total_rewards = []
        total_lengths = []
        for _ in range(num_episodes):
            observation = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False

            while not done:
                action, _ = self.select_action(observation)
                observation, reward, done, _ = env.step(action)
                episode_reward += reward
                episode_length += 1

            total_rewards.append(episode_reward)
            total_lengths.append(episode_length)

        return {
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'mean_length': np.mean(total_lengths),
            'std_length': np.std(total_lengths)
        }

    @staticmethod
    def train_ppo_agent(agent, env, total_timesteps, save_path):
        try:
            episode_rewards, episode_lengths = agent.train(env, total_timesteps, save_path)
            return episode_rewards, episode_lengths
        except Exception as e:
            agent.logger.error(f"An error occurred during training: {e}")
            raise e