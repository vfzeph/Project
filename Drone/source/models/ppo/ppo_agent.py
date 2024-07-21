import os
import sys
import json
import logging
import numpy as np
from collections import deque
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
            return (
                torch.tensor(np.vstack(self.actions), device=device).float(),
                torch.tensor(np.vstack(self.states), device=device).float(),
                torch.tensor(np.stack(self.visuals), device=device).float(),
                torch.tensor(np.array(self.log_probs), device=device).float(),
                torch.tensor(np.array(self.rewards), device=device).float(),
                torch.tensor(np.array(self.dones), device=device).bool(),
                torch.tensor(np.vstack(self.goals), device=device).float() if self.goals else None
            )

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
            self.config['policy_network']['input_size'],
            self.config['policy_network']['output_size'],
            self.config['ppo']['continuous'],
            self.config['policy_network']['hidden_layers'],
            policy_config
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
            self.config['policy_network']['input_size'],
            self.config['critic_network']['hidden_layers'],
            critic_config
        ).to(self.device)

        self.icm = ICM(self.config['icm']).to(self.device)

        self.predictive_model = PredictiveModel(
            self.config['policy_network']['input_size'] + self.config['policy_network']['output_size'],
            self.config['policy_network']['input_size'],
            self.config['predictive_model']['hidden_layers'],
            self.config['icm']['cnn']
        ).to(self.device)

        self.parameters = list(self.policy_network.parameters()) + \
                          list(self.critic_network.parameters()) + \
                          list(self.icm.parameters()) + \
                          list(self.predictive_model.parameters())

        # Check if all networks are on the same device
        networks = [self.policy_network, self.critic_network, self.icm, self.predictive_model]
        if not all(next(net.parameters()).device == self.device for net in networks):
            self.logger.warning("Not all networks are on the same device!")

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
            self.config['policy_network']['input_size'] + self.config['hrl']['sub_goal_dim'],
            self.config['policy_network']['output_size'],
            self.config['policy_network']['hidden_layers']
        ).to(self.device)
        self.hrl_agent = HierarchicalRLAgent(self.high_level_policy, self.low_level_policy)

    def setup_curriculum_learning(self):
        self.curriculum = CurriculumLearning(self.config)

    def setup_multi_agent_cooperation(self):
        self.multi_agent_cooperation = MultiAgentCooperation(
            num_agents=self.config['multi_agent']['num_agents'],
            state_dim=self.config['policy_network']['input_size'],
            action_dim=self.config['policy_network']['output_size'],
            hidden_layers=self.config['policy_network']['hidden_layers']
        )

    def select_action(self, observation):
        state = torch.FloatTensor(observation['state']).to(self.device)
        visual = torch.FloatTensor(observation['visual']).to(self.device)
        
        print(f"select_action - State shape: {state.shape}")
        print(f"select_action - Visual shape: {visual.shape}")

        with torch.no_grad():
            if self.config['ppo']['continuous']:
                mean, std = self.policy_network(state, visual)
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)
            else:
                probs = self.policy_network(state, visual)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)

        print(f"select_action - Action shape: {action.shape}")
        print(f"select_action - Log prob shape: {log_prob.shape}")

        return action.cpu().numpy().flatten(), log_prob.cpu().numpy()

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
            if 'policy_state_dict' in checkpoint:
                self.policy_network.load_state_dict(checkpoint['policy_state_dict'])
            if 'critic_state_dict' in checkpoint:
                self.critic_network.load_state_dict(checkpoint['critic_state_dict'])
            if 'icm_state_dict' in checkpoint:
                self.icm.load_state_dict(checkpoint['icm_state_dict'])
            if 'predictive_model_state_dict' in checkpoint:
                self.predictive_model.load_state_dict(checkpoint['predictive_model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'lr_scheduler_state_dict' in checkpoint:
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            if 'high_level_policy_state_dict' in checkpoint and checkpoint['high_level_policy_state_dict'] is not None:
                self.high_level_policy.load_state_dict(checkpoint['high_level_policy_state_dict'])
            if 'low_level_policy_state_dict' in checkpoint and checkpoint['low_level_policy_state_dict'] is not None:
                self.low_level_policy.load_state_dict(checkpoint['low_level_policy_state_dict'])
            self.logger.info('Model loaded successfully')
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {e}")

    def train(self, env, total_timesteps, save_path):
        episode_rewards = []
        episode_lengths = []
        episode_reward = 0
        episode_length = 0

        observation = env.reset()
        for timestep in range(total_timesteps):
            action, log_prob = self.select_action(observation)
            next_observation, reward, done, _ = env.step(action)
            
            # Ensure state and visual are properly formatted
            state = observation['state']
            visual = observation['visual']
            if isinstance(visual, np.ndarray):
                visual = torch.FloatTensor(visual)
            
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

def train_agents(ppo_agent, env, config, logger):
    total_timesteps = config["num_timesteps"]
    ppo_save_path = config["logging"]["model_save_path"] + "/ppo_trained_model.pth"
    episode_rewards, episode_lengths = ppo_agent.train(env, total_timesteps, ppo_save_path)
    logger.info("PPO training completed and model saved.")
    return episode_rewards, episode_lengths

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, filename='ppo_training.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('PPOTraining')

    try:
        os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

        config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../configs/learning/ppo_config.json'))
        print(f"Loading configuration from: {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)

        logger = CustomLogger("AirSimEnvLogger", log_dir="./logs")
        
        # Initialize AirSim client
        client = airsim.MultirotorClient()
        client.confirmConnection()
        
        # Pass the client to the environment
        env = AirSimEnv(state_dim=config['policy_network']['input_size'], 
                        action_dim=config['policy_network']['output_size'], 
                        config=config, 
                        logger=logger, 
                        tensorboard_log_dir="./logs/tensorboard_logs", 
                        log_enabled=True)
        env.set_client(client)
        env = DummyVecEnv([lambda: env])

        ppo_agent = PPOAgent(config, logger=logger, drone_controller=env.envs[0].drone_controller)
        episode_rewards, episode_lengths = train_agents(ppo_agent, env, config, logger)

        # Evaluate the trained agent
        eval_results = ppo_agent.evaluate(env, num_episodes=5)
        logger.info(f"Evaluation results: {eval_results}")

        # Save episode rewards and lengths to a file
        np.savetxt("episode_rewards.txt", episode_rewards)
        np.savetxt("episode_lengths.txt", episode_lengths)

        logger.info("Training completed and models saved.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")