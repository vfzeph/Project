import os
import sys
import json
import logging
import numpy as np
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from stable_baselines3.common.vec_env import DummyVecEnv
from Drone.source.learning.curriculum_learning import CurriculumLearning
from Drone.source.learning.hierarchical_rl import HighLevelPolicy, LowLevelPolicy, HierarchicalRLAgent
from Drone.source.learning.multi_agent_cooperation import MultiAgentCooperation

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.append(project_root)

from Drone.source.envs.airsim_env import AirSimEnv
from Drone.source.models.nn.policy_network import AdvancedPolicyNetwork
from Drone.source.models.nn.critic_network import AdvancedCriticNetwork
from Drone.source.models.ppo.ppo_utils import compute_gae, normalize
from Drone.source.models.nn.common_layers import ICM
from Drone.source.utilities.custom_logger import CustomLogger

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
            self.actions.append(np.atleast_1d(action))
            self.states.append(np.atleast_2d(state))
            self.visuals.append(np.atleast_3d(visual))
            self.log_probs.append(log_prob)
            self.rewards.append(reward if reward is not None else 0.0)
            self.dones.append(done)
            self.goals.append(np.atleast_1d(goal))

        def get_tensors(self, device):
            return (
                torch.tensor(np.vstack(self.actions), device=device).float(),
                torch.tensor(np.vstack(self.states), device=device).float(),
                torch.tensor(np.vstack(self.visuals), device=device).float(),
                torch.tensor(np.array(self.log_probs), device=device).float(),
                torch.tensor(np.array(self.rewards), device=device).float(),
                torch.tensor(np.array(self.dones), device=device).bool(),
                torch.tensor(np.vstack(self.goals), device=device).float()
            )

    def __init__(self, config):
        self.config = config
        self.state_dim = config['policy_network']['input_size']
        self.action_dim = config['policy_network']['output_size']
        self.action_space = config['policy_network']['output_size']
        self.logger = logging.getLogger(__name__)

        device_config = config['ppo']['device']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device_config == 'auto' else torch.device(device_config)
        self.logger.info(f"Using device: {self.device}")

        self.policy_network = AdvancedPolicyNetwork(
            config['policy_network']['input_size'], 
            config['policy_network']['output_size'], 
            continuous=True, 
            hidden_sizes=config['policy_network']['hidden_layers'],
            input_channels=3  # Assuming RGB images
        ).to(self.device)

        self.critic_network = AdvancedCriticNetwork(
            config['critic_network']['input_size'], 
            hidden_sizes=config['critic_network']['hidden_layers'],
            input_channels=3  # Assuming RGB images
        ).to(self.device)

        self.icm = ICM(config['icm']).to(self.device)

        self.optimizer = optim.Adam(
            list(self.policy_network.parameters()) + list(self.critic_network.parameters()) + list(self.icm.parameters()), 
            lr=config['ppo']['learning_rate']
        )

        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        self.gamma = config['ppo']['gamma']
        self.tau = config['ppo']['gae_lambda']
        self.epsilon = config['ppo']['clip_range']
        self.k_epochs = config['ppo']['n_epochs']
        self.clip_grad = config['ppo']['max_grad_norm']
        self.continuous = True  # Assuming continuous action space for simplicity
        self.writer = SummaryWriter(config['ppo']['tensorboard_log'])
        self.memory = self.Memory()

        self.initial_epsilon = config['exploration']['initial_epsilon']
        self.epsilon_decay_rate = config['exploration']['epsilon_decay_rate']
        self.min_epsilon = config['exploration']['min_epsilon']
        self.epsilon = self.initial_epsilon

        self.entropy_coef = config['ppo']['ent_coef']
        self.icm_weight = config['ppo']['icm_weight']

        if config['hrl']['use_hierarchical']:
            self.high_level_policy = HighLevelPolicy(
                config['hrl']['high_level_policy']['input_size'],
                config['hrl']['sub_goal_dim'],
                config['hrl']['high_level_policy']['hidden_layers']
            ).to(self.device)
            self.low_level_policy = LowLevelPolicy(
                config['policy_network']['input_size'] + config['hrl']['sub_goal_dim'],
                config['policy_network']['output_size'],
                config['policy_network']['hidden_layers']
            ).to(self.device)
            self.hrl_agent = HierarchicalRLAgent(self.high_level_policy, self.low_level_policy)

        if config['curriculum_learning']['use_curriculum']:
            difficulty_increment = config['curriculum_learning']['difficulty_increment']
            difficulty_threshold = config['curriculum_learning']['difficulty_threshold']
            self.curriculum = CurriculumLearning(config, difficulty_increment, difficulty_threshold)

        if config['multi_agent']['use_multi_agent']:
            self.multi_agent_cooperation = MultiAgentCooperation(
                num_agents=config['multi_agent']['num_agents'],
                state_dim=config['policy_network']['input_size'],
                action_dim=config['policy_network']['output_size'],
                hidden_layers=config['policy_network']['hidden_layers']
            )

    def select_action(self, state, visual, goal=None):
        state = np.atleast_1d(state)
        self.logger.debug(f"Original state shape: {state.shape}")

        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        visual_tensor = torch.from_numpy(visual).float().unsqueeze(0).to(self.device)
        self.logger.debug(f"Corrected state tensor shape: {state_tensor.shape}")
        self.logger.debug(f"Visual tensor shape: {visual_tensor.shape}")

        with torch.cuda.amp.autocast(enabled=self.device.type == 'cuda'):
            if self.continuous:
                action_mean, action_std_log = self.policy_network(state_tensor, visual_tensor)
                action_std = torch.exp(action_std_log)
                dist = torch.distributions.Normal(action_mean, action_std)
                action = dist.sample()
                action_log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
            else:
                action_probs = self.policy_network(state_tensor, visual_tensor)
                dist = torch.distributions.Categorical(probs=action_probs)
                action = dist.sample()
                action_log_prob = dist.log_prob(action)

        action = action.cpu().numpy().flatten()
        action_log_prob = action_log_prob.detach().cpu().numpy().flatten()

        if np.random.rand() < self.epsilon:
            action_index = np.random.randint(self.action_space)
            self.logger.info(f"Exploration: Random action {action_index} selected.")
        else:
            action_value = action[0]
            action_index = int(np.clip(np.round(action_value).astype(int), 0, self.action_space - 1))
            self.logger.info(f"Exploitation: Action {action_index} selected.")

        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay_rate)

        self.memory.add(action_index, state, visual, action_log_prob, 0, False, goal)
        return action_index, action_log_prob

    def update(self):
        if len(self.memory.actions) < self.config['ppo']['batch_size']:
            return  # Skip update if not enough samples in the buffer

        actions, states, visuals, log_probs, rewards, dones, goals = self.memory.get_tensors(self.device)

        next_value = self.critic_network(states[-1:], visuals[-1:]).detach()
        returns, advantages = compute_gae(next_value, rewards, dones, self.critic_network(states, visuals), self.gamma, self.tau)

        if self.config['advanced_training_techniques']['normalize_advantages']:
            returns = normalize(returns).view(-1, 1)
            advantages = normalize(advantages).view(-1, 1)

        old_log_probs = log_probs.detach()

        for epoch in range(self.k_epochs):
            with torch.cuda.amp.autocast(enabled=self.device.type == 'cuda'):
                new_log_probs, state_values, entropy = self.evaluate(states, visuals, actions)
                ratios = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages
                self.policy_loss = -(torch.min(surr1, surr2)).mean()

                self.value_loss = nn.functional.mse_loss(state_values, returns)
                
                intrinsic_rewards = self.icm.intrinsic_reward(states, states[1:], actions, visuals, visuals[1:])
                self.total_loss = self.policy_loss + self.config['ppo']['vf_coef'] * self.value_loss - self.entropy_coef * entropy + self.icm_weight * intrinsic_rewards.mean()

            self.optimizer.zero_grad()
            if self.scaler:
                self.scaler.scale(self.total_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.clip_grad)
                torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(), self.clip_grad)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.total_loss.backward()
                self.optimizer.step()

            if epoch % (self.k_epochs // 2) == 0:
                self.logger.info(f'Epoch {epoch+1}/{self.k_epochs}, Policy Loss: {self.policy_loss.item()}, Value Loss: {self.value_loss.item()}, Total Loss: {self.total_loss.item()}, Entropy: {entropy.item()}')
                self.writer.add_scalar('Training/Policy Loss', self.policy_loss.item(), epoch)
                self.writer.add_scalar('Training/Value Loss', self.value_loss.item(), epoch)
                self.writer.add_scalar('Training/Total Loss', self.total_loss.item(), epoch)
                self.writer.add_scalar('Training/Entropy', entropy.item(), epoch)

        self.memory.reset()

    def evaluate(self, states, visuals, actions):
        with torch.cuda.amp.autocast(enabled=self.device.type == 'cuda'):
            if self.continuous:
                action_means, action_std_logs = self.policy_network(states, visuals)
                action_stds = torch.exp(action_std_logs)
                dist = torch.distributions.Normal(action_means, action_stds)
                log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
                entropy = dist.entropy().mean()
            else:
                action_probs = self.policy_network(states, visuals)
                dist = torch.distributions.Categorical(probs=action_probs)
                log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()

            state_values = self.critic_network(states, visuals).squeeze(1)
        return log_probs, state_values, entropy

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy_state_dict': self.policy_network.state_dict(),
            'critic_state_dict': self.critic_network.state_dict(),
            'icm_state_dict': self.icm.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'high_level_policy_state_dict': self.high_level_policy.state_dict() if hasattr(self, 'high_level_policy') else None,
            'low_level_policy_state_dict': self.low_level_policy.state_dict() if hasattr(self, 'low_level_policy') else None
        }, path)
        self.logger.info(f'Model saved at {path}')

    def load_model(self, path):
        checkpoint = torch.load(path)
        if 'policy_state_dict' in checkpoint and 'critic_state_dict' in checkpoint and 'optimizer_state_dict' in checkpoint:
            def rename_keys(state_dict, old_prefix, new_prefix):
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith(old_prefix):
                        new_key = new_prefix + k[len(old_prefix):]
                        new_state_dict[new_key] = v
                    else:
                        new_state_dict[k] = v
                return new_state_dict

            checkpoint['policy_state_dict'] = rename_keys(checkpoint['policy_state_dict'], 'cnn.convs.', 'cnn.layers.')
            checkpoint['critic_state_dict'] = rename_keys(checkpoint['critic_state_dict'], 'cnn.convs.', 'cnn.layers.')

            self.policy_network.load_state_dict(checkpoint['policy_state_dict'])
            self.critic_network.load_state_dict(checkpoint['critic_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'high_level_policy_state_dict' in checkpoint and checkpoint['high_level_policy_state_dict'] is not None:
                self.high_level_policy.load_state_dict(checkpoint['high_level_policy_state_dict'])
            if 'low_level_policy_state_dict' in checkpoint and checkpoint['low_level_policy_state_dict'] is not None:
                self.low_level_policy.load_state_dict(checkpoint['low_level_policy_state_dict'])
            self.logger.info('Model loaded')
        else:
            self.logger.warning('Checkpoint does not contain required keys: policy_state_dict, critic_state_dict, optimizer_state_dict')

def train_ppo_agent(agent, env, total_timesteps, save_path):
    try:
        for num_timesteps in range(total_timesteps):
            state, visual = env.reset()
            done = False
            while not done:
                goal = agent.high_level_policy(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(agent.device)).cpu().detach().numpy().flatten() if agent.config['hrl']['use_hierarchical'] else None
                action, log_prob = agent.select_action(state, visual, goal)
                next_state, next_visual, reward, done, _ = env.step(action)
                agent.memory.add(action, state, visual, log_prob, reward, done, goal)
                state, visual = next_state, next_visual
                if done:
                    agent.update()
                    agent.save_model(save_path)  # Save model at the end of each episode
    except Exception as e:
        agent.logger.error(f"An error occurred during training: {e}")
    finally:
        agent.save_model(save_path)  # Final save after training

def train_agents(ppo_agent, env, config, logger, writer, scheduler, data_processor, data_visualizer):
    total_timesteps = config["num_timesteps"]
    ppo_save_path = config["logging"]["model_save_path"] + "/ppo_trained_model.pth"
    train_ppo_agent(ppo_agent, env, total_timesteps, ppo_save_path)
    logger.info("PPO training completed and model saved.")

if __name__ == '__main__':
    try:
        os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

        config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../configs/learning/ppo_config.json'))
        print(f"Loading configuration from: {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)

        logger = CustomLogger("AirSimEnvLogger", log_dir="./logs")
        env = AirSimEnv(config=config, logger=logger, tensorboard_log_dir="./logs/tensorboard_logs", log_enabled=True)
        env = DummyVecEnv([lambda: env])

        ppo_agent = PPOAgent(config)

        # Initialize other components if necessary
        writer = SummaryWriter(config["logging"]["tensorboard_log_dir"])
        scheduler = None  # Define your scheduler if required
        data_processor = None  # Define your data processor if required
        data_visualizer = None  # Define your data visualizer if required

        train_agents(ppo_agent, env, config, logger, writer, scheduler, data_processor, data_visualizer)
        logger.info("Training completed and models saved.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
