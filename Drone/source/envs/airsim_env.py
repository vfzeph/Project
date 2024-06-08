import json
import logging
import os
import time
import airsim
import numpy as np
import gym
from gym import spaces
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import random
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(project_root)

from threading import Thread, Lock
from collections import deque
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from Drone.source.models.nn.common_layers import ICM  # Import the necessary class

class CustomLogger:
    def __init__(self, name: str, log_dir: str = None):
        self.name = name
        self.log_dir = log_dir
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.INFO)
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            file_path = os.path.join(self.log_dir, f'{self.name}.log')
            file_handler = logging.FileHandler(file_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    def info(self, message: str):
        self.logger.info(message)

def randomize_environment(client):
    weather_params = [
        airsim.WeatherParameter.Rain,
        airsim.WeatherParameter.Enabled
    ]
    weather = random.choice(weather_params)
    client.simSetWeatherParameter(weather, random.uniform(0, 1))

class AirSimEnv(gym.Env):
    def __init__(self, config, logger=None):
        super(AirSimEnv, self).__init__()
        self.config = config  # Store config as an instance attribute
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        self.state_dim = config['environment']['state_dim']
        self.action_dim = config['environment']['action_dim']
        self.target_position = np.array([0, 0, config['environment']['height_target']])
        self.action_frequency = config['environment']['duration']
        self.max_episode_duration = config['environment']['max_env_steps']
        self.exploration_area = config['environment']['exploration_area']
        self.epsilon = config['exploration']['initial_epsilon']
        self.epsilon_decay = config['exploration']['epsilon_decay_rate']
        self.min_epsilon = config['exploration']['min_epsilon']
        self.temperature = 1.0
        self.ucb_c = 2.0
        self.logger = logger
        self.log_enabled = config['logging']['tensorboard']
        self.writer = SummaryWriter(log_dir=config['logging']['tensorboard_log_dir']) if config['logging']['tensorboard'] else None

        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        self.action_counts = np.zeros(self.action_space.shape[0])
        self.action_values = np.zeros(self.action_space.shape[0])
        self.total_steps = 0
        self.prev_action = np.zeros(self.action_dim)
        self.prev_state = None

        # Initialize ICM module
        self.icm = ICM(self.state_dim, self.action_dim, image_channels=config['icm']['image_channels'])
        self.icm_optimizer = optim.Adam(self.icm.parameters(), lr=config['learning_rate'])

        # Randomize environment
        self.randomize_environment()

        # Initialize state normalization parameters
        self.state_means = np.zeros(self.state_dim)
        self.state_stddevs = np.ones(self.state_dim)
        self.state_buffer = deque(maxlen=1000)
        self.normalization_lock = Lock()

        # Start background thread for state normalization
        self.normalization_thread = Thread(target=self._update_state_normalization, daemon=True)
        self.normalization_thread.start()

    def log(self, message: str):
        if self.log_enabled and self.logger:
            self.logger.info(message)
        else:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

    def randomize_environment(self):
        randomize_environment(self.client)

    def reset(self):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
        self.start_time = time.time()
        self.prev_action = np.zeros(self.action_dim)
        self.action_counts = np.zeros(self.action_space.shape[0])
        self.action_values = np.zeros(self.action_space.shape[0])
        self.total_steps = 0
        self.prev_state = None
        self.log("Environment reset and takeoff completed.")
        state = self._get_state()
        image = self._get_image()
        with self.normalization_lock:
            self.state_buffer.append(state)
        return state, image

    def step(self, action):
        action = self._smooth_action(action)
        vx, vy, vz = float(action[0]), float(action[1]), float(action[2])
        self.client.moveByVelocityAsync(vx, vy, vz, self.action_frequency).join()
        time.sleep(self.action_frequency)
        new_state = self._get_state()
        new_image = self._get_image()
        reward = self._compute_reward(new_state, action, new_image)
        done = self._check_done(new_state)
        self.log(f"Action: {action}, Reward: {reward}, Done: {done}")
        self._update_action_values(action, reward)
        with self.normalization_lock:
            self.state_buffer.append(new_state)
        if self.writer:
            self.writer.add_scalar('Reward', reward, self.total_steps)
            self.writer.add_scalar('Epsilon', self.epsilon, self.total_steps)
            self.writer.flush()
        self.total_steps += 1
        return new_state, new_image, reward, done, {}

    def _smooth_action(self, action):
        action = np.tanh(action)
        action = self.prev_action + 0.5 * (action - self.prev_action)
        self.prev_action = action
        return action

    def _get_state(self):
        pose = self.client.simGetVehiclePose().position
        velocity = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        orientation = self.client.simGetVehiclePose().orientation
        angular_velocity = self.client.getMultirotorState().kinematics_estimated.angular_velocity

        position = np.array([pose.x_val, pose.y_val, pose.z_val])
        linear_velocity = np.array([velocity.x_val, velocity.y_val, velocity.z_val])
        orientation_quat = np.array([orientation.x_val, orientation.y_val, orientation.z_val, orientation.w_val])
        angular_velocity = np.array([angular_velocity.x_val, angular_velocity.y_val, angular_velocity.z_val])

        state = np.concatenate((position, linear_velocity, orientation_quat, angular_velocity), axis=0)

        # If the generated state vector has fewer elements than expected, pad it with zeros
        if state.shape[0] < self.state_dim:
            state = np.pad(state, (0, self.state_dim - state.shape[0]), 'constant')

        return self._normalize_state(state)

    def _get_image(self):
        response = self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])[0]
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(response.height, response.width, 3).astype(np.float32) / 255.0
        img_rgb = img_rgb.transpose(2, 0, 1)
        return img_rgb

    def _normalize_state(self, state):
        with self.normalization_lock:
            state = (state - self.state_means) / (self.state_stddevs + 1e-8)
        return state

    def _update_state_normalization(self):
        while True:
            with self.normalization_lock:
                if len(self.state_buffer) >= 1000:
                    states = np.array(self.state_buffer)
                    self.state_means = np.mean(states, axis=0)
                    self.state_stddevs = np.std(states, axis=0)
                    self.state_buffer.clear()

    def _compute_reward(self, state, action, new_image):
        distance_reward = self._compute_distance_reward(state)
        potential_reward = self._compute_potential_reward(state)
        collision_penalty = self._compute_collision_penalty()
        height_penalty = self._compute_height_penalty(state)
        time_penalty = self._compute_time_penalty()
        movement_penalty = self._compute_movement_penalty(action)
        smoothness_penalty = self._compute_smoothness_penalty(action)
        curiosity_reward = self._compute_curiosity_reward(state, action, new_image)
        exploration_bonus = self._compute_exploration_bonus(action)

        reward = (
            distance_reward + potential_reward + collision_penalty +
            height_penalty + time_penalty + movement_penalty +
            smoothness_penalty + curiosity_reward + exploration_bonus
        )
        return reward

    def _compute_distance_reward(self, state):
        position = state[:3]
        distance_to_target = np.linalg.norm(self.target_position - position)
        return -distance_to_target

    def _compute_potential_reward(self, state):
        position = state[:3]
        distance_to_target = np.linalg.norm(self.target_position - position)
        return 1.0 / (1.0 + distance_to_target)

    def _compute_collision_penalty(self):
        collision_info = self.client.simGetCollisionInfo()
        return -self.config['reward_adjustments']['collision_penalty'] if collision_info.has_collided else 0

    def _compute_height_penalty(self, state):
        current_height = state[2]
        height_target = self.target_position[2]
        height_tolerance = self.config['environment']['height_tolerance']
        height_penalty = self.config['environment']['height_penalty']
        return -height_penalty if abs(current_height - height_target) > height_tolerance else 0

    def _compute_time_penalty(self):
        time_penalty = self.config['environment']['movement_penalty']
        return -time_penalty * (time.time() - self.start_time)

    def _compute_movement_penalty(self, action):
        movement_penalty = self.config['environment']['movement_penalty']
        return -movement_penalty * np.linalg.norm(action)

    def _compute_smoothness_penalty(self, action):
        smoothness_penalty = self.config['environment']['smoothness_penalty']
        return -smoothness_penalty * np.linalg.norm(action - self.prev_action)

    def _compute_curiosity_reward(self, state, action, image):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        next_state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        next_image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

        curiosity_reward = self.icm.intrinsic_reward(state_tensor, next_state_tensor, action_tensor, image_tensor, next_image_tensor)
        self.prev_state = state
        return curiosity_reward

    def _compute_exploration_bonus(self, action):
        exploration_bonus = 0.2 * np.linalg.norm(action - self.prev_action)
        return exploration_bonus

    def _update_action_values(self, action, reward):
        action_index = np.argmax(action)
        self.action_counts[action_index] += 1
        n = self.action_counts[action_index]
        value = self.action_values[action_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.action_values[action_index] = new_value

    def _check_done(self, state):
        position = state[:3]
        distance_to_target = np.linalg.norm(self.target_position - position)
        if distance_to_target < 1.0:
            self.log("Target reached.")
            return True
        if time.time() - self.start_time > self.max_episode_duration:
            self.log("Episode timed out.")
            return True
        return False

    def select_action_ucb(self):
        total_counts = np.sum(self.action_counts)
        ucb_values = self.action_values + self.ucb_c * np.sqrt(np.log(total_counts + 1) / (self.action_counts + 1e-5))
        return np.argmax(ucb_values)

    def select_action_softmax(self):
        exp_values = np.exp(self.action_values / self.temperature)
        probabilities = exp_values / np.sum(exp_values)
        return np.random.choice(len(probabilities), p=probabilities)

    def close(self):
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
        self.client.reset()
        if self.logger:
            self.logger.info("Environment closed.")

if __name__ == '__main__':
    config_path = 'Drone/configs/learning/ppo_config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    logger = CustomLogger("AirSimEnvLogger", log_dir="./logs")
    env = DummyVecEnv([lambda: AirSimEnv(config=config, logger=logger)])
    
    policy_kwargs = dict(
        net_arch=[dict(pi=config['policy_network']['hidden_layers'], vf=config['critic_network']['hidden_layers'])],
        activation_fn=torch.nn.ReLU
    )
    
    # List of valid PPO parameters
    valid_ppo_params = [
        'learning_rate', 'n_steps', 'batch_size', 'n_epochs', 'gamma', 'gae_lambda',
        'clip_range', 'clip_range_vf', 'ent_coef', 'vf_coef', 'max_grad_norm', 
        'use_sde', 'sde_sample_freq', 'target_kl', 'tensorboard_log', 'verbose', 'seed', 'device'
    ]
    
    # Filter the valid PPO parameters
    ppo_config = {k: v for k, v in config['ppo'].items() if k in valid_ppo_params}
    
    model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, **ppo_config)
    
    model.learn(total_timesteps=config['num_timesteps'])
    model.save(config['logging']['model_save_path'])
