import json
import airsim
import numpy as np
import time
import os
import gym
from gym import spaces
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import random
from threading import Thread
from collections import deque

from Drone.source.models.nn.common_layers import ICM  # Import the necessary class
from Drone.source.models.nn.policy_network import AdvancedPolicyNetwork
from Drone.source.models.nn.critic_network import AdvancedCriticNetwork

# Custom Logger
class CustomLogger:
    def __init__(self, name, log_dir=None):
        self.name = name
        self.log_dir = log_dir
        self._setup_logger()

    def _setup_logger(self):
        import logging
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(self.log_dir, f'{self.name}.log'))
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def info(self, message):
        self.logger.info(message)

# Domain Randomization
def randomize_environment(client):
    weather_params = [
        airsim.WeatherParameter.Rain,
        airsim.WeatherParameter.Enabled
    ]
    weather = random.choice(weather_params)
    client.simSetWeatherParameter(weather, random.uniform(0, 1))

class AirSimEnv(gym.Env):
    def __init__(self, state_dim, action_dim, config, target_position=np.array([0, 0, -10]), action_frequency=1.0, log_enabled=False,
                 exploration_strategy="epsilon_greedy", epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.1,
                 temperature=1.0, ucb_c=2.0, logger=None, tensorboard_log_dir=None):
        super(AirSimEnv, self).__init__()
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.target_position = target_position
        self.action_frequency = action_frequency
        self.start_time = None
        self.max_episode_duration = 120
        self.exploration_area = {
            "x_min": -1000, "x_max": 1000,
            "y_min": -1000, "y_max": 1000,
            "z_min": -100, "z_max": 100
        }
        self.action_space = spaces.Discrete(len([
            (10, 0, 0), (-10, 0, 0),  # Slower Forward, Backward
            (30, 0, 0), (-30, 0, 0),  # Faster Forward, Backward
            (0, 10, 0), (0, -10, 0),  # Slower Right, Left
            (0, 30, 0), (0, -30, 0),  # Faster Right, Left
            (0, 0, 10), (0, 0, -10),  # Slower Up, Down
            (0, 0, 30), (0, 0, -30),  # Faster Up, Down
            (10, 10, 0), (-10, -10, 0),  # Diagonal movements
            (10, -10, 0), (-10, 10, 0),  # Diagonal movements
            (0, 0, 0)  # Hover
        ]))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
        self.log_enabled = log_enabled
        self.exploration_strategy = exploration_strategy
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.temperature = temperature
        self.ucb_c = ucb_c
        self.action_counts = np.zeros(self.action_space.n)
        self.action_values = np.zeros(self.action_space.n)
        self.total_steps = 0
        self.prev_action = np.array([0, 0, 0])
        self.logger = logger
        self.writer = SummaryWriter(log_dir=tensorboard_log_dir) if tensorboard_log_dir else None
        self.prev_state = None  # Used for curiosity-based reward

        # Initialize device
        device_config = config['ppo']['device']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device_config == 'auto' else torch.device(device_config)

        # Initialize ICM module
        self.icm = ICM(config['icm']).to(self.device)
        self.icm_optimizer = optim.Adam(self.icm.parameters(), lr=1e-3)

        # Initialize Policy and Critic Networks
        self.policy_network = AdvancedPolicyNetwork(
            state_dim, action_dim, continuous=config['ppo']['continuous'],
            hidden_sizes=config['policy_network']['hidden_layers'], 
            input_channels=config['icm']['image_channels'],
            use_attention=True
        ).to(self.device)

        self.critic_network = AdvancedCriticNetwork(
            state_dim, hidden_sizes=config['critic_network']['hidden_layers'],
            input_channels=config['icm']['image_channels']
        ).to(self.device)

        # Randomize environment
        self.randomize_environment()

        # Initialize state normalization parameters
        self.state_means = np.zeros(self.state_dim)
        self.state_stddevs = np.ones(self.state_dim)
        self.state_buffer = deque(maxlen=1000)

        # Start background thread for state normalization
        self.normalization_thread = Thread(target=self._update_state_normalization)
        self.normalization_thread.start()

    def log(self, message):
        if self.log_enabled:
            if self.logger:
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
        self.prev_action = np.array([0, 0, 0])
        self.action_counts = np.zeros(self.action_space.n)
        self.action_values = np.zeros(self.action_space.n)
        self.total_steps = 0
        self.prev_state = None
        self.log("Environment reset and takeoff completed.")
        state = self._get_state()
        image = self._get_image()
        self.state_buffer.append(state)
        return state, image

    def step(self, action_index):
        if self.exploration_strategy == "ucb":
            action_index = self.select_action_ucb()
        elif self.exploration_strategy == "softmax":
            action_index = self.select_action_softmax()
        elif self.exploration_strategy == "epsilon_greedy":
            if random.random() < self.epsilon:
                action_index = np.random.randint(self.action_space.n)
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        action = self._map_action(action_index)
        action = action + np.random.normal(0, 0.1, size=action.shape)
        # Variable duration based on action magnitude
        duration = (0.2 if np.linalg.norm(action) > 20 else 0.5) / self.action_frequency
        action = self._smooth_action(action)
        vx, vy, vz = float(action[0]), float(action[1]), float(action[2])
        self.client.moveByVelocityAsync(vx, vy, vz, duration).join()
        time.sleep(duration)
        new_state = self._get_state()
        new_image = self._get_image()
        reward = self._compute_reward(new_state, action, new_image)
        done = self._check_done(new_state)
        self.log(f"Action: {action_index}, Reward: {reward}, Done: {done}")
        self._update_action_values(action_index, reward)
        self.state_buffer.append(new_state)
        if self.writer:
            self.writer.add_scalar('Reward', reward, self.total_steps)
            self.writer.add_scalar('Epsilon', self.epsilon, self.total_steps)
            self.writer.flush()
        return new_state, new_image, reward, done, {}

    def _map_action(self, action_index):
        actions = [
            (10, 0, 0), (-10, 0, 0),  # Slower Forward, Backward
            (30, 0, 0), (-30, 0, 0),  # Faster Forward, Backward
            (0, 10, 0), (0, -10, 0),  # Slower Right, Left
            (0, 30, 0), (0, -30, 0),  # Faster Right, Left
            (0, 0, 10), (0, 0, -10),  # Slower Up, Down
            (0, 0, 30), (0, 0, -30),  # Faster Up, Down
            (10, 10, 0), (-10, -10, 0),  # Diagonal movements
            (10, -10, 0), (-10, 10, 0),  # Diagonal movements
            (0, 0, 0)  # Hover
        ]
        return np.array(actions[action_index])

    def _smooth_action(self, action):
        action = np.tanh(action)  # Apply non-linear transformation (hyperbolic tangent)
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
        if state.shape[0] < 15:
            state = np.pad(state, (0, 15 - state.shape[0]), 'constant', constant_values=(0, 0))
        return self._normalize_state(state)

    def _get_image(self):
        response = self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])[0]
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(response.height, response.width, 3).astype(np.float32) / 255.0  # Normalize image
        img_rgb = img_rgb.transpose(2, 0, 1)  # Convert to [channels, height, width] format
        return img_rgb

    def _normalize_state(self, state):
        """Normalize state values to improve learning stability."""
        if state.shape != self.state_means.shape or state.shape != self.state_stddevs.shape:
            raise ValueError(f"Shape mismatch: state {state.shape}, state_means {self.state_means.shape}, state_stddevs {self.state_stddevs.shape}")
        state = (state - self.state_means) / (self.state_stddevs + 1e-8)
        return state

    def _update_state_normalization(self):
        while True:
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
        return -50 if collision_info.has_collided else 0

    def _compute_height_penalty(self, state):
        current_height = state[2]
        height_target = self.target_position[2]
        height_tolerance = 1.0
        height_penalty = 1.0
        return -height_penalty if abs(current_height - height_target) > height_tolerance else 0

    def _compute_time_penalty(self):
        time_penalty = 0.01
        return -time_penalty * (time.time() - self.start_time)

    def _compute_movement_penalty(self, action):
        movement_penalty = 0.1
        return -movement_penalty * np.linalg.norm(action)

    def _compute_smoothness_penalty(self, action):
        smoothness_penalty = 0.1
        return -smoothness_penalty * np.linalg.norm(action - self.prev_action)

    def _compute_curiosity_reward(self, state, action, image):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        next_state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(self.device)
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(self.device)
        next_image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(self.device)

        curiosity_reward = self.icm.intrinsic_reward(state_tensor, next_state_tensor, action_tensor, image_tensor, next_image_tensor)
        self.prev_state = state
        return curiosity_reward

    def _compute_exploration_bonus(self, action):
        exploration_bonus = 0.1 * np.linalg.norm(action - self.prev_action)
        return exploration_bonus

    def _update_action_values(self, action_index, reward):
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
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'learning', 'ppo_config.json')
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    logger = CustomLogger("AirSimEnvLogger", log_dir="./logs")
    env = AirSimEnv(state_dim=15, action_dim=4, config=config, logger=logger, tensorboard_log_dir="./logs/tensorboard_logs", log_enabled=True)
    state, image = env.reset()
    done = False
    while not done:
        action = np.random.randint(env.action_space.n)  # Random action for testing
        state, image, reward, done, _ = env.step(action)
        if done:
            state, image = env.reset()
    env.close()
