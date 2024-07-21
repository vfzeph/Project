import sys
import os
import json
import airsim
import numpy as np
import time
import gym
from gym import spaces
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import random
from threading import Thread, Lock
from collections import deque
import torch.nn.functional as F

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Drone.source.models.nn.common_layers import ICM
from Drone.source.models.nn.policy_network import AdvancedPolicyNetwork
from Drone.source.models.nn.critic_network import AdvancedCriticNetwork
from Drone.source.models.nn.Predictive_model import PredictiveModel
from Drone.source.envs.drone_controller import DroneController

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

    def error(self, message):
        self.logger.error(message)

def randomize_environment(client):
    weather_params = [
        airsim.WeatherParameter.Rain,
        airsim.WeatherParameter.Fog,
        airsim.WeatherParameter.Dust,
        airsim.WeatherParameter.Snow
    ]
    weather = random.choice(weather_params)
    client.simSetWeatherParameter(weather, random.uniform(0, 1))

class CurriculumLearning:
    def __init__(self, config):
        self.current_difficulty = config['curriculum_learning']['initial_difficulty']
        self.max_difficulty = config['curriculum_learning'].get('max_difficulty', 10)
        self.difficulty_increment = config['curriculum_learning'].get('difficulty_increment', 0.5)
        self.reward_threshold = config['curriculum_learning'].get('reward_threshold', 50)

    def update_difficulty(self, average_reward):
        if average_reward > self.reward_threshold:
            self.current_difficulty = min(self.current_difficulty + self.difficulty_increment, self.max_difficulty)
            self.reward_threshold *= 1.1  # Increase the threshold for the next level

    def get_action_scale(self):
        return 1 + (self.current_difficulty * 1.0)  # Scales from 1 to 11 as difficulty increases

class AirSimEnv(gym.Env):
    class State:
        INITIALIZING = 'initializing'
        RESETTING = 'resetting'
        TAKING_OFF = 'taking_off'
        RUNNING = 'running'
        DONE = 'done'

    def __init__(self, state_dim, action_dim, config, target_position=np.array([0, 0, -10]), action_frequency=1.0, log_enabled=False,
                 exploration_strategy="epsilon_greedy", epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.1,
                 temperature=1.0, ucb_c=2.0, logger=None, tensorboard_log_dir=None):
        super(AirSimEnv, self).__init__()
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.drone_controller = DroneController(self.client)
        self.state_dim = state_dim
        self.max_episode_steps = config.get('max_episode_steps', 1000)
        self.current_step = 0
        self.action_dim = action_dim
        self.logger = logger or CustomLogger("AirSimEnvLogger", log_dir="./logs")
        self.logger.info(f"Initialized with state_dim: {self.state_dim}, action_dim: {self.action_dim}")
        self.target_position = target_position
        self.action_frequency = action_frequency
        self.start_time = None
        self.max_episode_duration = 120
        self.exploration_area = {
            "x_min": -1000, "x_max": 1000,
            "y_min": -1000, "y_max": 1000,
            "z_min": -100, "z_max": 100
        }
        
        self.image_height = config['icm']['image_height']
        self.image_width = config['icm']['image_width']
        self.image_channels = config['icm']['image_channels']
        self.image_size = self.image_height * self.image_width * self.image_channels
        self.total_obs_size = self.state_dim + self.image_size
        
        self.action_space = spaces.Box(low=-10, high=10, shape=(self.action_dim,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'state': spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32),
            'visual': spaces.Box(low=0, high=1, shape=(self.image_height, self.image_width, self.image_channels), dtype=np.float32)
        })
        
        self.log_enabled = log_enabled
        self.exploration_strategy = exploration_strategy
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.position_history = deque(maxlen=10)
        self.min_epsilon = min_epsilon
        self.temperature = temperature
        self.ucb_c = ucb_c
        self.total_steps = 0
        self.prev_action = np.zeros(self.action_dim)
        self.writer = SummaryWriter(log_dir=tensorboard_log_dir) if tensorboard_log_dir else None
        self.prev_state = None
        self.state = None
        self.next_state = None
        self.current_image = None

        self.logger.info("Initializing AirSimEnv")
        self.logger.info(f"Configuration: {json.dumps(config, indent=2)}")

        # Initialize device
        device_config = config['ppo']['device']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device_config == 'auto' else torch.device(device_config)

        self._initialize_icm(config)
        self._initialize_policy_network(config)
        self._initialize_critic_network(config)
        self._initialize_predictive_model(config)

        # Randomize environment
        self.randomize_environment()

        # Initialize state normalization parameters
        self.state_means = np.zeros(self.state_dim)
        self.state_stddevs = np.ones(self.state_dim)
        self.state_buffer = deque(maxlen=1000)
        self.state_buffer_lock = Lock()

        # Start background thread for state normalization
        self.normalization_thread = Thread(target=self._update_state_normalization)
        self.normalization_thread.start()

        # Initialize FSM state
        self.current_fsm_state = self.State.INITIALIZING

        # Initialize reward weights
        self.reward_weights = {
            'distance': 1.0,
            'velocity': 0.1,
            'collision': 5.0,
            'height': 0.5,
            'movement': 0.5,
            'smoothness': 0.1,
            'curiosity': 1.0,
            'exploration': 0.5
        }

        # Initialize curriculum learning
        self.curriculum = CurriculumLearning(config)

        # Initialize action history
        self.action_history = deque(maxlen=5)

    def set_client(self, client):
        self.client = client
        self.drone_controller.client = client
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

    def _initialize_icm(self, config):
        try:
            if 'icm' in config:
                self.icm = ICM(config['icm']).to(self.device)
                self.icm_optimizer = optim.Adam(self.icm.parameters(), lr=config['learning_rate'])
            else:
                self.icm = None
                self.icm_optimizer = None
        except Exception as e:
            self.logger.error(f"Error initializing ICM: {e}")
            raise

    def _initialize_policy_network(self, config):
        try:
            continuous = config['ppo'].get('continuous', True)
            
            policy_config = {
                'image_channels': config['icm']['image_channels'],
                'image_height': config['icm']['image_height'],
                'image_width': config['icm']['image_width'],
                'cnn': config['icm']['cnn'],
                'use_batch_norm': config['policy_network'].get('use_batch_norm', True),
                'use_dropout': config['policy_network'].get('use_dropout', True),
                'dropout_rate': config['policy_network'].get('dropout_rate', 0.2),
                'use_attention': config['policy_network'].get('use_attention', True),
                'num_action_heads': config['policy_network'].get('num_action_heads', 1),
            }
            
            self.policy_network = AdvancedPolicyNetwork(
                self.state_dim, 
                self.action_dim, 
                continuous=continuous,
                hidden_sizes=config['policy_network']['hidden_layers'],
                config=policy_config
            ).to(self.device)
        except KeyError as e:
            self.logger.error(f"Missing key in configuration for policy network: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error initializing policy network: {e}")
            raise

    def _initialize_critic_network(self, config):
        try:
            critic_config = {
                'image_channels': config['icm']['image_channels'],
                'image_height': config['icm']['image_height'],
                'image_width': config['icm']['image_width'],
                'cnn': config['icm']['cnn'],
                'use_batch_norm': config['critic_network'].get('use_batch_norm', True),
                'use_dropout': config['critic_network'].get('use_dropout', True),
                'dropout_rate': config['critic_network'].get('dropout_rate', 0.2),
                'use_attention': config['critic_network'].get('use_attention', True),
            }

            self.critic_network = AdvancedCriticNetwork(
                self.state_dim,
                hidden_sizes=config['critic_network']['hidden_layers'],
                config=critic_config
            ).to(self.device)
        except KeyError as e:
            self.logger.error(f"Missing key in configuration for critic network: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error initializing critic network: {e}")
            raise

    def _initialize_predictive_model(self, config):
        try:
            cnn_config = config['icm']['cnn']
            cnn_config['image_channels'] = config['icm']['image_channels']
            cnn_config['image_height'] = config['icm']['image_height']
            cnn_config['image_width'] = config['icm']['image_width']

            self.predictive_model = PredictiveModel(
                self.state_dim + self.action_dim,
                self.state_dim,
                config['predictive_model']['hidden_layers'],
                cnn_config
            ).to(self.device)
            self.predictive_model_optimizer = optim.Adam(
                self.predictive_model.parameters(),
                lr=config['predictive_model']['learning_rate']
            )
        except KeyError as e:
            self.logger.error(f"Missing key in configuration for predictive model: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error initializing predictive model: {e}")
            raise

    def randomize_environment(self):
        randomize_environment(self.client)

    def reset(self):
        self.current_fsm_state = self.State.RESETTING
        self._reset_env()
        self.current_fsm_state = self.State.TAKING_OFF
        self.drone_controller.takeoff()
        self.current_fsm_state = self.State.RUNNING
        self.start_time = time.time()
        self.prev_action = np.zeros(self.action_dim)
        self.total_steps = 0
        self.current_step = 0
        self.prev_state = None
        self.action_history.clear()
        self.logger.info("Environment reset and takeoff completed.")  # Changed from self.log to self.logger.info
        self.state = self._get_state()
        self.current_image = self._get_image()
        with self.state_buffer_lock:
            self.state_buffer.append(self.state)
        
        observation = {
            'state': self.state,
            'visual': self.current_image
        }
        
        self.logger.info(f"Reset completed. Observation keys: {observation.keys()}")
        self.logger.info(f"State shape: {observation['state'].shape}, Visual shape: {observation['visual'].shape}")
        
        return observation

    def _reset_env(self):
        self.drone_controller.reset()

    def step(self, action):
        if self.current_fsm_state != self.State.RUNNING:
            self.logger.error("Step called while not in RUNNING state.")
            return {'state': np.zeros(self.state_dim), 'visual': np.zeros((self.config['icm']['image_height'], self.config['icm']['image_width'], self.config['icm']['image_channels']))}, 0, True, {}
        
        try:
            # Clip action to be within the drone's capabilities
            action = np.clip(action, -self.config['environment']['action_scale'], self.config['environment']['action_scale'])
            
            vx, vy, vz, yaw_rate = action

            duration = self.config['environment']['min_action_interval']
            
            self.drone_controller.move_by_velocity_z_yaw_rate(vx, vy, -vz, yaw_rate, duration)
            time.sleep(duration)

            new_state = self._get_state()
            new_image = self._get_image()

            reward = self._compute_reward(self.current_state, action, self.current_image, new_state, new_image)
            done = self._check_done(new_state)

            self.logger.info(f"Action: {action}, Velocity: ({vx}, {vy}, {vz}), Yaw Rate: {yaw_rate}, Duration: {duration}, Reward: {reward}, Done: {done}")
            
            info = {
                'position': new_state[:3],
                'velocity': new_state[3:6],
                'orientation': new_state[6:9],
                'angular_velocity': new_state[9:12],
            }

            observation = {
                'state': new_state,
                'visual': new_image
            }

            self.current_state = new_state
            self.current_image = new_image

            return observation, reward, done, info

        except Exception as e:
            self.logger.error(f"Error in step function: {str(e)}")
            raise
        
    def _smooth_action(self, action):
        alpha = 0.7  # Increase this value for more smoothing (0.5 to 0.8 range)
        action = np.tanh(action)  # Apply non-linear transformation
        smoothed_action = alpha * self.prev_action + (1 - alpha) * action
        self.prev_action = smoothed_action
        return smoothed_action

    def _get_state(self):
        position = self.drone_controller.get_position()
        velocity = self.drone_controller.get_velocity()
        orientation = self.drone_controller.get_orientation()
        angular_velocity = self.drone_controller.get_angular_velocity()

        if hasattr(position, 'x_val'):
            position = np.array([position.x_val, position.y_val, position.z_val])
        if hasattr(velocity, 'x_val'):
            velocity = np.array([velocity.x_val, velocity.y_val, velocity.z_val])

        orientation = np.array(orientation)

        state = np.concatenate([
            position,
            velocity,
            orientation,
            angular_velocity
        ])

        state = np.pad(state, (0, max(0, self.state_dim - len(state))))[:self.state_dim]
        
        return state
        
    def _get_image(self):
        responses = self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
        response = responses[0]
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(self.image_height, self.image_width, self.image_channels)
        img_rgb = img_rgb.astype(np.float32) / 255.0
        return img_rgb

    def _normalize_state(self, state):
        if state.shape != self.state_means.shape:
            raise ValueError(f"State shape {state.shape} does not match normalization parameters shape {self.state_means.shape}")
        return (state - self.state_means) / (self.state_stddevs + 1e-8)

    def _update_state_normalization(self):
        while True:
            with self.state_buffer_lock:
                if len(self.state_buffer) >= 1000:
                    states = np.array(self.state_buffer)
                    self.state_means = np.mean(states, axis=0)
                    self.state_stddevs = np.std(states, axis=0)
                    self.state_buffer.clear()

    def _compute_reward(self, state, action, current_image, next_state, next_image):
        try:
            reward_components = {}

            # Distance to target reward
            target_position = self.config['environment'].get('target_position', np.zeros(3))
            current_position = state[:3]
            next_position = next_state[:3]
            current_distance = np.linalg.norm(target_position - current_position)
            next_distance = np.linalg.norm(target_position - next_position)
            reward_components['distance'] = (current_distance - next_distance) * self.config['environment']['reward_scale']

            # Velocity reward
            velocity = next_state[3:6]
            reward_components['velocity'] = -np.linalg.norm(velocity) * self.config['environment']['movement_penalty']

            # Collision penalty
            collision_info = self.drone_controller.get_collision_info()
            reward_components['collision'] = self.config['environment']['collision_penalty'] if collision_info.has_collided else 0

            # Height penalty
            current_height = current_position[2]
            height_target = self.config['environment']['height_target']
            height_tolerance = self.config['environment']['height_tolerance']
            reward_components['height'] = self.config['environment']['height_penalty'] if abs(current_height - height_target) > height_tolerance else 0

            # Smoothness penalty
            reward_components['smoothness'] = self.config['environment']['smoothness_penalty'] * np.linalg.norm(action - self.prev_action) if hasattr(self, 'prev_action') else 0
            self.prev_action = action

            # Reward for larger movements
            movement_magnitude = np.linalg.norm(action[:3])
            reward_components['movement'] = movement_magnitude * self.config['environment']['large_movement_reward']

            # Penalize staying in the same position
            position_change = np.linalg.norm(next_position - current_position)
            reward_components['stationary'] = -self.config['environment']['stationary_penalty'] if position_change < 0.1 else 0

            # Penalize revisiting recent positions
            reward_components['revisit'] = sum(-self.config['environment']['revisit_penalty'] 
                                            for pos in self.position_history 
                                            if np.linalg.norm(next_position - pos) < 1.0)

            # Update position history
            self.position_history.append(next_position)

            # Task completion reward (if applicable)
            reward_components['task_completion'] = self.config['environment']['task_completion_reward'] if next_distance < self.config['environment']['goal_threshold'] else 0

            # Curiosity-driven reward (based on image difference)
            if self.config['environment'].get('use_curiosity_reward', False):
                image_diff = np.mean(np.abs(next_image - current_image))
                reward_components['curiosity'] = image_diff * self.config['environment']['curiosity_reward_scale']

            # Apply scaling factors
            for key in reward_components:
                reward_components[key] *= self.config['environment'].get(f'{key}_reward_scale', 1.0)

            # Compute total reward
            total_reward = sum(reward_components.values())

            self.logger.info(f"Reward breakdown - {', '.join([f'{k}: {v:.2f}' for k, v in reward_components.items()])}, Total: {total_reward:.2f}")

            return total_reward

        except Exception as e:
            self.logger.error(f"Error in _compute_reward: {str(e)}")
            raise
    def _compute_distance_reward(self, state):
        position = state[:3]
        distance_to_target = np.linalg.norm(self.target_position - position)
        return -distance_to_target

    def _compute_velocity_reward(self, state):
        velocity = state[3:6]
        return -np.linalg.norm(velocity)  # Penalize high velocities

    def _compute_collision_penalty(self):
        collision_info = self.drone_controller.get_collision_info()
        return -50 if collision_info.has_collided else 0

    def _compute_height_penalty(self, state):
        current_height = state[2]
        height_target = self.target_position[2]
        height_tolerance = 1.0
        height_penalty = 1.0
        return -height_penalty if abs(current_height - height_target) > height_tolerance else 0

    def _compute_movement_penalty(self, action):
        movement_penalty = 0.1
        return -movement_penalty * np.linalg.norm(action)

    def _compute_smoothness_penalty(self, action):
        smoothness_penalty = 0.1
        return -smoothness_penalty * np.linalg.norm(action - self.prev_action)

    def _compute_curiosity_reward(self, state, action, current_image, next_state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(self.device)
        action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        current_image_tensor = torch.tensor(current_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)
        next_image_tensor = torch.tensor(self.current_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)

        curiosity_reward = self.icm.intrinsic_reward(state_tensor, next_state_tensor, action_tensor, current_image_tensor, next_image_tensor)
        return curiosity_reward.item()
        
    def _compute_exploration_bonus(self, state):
        if self.prev_state is not None:
            distance_traveled = np.linalg.norm(state[:3] - self.prev_state[:3])
            exploration_bonus = 0.5 * distance_traveled
        else:
            exploration_bonus = 0
        
        self.prev_state = state.copy()
        return exploration_bonus

    def _check_done(self, state):
        position = state[:3]
        distance_to_target = np.linalg.norm(self.target_position - position)
        collision_info = self.drone_controller.get_collision_info()

        if distance_to_target < 1.0:
            self.logger.info("Target reached.")
            return True
        if collision_info.has_collided:
            self.logger.info("Collision detected.")
            return True
        if self.current_step >= self.max_episode_steps:
            self.logger.info("Maximum episode steps reached.")
            return True
        return False
    
    def get_goal(self):
        return self.target_position
    
    def close(self):
        self.drone_controller.land()
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
        self.client.reset()
        if self.writer:
            self.writer.close()
        if self.logger:
            self.logger.info("Environment closed.")

    def update_curriculum(self, average_reward):
        self.curriculum.update_difficulty(average_reward)
        self.logger.info(f"Updated curriculum difficulty to {self.curriculum.current_difficulty}")

    def _train_predictive_model(self, state, action, next_state, image):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(self.device)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(self.device)
        image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)

        predicted_next_state = self.predictive_model(state_tensor, action_tensor, image_tensor)
        loss = F.mse_loss(predicted_next_state, next_state_tensor)

        self.predictive_model_optimizer.zero_grad()
        loss.backward()
        self.predictive_model_optimizer.step()

        return loss.item()

if __name__ == '__main__':
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'learning', 'ppo_config.json')
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    logger = CustomLogger("AirSimEnvLogger", log_dir="./logs")
    env = AirSimEnv(state_dim=config['policy_network']['input_size'], 
                    action_dim=config['policy_network']['output_size'], 
                    config=config, 
                    logger=logger, 
                    tensorboard_log_dir="./logs/tensorboard_logs", 
                    log_enabled=True)
    try:
        observation = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = env.action_space.sample()  # Random action for testing
            observation, reward, done, info = env.step(action)
            episode_reward += reward
            if done:
                logger.info(f"Episode finished. Total reward: {episode_reward}")
                env.update_curriculum(episode_reward)  # Update curriculum after each episode
                observation = env.reset()
                episode_reward = 0
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
    finally:
        env.close()