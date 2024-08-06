import os
import sys

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json
import numpy as np
import torch
import airsim

from Drone.source.models.ppo.ppo_agent import PPOAgent
from Drone.source.envs.airsim_env import AirSimEnv
from Drone.source.utilities.custom_logger import CustomLogger

def test_ppo_agent():
    # Setup logging
    logger = CustomLogger("TestPPOAgent", log_dir="./logs")
    logger.info("Starting PPO Agent test")

    # Load configuration
    config_path = os.path.join(project_root, 'Drone', 'configs', 'learning', 'ppo_config.json')
    logger.info(f"Attempting to load config from: {config_path}")
    
    if not os.path.exists(config_path):
        logger.error(f"Config file not found at {config_path}")
        return

    with open(config_path, 'r') as f:
        config = json.load(f)
    logger.info(f"Loaded configuration from {config_path}")

    # Initialize AirSim client
    client = airsim.MultirotorClient()
    client.confirmConnection()
    logger.info("AirSim client connection confirmed")

    # Initialize environment
    env = AirSimEnv(
        state_dim=config['policy_network']['input_size'], 
        action_dim=config['policy_network']['output_size'], 
        config=config, 
        logger=logger, 
        tensorboard_log_dir="./logs/tensorboard_logs", 
        log_enabled=True
    )
    logger.info("AirSimEnv initialized")

    # Initialize agent
    agent = PPOAgent(config, logger=logger, drone_controller=env.drone_controller)
    logger.info("PPOAgent initialized")

    # Test select_action
    logger.info("Testing select_action...")
    observation = env.reset()
    action, log_prob = agent.select_action(observation)
    logger.info(f"Action shape: {action.shape}, Log prob shape: {log_prob.shape}")

    # Test basic training loop
    logger.info("Testing basic training loop...")
    total_timesteps = 1000  # A small number for testing
    save_path = os.path.join(project_root, 'models', 'test_ppo_model.pth')

    try:
        episode_rewards, episode_lengths = agent.train(env, total_timesteps, save_path)
        logger.info(f"Training completed. Total episodes: {len(episode_rewards)}")
        logger.info(f"Average reward: {np.mean(episode_rewards):.2f}")
        logger.info(f"Average episode length: {np.mean(episode_lengths):.2f}")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        return

    # Test model saving and loading
    logger.info("Testing model saving and loading...")
    agent.save_model(save_path)
    try:
        agent.load_model(save_path)
        logger.info("Model saved and loaded successfully.")
    except Exception as e:
        logger.error(f"Error in saving or loading model: {e}")

    # Test evaluation
    logger.info("Testing evaluation...")
    eval_results = agent.evaluate(env, num_episodes=5)
    logger.info(f"Evaluation results: {eval_results}")

    logger.info("PPO Agent test completed")

if __name__ == "__main__":
    test_ppo_agent()