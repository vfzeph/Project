import os
import sys
import json
import numpy as np
import torch
import airsim
from stable_baselines3.common.vec_env import DummyVecEnv

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Drone.source.models.ppo.ppo_agent import PPOAgent
from Drone.source.envs.airsim_env import AirSimEnv
from Drone.source.utilities.custom_logger import CustomLogger

def run_training():
    # Setup logging
    logger = CustomLogger("PPOAgentTraining", log_dir="./logs")
    logger.info("Starting PPO Agent training")

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

    # Wrap the environment
    env = DummyVecEnv([lambda: env])

    # Initialize agent
    agent = PPOAgent(config, logger=logger, drone_controller=env.envs[0].drone_controller)
    logger.info("PPOAgent initialized")

    # Set up training parameters
    total_timesteps = config['training']['total_timesteps']
    save_path = os.path.join(project_root, 'models', 'trained_ppo_model.pth')

    # Train the agent
    logger.info(f"Starting training for {total_timesteps} timesteps...")
    try:
        episode_rewards, episode_lengths = agent.train(env, total_timesteps, save_path)
        logger.info("Training completed successfully")
        logger.info(f"Total episodes: {len(episode_rewards)}")
        logger.info(f"Average reward: {np.mean(episode_rewards):.2f}")
        logger.info(f"Average episode length: {np.mean(episode_lengths):.2f}")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        return

    # Save the final model
    agent.save_model(save_path)
    logger.info(f"Final model saved to {save_path}")

    # Evaluate the trained agent
    logger.info("Evaluating trained agent...")
    eval_results = agent.evaluate(env, num_episodes=10)
    logger.info(f"Evaluation results: {eval_results}")

    logger.info("PPO Agent training and evaluation completed")

if __name__ == "__main__":
    run_training()