import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import torch
from collections import deque, namedtuple
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Custom imports
from Drone.source.envs.airsim_env import AirSimEnv
from Drone.source.models.ppo.ppo_agent import PPOAgent
from Drone.source.utilities.custom_logger import CustomLogger
from Drone.source.utilities.data_processing import DataProcessor
from Drone.source.utilities.visualization import DataVisualizer

def configure_logger(name, log_dir='./logs'):
    """Configure and return a custom logger."""
    logger = CustomLogger(name, log_dir=log_dir).get_logger()
    return logger

def load_config(config_path):
    """Load configuration from a JSON file."""
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    return config

def read_tensorboard_logs(log_dir, scalar_name):
    """Read and return scalar values from TensorBoard logs."""
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    scalar_values = event_acc.Scalars(scalar_name)
    steps = [scalar.step for scalar in scalar_values]
    values = [scalar.value for scalar in scalar_values]
    return steps, values

def visualize_training(writer, epoch, iteration, epoch_rewards, policy_loss, value_loss, total_loss, entropy):
    """Visualize training metrics using TensorBoard."""
    if policy_loss is not None:
        writer.add_scalar('Training/Policy Loss', policy_loss, epoch * 1000 + iteration)
    if value_loss is not None:
        writer.add_scalar('Training/Value Loss', value_loss, epoch * 1000 + iteration)
    if total_loss is not None:
        writer.add_scalar('Training/Total Loss', total_loss, epoch * 1000 + iteration)
    if entropy is not None:
        writer.add_scalar('Training/Entropy', entropy, epoch * 1000 + iteration)
    writer.add_scalar('Training/Epoch Rewards', epoch_rewards, epoch * 1000 + iteration)
    writer.flush()

def train_agent(agent, env, config, logger, writer, scheduler, data_processor, data_visualizer):
    """Train the PPO agent."""
    num_epochs = config['ppo']['n_epochs']
    log_interval = config['logging']['log_interval']
    save_interval = config['model_checkpointing']['checkpoint_interval']
    model_save_path = os.path.join(project_root, config['model_checkpointing']['checkpoint_dir'])
    checkpoint_path = os.path.join(model_save_path, 'ppo_agent_checkpoint.pt')
    early_stopping_patience = config['early_stopping']['patience']
    best_reward = float('-inf')
    patience_counter = 0

    os.makedirs(model_save_path, exist_ok=True)

    start_epoch = 0  # Initialize start_epoch to 0 before conditional check

    # Load checkpoint if exists
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        if 'policy_state_dict' in checkpoint and 'critic_state_dict' in checkpoint and 'optimizer_state_dict' in checkpoint:
            agent.load_model(checkpoint_path)
            start_epoch = checkpoint['epoch'] + 1
            best_reward = checkpoint['best_reward']
            patience_counter = checkpoint['patience_counter']
            logger.info(f"Resuming training from epoch {start_epoch}")
        else:
            logger.warning('Checkpoint does not contain required keys: policy_state_dict, critic_state_dict, optimizer_state_dict')

    for epoch in range(start_epoch, num_epochs):
        state, visual = env.reset()
        done = False
        epoch_rewards = 0
        iteration = 0

        while not done:
            action, log_prob = agent.select_action(state, visual)
            next_state, next_visual, reward, done, _ = env.step(action)
            agent.memory.add(action, state, visual, log_prob, reward, done)
            state, visual = next_state, next_visual
            epoch_rewards += reward

            if done:
                agent.update()

            policy_loss = agent.policy_loss.item() if agent.policy_loss is not None else None
            value_loss = agent.value_loss.item() if agent.value_loss is not None else None
            total_loss = agent.total_loss.item() if agent.total_loss is not None else None
            entropy = agent.entropy.item() if agent.entropy is not None else None

            visualize_training(writer, epoch, iteration, epoch_rewards, policy_loss, value_loss, total_loss, entropy)

            if iteration % log_interval == 0:
                logger.info(f'Epoch {epoch}, Iteration {iteration}: Reward: {epoch_rewards}, Policy Loss: {policy_loss}, Value Loss: {value_loss}, Total Loss: {total_loss}, Entropy: {entropy}')
                writer.flush()

            iteration += 1

        if epoch % save_interval == 0:
            save_path = os.path.join(model_save_path, f"ppo_agent_epoch_{epoch}.pt")
            agent.save_model(save_path)
            torch.save({
                'epoch': epoch,
                'best_reward': best_reward,
                'patience_counter': patience_counter,
                'policy_state_dict': agent.policy_network.state_dict(),
                'critic_state_dict': agent.critic_network.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
            }, checkpoint_path)
            logger.info(f"Checkpoint saved at epoch {epoch}")

        scheduler.step(epoch_rewards)
        if epoch_rewards > best_reward:
            best_reward = epoch_rewards
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    writer.close()

    # Post-training visualization
    tb_log_dir = config['logging']['tensorboard_log_dir']
    scalar_name = 'Training/Epoch Rewards'
    steps, values = read_tensorboard_logs(tb_log_dir, scalar_name)
    df = pd.DataFrame({'steps': steps, 'values': values})
    
    if df.empty:
        logger.warning("DataFrame is empty. Skipping visualization.")
    else:
        try:
            cleaned_df = data_processor.clean_data(df)
            transformed_df = data_processor.transform_data(cleaned_df, {"values": lambda x: x * 2})

            data_visualizer.plot_time_series(transformed_df, x='steps', y='values', title='Transformed Epoch Rewards')
            data_visualizer.plot_histogram(transformed_df, column='values', title='Values Distribution')
            data_visualizer.plot_correlation_matrix(transformed_df, title='Correlation Matrix')
            data_visualizer.plot_scatter(transformed_df, x='steps', y='values', title='Scatter Plot')
        except Exception as e:
            logger.error(f"Visualization error: {e}")

def main():
    config_path = os.path.join(project_root, 'Drone', 'configs', 'learning', 'ppo_config.json')
    config = load_config(config_path)

    logger = configure_logger(__name__, config['logging']['log_dir'])
    writer = SummaryWriter(log_dir=config['logging']['tensorboard_log_dir'])

    target_position = np.array([0, 0, config['environment']['height_target']])
    action_frequency = config['environment']['duration']
    log_enabled = config['logging']['tensorboard']

    env = AirSimEnv(
        state_dim=config['environment']['state_dim'],
        action_dim=config['environment']['action_dim'],
        target_position=target_position,
        action_frequency=action_frequency,
        log_enabled=log_enabled,
        logger=logger,
        tensorboard_log_dir=config['logging']['tensorboard_log_dir'],
        exploration_strategy=config['exploration']['strategy'],
        epsilon=config['exploration']['initial_epsilon'],
        epsilon_decay=config['exploration']['epsilon_decay_rate'],
        min_epsilon=config['exploration']['min_epsilon'],
        temperature=1.0,
        ucb_c=2.0
    )

    device = config['ppo']['device']
    if device == 'auto':    
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    agent = PPOAgent(config, env.action_space.n)  # Pass the number of actions

    optimizer = agent.optimizer
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    data_processor = DataProcessor(logger)
    data_visualizer = DataVisualizer(logger)
    
    train_agent(agent, env, config, logger, writer, scheduler, data_processor, data_visualizer)

if __name__ == '__main__':
    main()
