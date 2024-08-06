import os
import sys

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Drone.source.models.ppo.train_ppo_agent import run_training
from Drone.source.utilities.custom_logger import CustomLogger

def main():
    logger = CustomLogger("DriverLog", log_dir="./logs")
    logger.info("Starting the training process")

    try:
        run_training()
    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
    
    logger.info("Training process completed")

if __name__ == "__main__":
    main()