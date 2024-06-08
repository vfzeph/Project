import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
import sys

# Ensure project root is in the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.append(project_root)

# Assuming these modules are in your project structure:
from Drone.source.models.nn.shared_components import ResidualBlock, AttentionLayer

def configure_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger

logger = configure_logger()

class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )

    def forward(self, x):
        return self.layers(x)

class ICM(nn.Module):
    def __init__(self, state_dim, action_dim, image_channels):
        super().__init__()
        self.cnn = CNNFeatureExtractor(image_channels)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )

        # Dummy input to initialize CNN and get output dimension
        self.cnn(torch.zeros(1, image_channels, 144, 256))

        # Manually set the correct input dimensions based on concatenated feature size
        self.forward_input_dim = 4740  # Corrected based on debug output
        self.inverse_input_dim = 4740  # Corrected based on debug output

        self.forward_model = nn.Linear(self.forward_input_dim, 128)  # Output to some dimension, adjust as needed
        self.inverse_model = nn.Linear(self.inverse_input_dim, action_dim)

    def forward(self, state, next_state, action, image, next_image):
        state_feat = self.state_encoder(state)
        next_state_feat = self.state_encoder(next_state)
        image_feat = self.cnn(image)
        next_image_feat = self.cnn(next_image)

        state_action_feat = torch.cat([state_feat, image_feat, action], dim=1)
        next_state_action_feat = torch.cat([next_state_feat, next_image_feat, action], dim=1)

        action_pred = self.inverse_model(state_action_feat)
        next_state_pred = self.forward_model(next_state_action_feat)

        return state_feat, next_state_feat, action_pred, next_state_pred

    def intrinsic_reward(self, state, next_state, action, image, next_image):
        _, next_state_feat, _, next_state_pred = self.forward(state, next_state, action, image, next_image)
        reward = F.mse_loss(next_state_feat, next_state_pred, reduction='none').mean(dim=1)
        return reward

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    icm = ICM(15, 4, 3)
    test_state = torch.rand(1, 15)
    test_next_state = torch.rand(1, 15)
    test_action = torch.rand(1, 4)
    test_image = torch.rand(1, 3, 144, 256)
    test_next_image = torch.rand(1, 3, 144, 256)

    icm.eval()
    with torch.no_grad():
        output = icm(test_state, test_next_state, test_action, test_image, test_next_image)
        reward = icm.intrinsic_reward(test_state, test_next_state, test_action, test_image, test_next_image)
        logging.info(f"Intrinsic reward: {reward}")