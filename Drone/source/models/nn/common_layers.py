import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.append(project_root)

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
    def __init__(self, input_channels, height=144, width=256):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2)
        self._to_linear = self.determine_to_linear(input_channels, height, width)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        return x.view(x.size(0), -1)

    def determine_to_linear(self, input_channels, height, width):
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, height, width)
            output = self.forward(dummy_input)
            return output.shape[1]


class ICM(nn.Module):
    def __init__(self, state_dim, action_dim, image_channels):
        super().__init__()
        self.cnn = CNNFeatureExtractor(image_channels)
        self.cnn.determine_to_linear(image_channels, 144, 256)  # Assuming 144x256 input images
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())

        # Dimensions
        self.forward_input_dim = 128 + self.cnn._to_linear + action_dim
        self.inverse_input_dim = 128 + self.cnn._to_linear + action_dim

        # Models
        self.forward_model = nn.Sequential(nn.Linear(self.forward_input_dim, 128), nn.ReLU(), nn.Linear(128, 128))
        self.inverse_model = nn.Sequential(nn.Linear(self.inverse_input_dim, 128), nn.ReLU(), nn.Linear(128, action_dim))

    def forward(self, state, next_state, action, image, next_image):
        state_feat = self.state_encoder(state)
        next_state_feat = self.state_encoder(next_state)
        image_feat = self.cnn(image)
        next_image_feat = self.cnn(next_image)
        
        state_action_feat = torch.cat([state_feat, image_feat, action], dim=1)
        next_state_action_feat = torch.cat([next_state_feat, next_image_feat, action], dim=1)

        action_pred = self.inverse_model(next_state_action_feat)
        next_state_pred = self.forward_model(state_action_feat)

        return state_feat, next_state_feat, action_pred, next_state_pred

    def intrinsic_reward(self, state, next_state, action, image, next_image):
        state_feat, next_state_feat, _, next_state_pred = self.forward(state, next_state, action, image, next_image)
        reward = F.mse_loss(next_state_feat, next_state_pred, reduction='none').mean(dim=1)
        return reward
