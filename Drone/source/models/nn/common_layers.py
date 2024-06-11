import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.append(project_root)
from Drone.source.models.nn.shared_components import ResidualBlock, AttentionLayer

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
        self._to_linear = None  # This will be initialized on the first forward pass

    def forward(self, x):
        x = self.layers(x)
        if self._to_linear is None:
            self._to_linear = x.numel() // x.shape[0]  # Calculate flat output dimension dynamically
        return x

class ICM(nn.Module):
    def __init__(self, config):
        super().__init__()
        state_dim = config['state_dim']
        action_dim = config['action_dim']
        image_channels = config['image_channels']
        
        self.cnn = CNNFeatureExtractor(image_channels)
        self.cnn(torch.zeros(1, image_channels, 144, 256))  # Properly initialize CNN output dimension

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, config['state_encoder']['hidden_dim']),
            nn.ReLU()
        )

        self.forward_model = nn.Sequential(
            nn.Linear(self.cnn._to_linear + config['state_encoder']['hidden_dim'] + action_dim, config['forward_model']['hidden_dim']),
            nn.ReLU(),
            nn.Linear(config['forward_model']['hidden_dim'], config['state_encoder']['hidden_dim'])
        )

        self.inverse_model = nn.Sequential(
            nn.Linear(self.cnn._to_linear + config['state_encoder']['hidden_dim'] * 2, config['inverse_model']['hidden_dim']),
            nn.ReLU(),
            nn.Linear(config['inverse_model']['hidden_dim'], action_dim)
        )

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
    config = {
        "state_dim": 15,
        "action_dim": 4,
        "image_channels": 3,
        "state_encoder": {"hidden_dim": 128},
        "forward_model": {"hidden_dim": 128},
        "inverse_model": {"hidden_dim": 128}
    }
    icm = ICM(config)
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
