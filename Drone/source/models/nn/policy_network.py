import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
import sys

# Ensure project root is in the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.append(project_root)

from Drone.source.models.nn.common_layers import CNNFeatureExtractor
from Drone.source.models.nn.shared_components import ResidualBlock

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

class AdvancedPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, continuous, hidden_sizes, input_channels):
        super(AdvancedPolicyNetwork, self).__init__()
        self.continuous = continuous
        self.cnn = CNNFeatureExtractor(input_channels)
        
        if self.cnn._to_linear is None:
            raise ValueError("CNNFeatureExtractor _to_linear attribute is None. Ensure determine_to_linear is called properly.")
        
        combined_input_dim = state_dim + self.cnn._to_linear

        self.layers = nn.ModuleList()
        input_dim = combined_input_dim

        for size in hidden_sizes:
            self.layers.append(nn.Linear(input_dim, size))
            input_dim = size

        if continuous:
            self.mean = nn.Linear(hidden_sizes[-1], action_dim)
            self.log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.action_head = nn.Linear(hidden_sizes[-1], action_dim)

        self.init_weights()

    def forward(self, state, visual_input):
        visual_features = self.cnn(visual_input)
        x = torch.cat((state, visual_features), dim=1)  # Concatenate state and visual features
        for layer in self.layers:
            x = F.leaky_relu(layer(x))

        if self.continuous:
            action_mean = self.mean(x)
            action_std = torch.exp(self.log_std)
            return action_mean, action_std
        else:
            action_probs = F.softmax(self.action_head(x), dim=-1)
            return action_probs

    def init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='leaky_relu')
                nn.init.constant_(layer.bias, 0)

if __name__ == "__main__":
    logger.info("Initializing and testing the AdvancedPolicyNetwork.")
    state_dim = 15
    action_dim = 3
    hidden_sizes = [256, 256]
    input_channels = 3  # Assuming RGB images

    network = AdvancedPolicyNetwork(state_dim, action_dim, continuous=True, hidden_sizes=hidden_sizes, input_channels=input_channels)
    test_state_input = torch.rand(1, state_dim)
    test_visual_input = torch.rand(1, input_channels, 144, 256)  # Use the actual input dimensions
    network.eval()
    with torch.no_grad():
        action_output = network(test_state_input, test_visual_input)
        if isinstance(action_output, tuple):
            logger.info(f"Action outputs: mean={action_output[0]}, std={action_output[1]}")
        else:
            logger.info(f"Action probabilities: {action_output}")
