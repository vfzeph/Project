import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
import sys

# Ensure project root is in the path
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


class AdvancedPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, continuous, hidden_sizes, input_channels):
        super().__init__()
        self.continuous = continuous
        self.cnn = CNNFeatureExtractor(input_channels)
        
        # Initialize CNN on a dummy input to set up dimensions
        dummy_visual = torch.zeros(1, input_channels, 144, 256)
        cnn_output_dim = self.cnn(dummy_visual).shape[1]

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_sizes[0]),
            nn.ReLU()
        )

        # Calculate the combined input dimension for the first policy layer
        combined_input_dim = hidden_sizes[0] + cnn_output_dim
        
        # Define the neural network layers using ModuleList
        self.policy_layers = nn.ModuleList()
        for size in hidden_sizes:
            self.policy_layers.append(nn.Sequential(
                nn.Linear(combined_input_dim, size),
                nn.ReLU()
            ))
            combined_input_dim = size

        # Final layer to output the mean or action probabilities
        if continuous:
            self.mean = nn.Linear(hidden_sizes[-1], action_dim)
            self.log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.action_head = nn.Linear(hidden_sizes[-1], action_dim)

    def forward(self, state, visual_input):
        visual_features = self.cnn(visual_input)
        state_features = self.state_encoder(state)
        x = torch.cat((state_features, visual_features), dim=1)
        
        for layer in self.policy_layers:
            x = layer(x)  # Apply layer transformations

        if self.continuous:
            action_mean = self.mean(x)
            action_std = torch.exp(self.log_std)  # Ensure the standard deviation is positive
            return action_mean, action_std
        else:
            action_probs = F.softmax(self.action_head(x), dim=-1)
            return action_probs

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    state_dim = 15
    action_dim = 4
    hidden_sizes = [256, 256]
    input_channels = 3
    continuous = True

    policy_network = AdvancedPolicyNetwork(state_dim, action_dim, continuous, hidden_sizes, input_channels)
    test_state = torch.rand(1, state_dim)
    test_visual_input = torch.rand(1, input_channels, 144, 256)
    policy_network.eval()

    with torch.no_grad():
        action_output = policy_network(test_state, test_visual_input)
        print(f'Action output: {action_output}')