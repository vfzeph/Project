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
        self.output_dim = None  # This will be initialized on the first forward pass

    def forward(self, x):
        x = self.layers(x)
        return x

    def determine_to_linear(self):
        dummy_input = torch.zeros(1, 3, 144, 256)
        self.output_dim = self(dummy_input).shape[1]

class AdvancedCriticNetwork(nn.Module):
    def __init__(self, state_dim, hidden_sizes, dropout_rate=0.2, input_channels=3, use_attention=True):
        super(AdvancedCriticNetwork, self).__init__()
        self.cnn = CNNFeatureExtractor(input_channels)
        self.cnn.determine_to_linear()  # Properly initialize CNN output dimension

        combined_input_dim = state_dim + self.cnn.output_dim
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.attention_layers = nn.ModuleList() if use_attention else None

        for size in hidden_sizes:
            self.layers.append(nn.Linear(combined_input_dim, size))
            self.batch_norms.append(nn.BatchNorm1d(size))
            self.dropouts.append(nn.Dropout(dropout_rate))
            if use_attention:
                self.attention_layers.append(AttentionLayer(size, size))
            combined_input_dim = size

        self.residual_block = ResidualBlock(hidden_sizes[-1], hidden_sizes[-1], dropout_rate)
        self.value_head = nn.Linear(hidden_sizes[-1], 1)
        self.init_weights()

    def forward(self, state, visual_input):
        visual_features = self.cnn(visual_input)
        x = torch.cat((state, visual_features), dim=1)
        for i, (layer, bn, dropout) in enumerate(zip(self.layers, self.batch_norms, self.dropouts)):
            x = F.leaky_relu(layer(x))
            x = bn(x)
            x = dropout(x)
            if self.attention_layers:
                x = self.attention_layers[i](x)
        x = self.residual_block(x)
        value = self.value_head(x)
        return value

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)

if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    state_dim = 15
    hidden_sizes = [256, 256]
    network = AdvancedCriticNetwork(state_dim, hidden_sizes)
    test_state_input = torch.rand(1, state_dim)
    test_visual_input = torch.rand(1, 3, 144, 256)
    network.eval()
    with torch.no_grad():
        output = network(test_state_input, test_visual_input)
        logger.info(f"Output value: {output}")
