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

class AdvancedCriticNetwork(nn.Module):
    def __init__(self, state_dim, hidden_sizes=[256, 256], dropout_rate=0.2, input_channels=3):
        super(AdvancedCriticNetwork, self).__init__()
        self.cnn = CNNFeatureExtractor(input_channels)
        
        if self.cnn._to_linear is None:
            raise ValueError("CNNFeatureExtractor _to_linear attribute is None. Ensure determine_to_linear is called properly.")
        
        combined_input_dim = state_dim + self.cnn._to_linear

        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        input_dim = combined_input_dim

        for size in hidden_sizes:
            self.layers.append(nn.Linear(input_dim, size))
            self.batch_norms.append(nn.BatchNorm1d(size))
            self.dropouts.append(nn.Dropout(dropout_rate))
            input_dim = size

        self.residual_block = ResidualBlock(input_dim, hidden_sizes[-1], dropout_rate)
        self.value_head = nn.Linear(hidden_sizes[-1], 1)
        self.init_weights()

    def forward(self, state, visual_input):
        visual_features = self.cnn(visual_input)
        x = torch.cat((state, visual_features), dim=1)

        logger.debug(f"Combined input shape: {x.shape}")

        for i, (layer, bn, dropout) in enumerate(zip(self.layers, self.batch_norms, self.dropouts)):
            x = F.leaky_relu(bn(layer(x)))
            x = dropout(x)
            logger.debug(f"Shape after layer {i}: {x.shape}")

        x = self.residual_block(x)
        logger.debug(f"Shape after residual block: {x.shape}")

        value = self.value_head(x)
        logger.debug(f"Output value shape: {value.shape}")

        return value

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)

if __name__ == "__main__":
    logger.info("Testing the AdvancedCriticNetwork with sample inputs.")
    state_dim = 15
    input_channels = 3
    hidden_sizes = [256, 256]
    
    network = AdvancedCriticNetwork(state_dim, hidden_sizes, input_channels=input_channels)
    test_state_input = torch.rand(1, state_dim)
    test_visual_input = torch.rand(1, input_channels, 144, 256)  # Use the actual input dimensions
    
    network.eval()
    with torch.no_grad():
        output = network(test_state_input, test_visual_input)
        logger.info(f"Output value: {output}")
