import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNFeatureExtractor(nn.Module):
    def __init__(self, config):
        super(CNNFeatureExtractor, self).__init__()
        self.cnn_layers = nn.ModuleList()
        in_channels = config['image_channels']
        
        for i in range(1, 5):  # Assuming up to 4 CNN layers
            if f'conv{i}' not in config['cnn']:
                break
            conv_config = config['cnn'][f'conv{i}']
            self.cnn_layers.append(nn.Conv2d(in_channels, conv_config['out_channels'], 
                                             kernel_size=conv_config['kernel_size'], 
                                             stride=conv_config['stride'],
                                             padding=conv_config.get('padding', 0)))
            self.cnn_layers.append(nn.ReLU())
            in_channels = conv_config['out_channels']
        
        self.flatten = nn.Flatten()

    def forward(self, x):
        # Ensure input is in the correct format (B, C, H, W)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.shape[1] != 3:
            x = x.permute(0, 3, 1, 2)
        
        for layer in self.cnn_layers:
            x = layer(x)
        return self.flatten(x)

class AdvancedCriticNetwork(nn.Module):
    def __init__(self, state_dim, hidden_sizes, config):
        super(AdvancedCriticNetwork, self).__init__()
        self.cnn = CNNFeatureExtractor(config)
        
        # Calculate CNN output size
        dummy_input = torch.zeros(1, config['image_channels'], config['image_height'], config['image_width'])
        cnn_out = self.cnn(dummy_input)
        self.cnn_out_size = cnn_out.view(1, -1).size(1)
        
        self.state_encoder = nn.Linear(state_dim, hidden_sizes[0])
        
        self.fc_layers = nn.ModuleList()
        in_size = hidden_sizes[0] + self.cnn_out_size
        for out_size in hidden_sizes[1:]:
            self.fc_layers.append(nn.Linear(in_size, out_size))
            if config.get('use_batch_norm', False):
                self.fc_layers.append(nn.GroupNorm(1, out_size))  # Use GroupNorm instead of BatchNorm
            self.fc_layers.append(nn.ReLU())
            if config.get('use_dropout', False):
                self.fc_layers.append(nn.Dropout(config.get('dropout_rate', 0.2)))
            in_size = out_size
        
        self.value_head = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, state, visual):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if visual.dim() == 3:
            visual = visual.unsqueeze(0)
        
        state_features = F.relu(self.state_encoder(state))
        visual_features = self.cnn(visual)
        combined = torch.cat([state_features, visual_features], dim=1)
        
        x = combined
        for layer in self.fc_layers:
            x = layer(x)
        
        return self.value_head(x)

    def evaluate_value(self, state, visual):
        with torch.no_grad():
            return self.forward(state, visual)