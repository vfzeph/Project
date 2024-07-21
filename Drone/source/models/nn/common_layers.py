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

class AttentionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        attention_weights = F.softmax(self.attention(x), dim=1)
        return torch.sum(attention_weights * x, dim=1)

class ICM(nn.Module):
    def __init__(self, icm_config):
        super(ICM, self).__init__()
        self.state_dim = icm_config['state_dim']
        self.action_dim = icm_config['action_dim']
        self.image_channels = icm_config['image_channels']
        self.image_height = icm_config['image_height']
        self.image_width = icm_config['image_width']

        self.cnn = CNNFeatureExtractor(icm_config)
        
        # Calculate CNN output size
        dummy_input = torch.zeros(1, self.image_height, self.image_width, self.image_channels)
        cnn_out = self.cnn(dummy_input)
        self.cnn_output_dim = cnn_out.view(1, -1).size(1)

        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, icm_config['state_encoder']['hidden_dim']),
            nn.ReLU()
        )

        forward_input_dim = self.cnn_output_dim + icm_config['state_encoder']['hidden_dim'] + self.action_dim
        self.forward_model = nn.Sequential(
            nn.Linear(forward_input_dim, icm_config['forward_model']['hidden_dim']),
            nn.ReLU(),
            nn.Linear(icm_config['forward_model']['hidden_dim'], self.state_dim)
        )

        inverse_input_dim = self.cnn_output_dim * 2 + icm_config['state_encoder']['hidden_dim'] * 2
        self.inverse_model = nn.Sequential(
            nn.Linear(inverse_input_dim, icm_config['inverse_model']['hidden_dim']),
            nn.ReLU(),
            nn.Linear(icm_config['inverse_model']['hidden_dim'], self.action_dim)
        )

    def forward(self, state, next_state, action, image, next_image):
        state_feat = self.state_encoder(state)
        next_state_feat = self.state_encoder(next_state)
        image_feat = self.cnn(image)
        next_image_feat = self.cnn(next_image)

        combined_feat = torch.cat([state_feat, image_feat, next_state_feat, next_image_feat], dim=1)
        action_pred = self.inverse_model(combined_feat)
        
        forward_input = torch.cat([state_feat, image_feat, action], dim=1)
        next_state_pred = self.forward_model(forward_input)

        return state_feat, next_state_feat, action_pred, next_state_pred

    def intrinsic_reward(self, state, next_state, action, image, next_image):
        _, _, action_pred, next_state_pred = self.forward(state, next_state, action, image, next_image)
        
        forward_loss = F.mse_loss(next_state_pred, next_state, reduction='none').sum(dim=-1)
        inverse_loss = F.mse_loss(action_pred, action, reduction='none').sum(dim=-1)
        
        intrinsic_reward = forward_loss + inverse_loss
        return intrinsic_reward