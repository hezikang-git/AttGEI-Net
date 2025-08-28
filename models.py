import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepGxE(nn.Module):
    """
    Deep learning model for predicting crop traits under genotype-environment interactions
    Combines genotype data and environmental data to predict phenotype
    """
    def __init__(self, geno_dim, env_dim, hidden_dim=256, dropout=0.3):
        super(DeepGxE, self).__init__()
        
        # Genotype feature extraction network - deeper network
        self.geno_encoder = nn.Sequential(
            nn.Linear(geno_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout)
        )
        
        # Environmental feature extraction network - deeper network
        self.env_encoder = nn.Sequential(
            nn.Linear(env_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout)
        )
        
        # Feature fusion network
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )
    
    def forward(self, geno, env):
        # Feature extraction
        geno_features = self.geno_encoder(geno)
        env_features = self.env_encoder(env)
        
        # Feature fusion (concatenation)
        fused_features = torch.cat((geno_features, env_features), dim=1)
        
        # Prediction
        output = self.fusion_layer(fused_features)
        
        return output

class ResidualBlock(nn.Module):
    """Residual block for CNNGxE model"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # If input and output channels differ, use 1x1 convolution to adjust channels
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(residual)
        out = self.relu(out)
        
        return out

class CNNGxE(nn.Module):
    """
    Enhanced CNN model for processing genotype-environment interactions
    Uses residual convolutional networks to extract environmental features, 
    and multilayer perceptron to extract genotype features
    """
    def __init__(self, geno_dim, env_dim, env_width=53, env_height=53, hidden_dim=256, dropout=0.3):
        super(CNNGxE, self).__init__()
        
        # Save environment feature dimensions
        self.env_width = env_width
        self.env_height = env_height
        self.env_dim = env_dim
        
        # Automatically adjust CNN input size to accommodate enhanced environmental features
        # Calculate minimum required area
        min_area = env_dim
        
        # Find suitable rectangle dimensions with aspect ratio close to 1:1
        self.actual_env_width = int(round(torch.sqrt(torch.tensor(min_area)).item()))
        self.actual_env_height = int((min_area + self.actual_env_width - 1) // self.actual_env_width)  # Round up
        
        # Ensure total area is greater than or equal to env_dim
        if self.actual_env_width * self.actual_env_height < min_area:
            self.actual_env_width += 1
        
        # Genotype feature extraction network - deeper network
        self.geno_encoder = nn.Sequential(
            nn.Linear(geno_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout)
        )
        
        # Environment feature direct processing layer - dimensionality reduction before CNN
        self.env_dim_reducer = nn.Sequential(
            nn.Linear(env_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Environment feature processing - conversion to CNN-processable dimensions
        self.env_reshaper = nn.Sequential(
            nn.Linear(hidden_dim, 64 * 8 * 8),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(64 * 8 * 8),
            nn.Dropout(dropout)
        )
        
        # Environment feature extraction network - using residual convolutional networks
        self.conv1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        
        # Add residual blocks
        self.res_block1 = ResidualBlock(32, 32)
        self.res_block2 = ResidualBlock(32, 64, stride=2)
        self.res_block3 = ResidualBlock(64, 64)
        self.res_block4 = ResidualBlock(64, 128, stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Flattened dimension of environmental features
        self.env_flatten_dim = 128
        
        # Environmental feature processing
        self.env_fc = nn.Sequential(
            nn.Linear(self.env_flatten_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout)
        )
        
        # Feature fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )
    
    def forward(self, geno, env):
        # Genotype feature extraction
        geno_features = self.geno_encoder(geno)
        
        # Environmental feature dimensionality reduction
        batch_size = env.size(0)
        
        # First use MLP to reduce environmental features
        reduced_env = self.env_dim_reducer(env)
        
        # Convert to CNN-processable shape
        cnn_input = self.env_reshaper(reduced_env)
        cnn_input = cnn_input.view(batch_size, 64, 8, 8)
        
        # Environmental feature extraction - residual convolutional networks
        x = self.conv1(cnn_input)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        
        x = self.avgpool(x)
        x = x.view(batch_size, -1)
        
        env_features = self.env_fc(x)
        
        # Feature fusion
        fused_features = torch.cat((geno_features, env_features), dim=1)
        
        # Prediction
        output = self.fusion_layer(fused_features)
        
        return output

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    
    def forward(self, query, key, value):
        attn_output, _ = self.multihead_attn(query, key, value)
        return attn_output

class AttentionGxE(nn.Module):
    """
    Attention model using multi-head attention mechanism to capture complex interactions
    between genotype and environment
    """
    def __init__(self, geno_dim, env_dim, hidden_dim=256, num_heads=4, dropout=0.3):
        super(AttentionGxE, self).__init__()
        
        # Ensure hidden_dim is divisible by num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Genotype encoder
        self.geno_encoder = nn.Sequential(
            nn.Linear(geno_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Environment encoder
        self.env_encoder = nn.Sequential(
            nn.Linear(env_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Multi-head attention layer
        self.mha = MultiHeadAttention(hidden_dim, num_heads)
        
        # Feature fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, geno, env):
        batch_size = geno.size(0)
        
        # Feature extraction
        geno_features = self.geno_encoder(geno)
        env_features = self.env_encoder(env)
        
        # Prepare inputs for multi-head attention - expand dimensions to match attention requirements
        # [batch_size, 1, hidden_dim]
        geno_features_expanded = geno_features.unsqueeze(1)
        env_features_expanded = env_features.unsqueeze(1)
        
        # Apply multi-head attention - use genotype features as query, environment features as key and value
        attended_geno = self.mha(geno_features_expanded, env_features_expanded, env_features_expanded)
        attended_env = self.mha(env_features_expanded, geno_features_expanded, geno_features_expanded)
        
        # Compress back to original dimensions [batch_size, hidden_dim]
        attended_geno = attended_geno.squeeze(1)
        attended_env = attended_env.squeeze(1)
        
        # Feature fusion - concatenate original features and attention-processed features
        fused_features = torch.cat((attended_geno, attended_env), dim=1)
        
        # Prediction
        output = self.fusion_layer(fused_features)
        
        return output

class CrossAttentionGxE(nn.Module):
    """
    Cross-attention model with increased feature extraction depth, interaction modeling capability,
    and self-attention mechanism designed specifically for complex genotype-environment interaction modeling
    """
    def __init__(self, geno_dim, env_dim, hidden_dim=512, num_heads=8, dropout=0.2):
        super(CrossAttentionGxE, self).__init__()
        
        # Ensure hidden_dim is divisible by num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Enhanced genotype encoder - deeper network
        self.geno_encoder = nn.Sequential(
            nn.Linear(geno_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # Enhanced environment encoder - deeper network
        self.env_encoder = nn.Sequential(
            nn.Linear(env_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # Multi-head self-attention mechanism, increased head count for richer feature relationships
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        # Interaction layer - specifically for genotype-environment interaction modeling
        self.interaction_layer = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.Dropout(dropout)
        )
        
        # Add direct interaction calculation - explicit genotype-environment interaction
        self.cross_layer = nn.Bilinear(hidden_dim, hidden_dim, hidden_dim//2)
        
        # Prediction layer
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim//4),
            nn.Linear(hidden_dim//4, 1)
        )

    def forward(self, geno, env):
        # Encode genotype and environment
        geno_features = self.geno_encoder(geno)
        env_features = self.env_encoder(env)
        
        # Self-attention processing
        batch_size = geno_features.size(0)
        # Concatenate genotype and environment features as a sequence for self-attention
        combined = torch.cat([geno_features.unsqueeze(1), env_features.unsqueeze(1)], dim=1)
        attn_output, _ = self.self_attention(combined, combined, combined)
        
        # Extract post-attention features
        attn_geno = attn_output[:, 0, :]
        attn_env = attn_output[:, 1, :]
        
        # Feature fusion - concatenation
        concat_features = torch.cat([attn_geno, attn_env], dim=1)
        interaction_features = self.interaction_layer(concat_features)
        
        # Explicit genotype-environment interaction
        cross_features = self.cross_layer(geno_features, env_features)
        
        # Fuse all features
        final_features = torch.cat([interaction_features, cross_features], dim=1)
        
        # Prediction
        output = self.predictor(final_features)
        
        return output