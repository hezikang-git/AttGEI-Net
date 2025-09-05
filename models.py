import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    
    def forward(self, query, key, value):
        attn_output, _ = self.multihead_attn(query, key, value)
        return attn_output

class AttGEINet(nn.Module):
    """
    Attention-based Genotype-Environment Interaction Network (AttGEI-Net)
    
    A cross-attention model with increased feature extraction depth, interaction modeling capability,
    and self-attention mechanism designed specifically for complex genotype-environment interaction modeling
    """
    def __init__(self, geno_dim, env_dim, hidden_dim=512, num_heads=8, dropout=0.2):
        super(AttGEINet, self).__init__()
        
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