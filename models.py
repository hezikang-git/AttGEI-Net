import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepGxE(nn.Module):
    """
    深度学习模型，用于预测基因型-环境交互作用下的作物性状
    结合基因型数据和环境数据预测表型
    """
    def __init__(self, geno_dim, env_dim, hidden_dim=256, dropout=0.3):
        super(DeepGxE, self).__init__()
        
        # 基因型特征提取网络 - 更深层网络
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
        
        # 环境特征提取网络 - 更深层网络
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
        
        # 特征融合网络
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
        # 特征提取
        geno_features = self.geno_encoder(geno)
        env_features = self.env_encoder(env)
        
        # 特征融合（拼接）
        fused_features = torch.cat((geno_features, env_features), dim=1)
        
        # 预测
        output = self.fusion_layer(fused_features)
        
        return output

class ResidualBlock(nn.Module):
    """残差块，用于CNNGxE模型"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 如果输入和输出通道数不同，用1x1卷积调整通道数
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
    增强版CNN模型，用于处理基因型和环境数据的交互
    使用残差卷积网络提取环境特征，多层感知机提取基因型特征
    """
    def __init__(self, geno_dim, env_dim, env_width=53, env_height=53, hidden_dim=256, dropout=0.3):
        super(CNNGxE, self).__init__()
        
        # 保存环境特征尺寸
        self.env_width = env_width
        self.env_height = env_height
        self.env_dim = env_dim
        
        # 自动调整CNN输入尺寸，确保足够容纳增强后的环境特征
        # 计算需要的最小面积
        min_area = env_dim
        
        # 找到合适的矩形尺寸，保持宽高比接近1:1
        self.actual_env_width = int(round(torch.sqrt(torch.tensor(min_area)).item()))
        self.actual_env_height = int((min_area + self.actual_env_width - 1) // self.actual_env_width)  # 向上取整
        
        # 确保总面积大于等于env_dim
        if self.actual_env_width * self.actual_env_height < min_area:
            self.actual_env_width += 1
        
        # 基因型特征提取网络 - 更深层网络
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
        
        # 环境特征直接处理层 - 为避免CNN处理过大的特征，先降维
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
        
        # 环境特征处理 - 转成CNN可处理的尺寸
        self.env_reshaper = nn.Sequential(
            nn.Linear(hidden_dim, 64 * 8 * 8),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(64 * 8 * 8),
            nn.Dropout(dropout)
        )
        
        # 环境特征提取网络 - 使用残差卷积网络
        self.conv1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        
        # 添加残差块
        self.res_block1 = ResidualBlock(32, 32)
        self.res_block2 = ResidualBlock(32, 64, stride=2)
        self.res_block3 = ResidualBlock(64, 64)
        self.res_block4 = ResidualBlock(64, 128, stride=2)
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 环境特征扁平化后的维度
        self.env_flatten_dim = 128
        
        # 环境特征处理
        self.env_fc = nn.Sequential(
            nn.Linear(self.env_flatten_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout)
        )
        
        # 特征融合层
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
        # 基因型特征提取
        geno_features = self.geno_encoder(geno)
        
        # 环境特征先降维，避免CNN处理过大的尺寸
        batch_size = env.size(0)
        
        # 先使用MLP将环境特征降维
        reduced_env = self.env_dim_reducer(env)
        
        # 转换为CNN可处理的形状
        cnn_input = self.env_reshaper(reduced_env)
        cnn_input = cnn_input.view(batch_size, 64, 8, 8)
        
        # 环境特征提取 - 残差卷积网络
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
        
        # 特征融合
        fused_features = torch.cat((geno_features, env_features), dim=1)
        
        # 预测
        output = self.fusion_layer(fused_features)
        
        return output

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    
    def forward(self, query, key, value):
        attn_output, _ = self.multihead_attn(query, key, value)
        return attn_output

class AttentionGxE(nn.Module):
    """
    注意力模型，使用多头注意力机制捕捉基因型和环境之间的交互
    """
    def __init__(self, geno_dim, env_dim, hidden_dim=256, num_heads=4, dropout=0.3):
        super(AttentionGxE, self).__init__()
        
        # 确保hidden_dim可以被num_heads整除
        assert hidden_dim % num_heads == 0, "hidden_dim必须能被num_heads整除"
        
        # 基因型编码器
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
        
        # 环境编码器
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
        
        # 多头注意力层
        self.mha = MultiHeadAttention(hidden_dim, num_heads)
        
        # 特征融合层
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
        
        # 特征提取
        geno_features = self.geno_encoder(geno)
        env_features = self.env_encoder(env)
        
        # 为多头注意力准备输入 - 扩展维度以符合注意力机制的输入要求
        # [batch_size, 1, hidden_dim]
        geno_features_expanded = geno_features.unsqueeze(1)
        env_features_expanded = env_features.unsqueeze(1)
        
        # 应用多头注意力 - 使用基因型特征作为query，环境特征作为key和value
        attended_geno = self.mha(geno_features_expanded, env_features_expanded, env_features_expanded)
        attended_env = self.mha(env_features_expanded, geno_features_expanded, geno_features_expanded)
        
        # 压缩回原始维度 [batch_size, hidden_dim]
        attended_geno = attended_geno.squeeze(1)
        attended_env = attended_env.squeeze(1)
        
        # 融合特征 - 拼接原始特征和注意力处理后的特征
        fused_features = torch.cat((attended_geno, attended_env), dim=1)
        
        # 预测
        output = self.fusion_layer(fused_features)
        
        return output

class CrossAttentionGxE(nn.Module):
    def __init__(self, geno_dim, env_dim, hidden_dim=512, num_heads=8, dropout=0.2):
        super(CrossAttentionGxE, self).__init__()
        
        # 确保hidden_dim可以被num_heads整除
        assert hidden_dim % num_heads == 0, "hidden_dim必须能被num_heads整除"
        
        # 增强的基因型编码器 - 更深层网络
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
        
        # 增强的环境编码器 - 更深层网络
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
        
        # 多头自注意力机制，增加头数获取更丰富的特征关系
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        # 交互层 - 特别针对基因型-环境交互建模
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
        
        # 添加直接交互计算 - 基因型与环境的显式交互
        self.cross_layer = nn.Bilinear(hidden_dim, hidden_dim, hidden_dim//2)
        
        # 预测层
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
        # 编码基因型和环境
        geno_features = self.geno_encoder(geno)
        env_features = self.env_encoder(env)
        
        # 自注意力处理
        batch_size = geno_features.size(0)
        # 将基因型和环境特征拼接成序列用于自注意力
        combined = torch.cat([geno_features.unsqueeze(1), env_features.unsqueeze(1)], dim=1)
        attn_output, _ = self.self_attention(combined, combined, combined)
        
        # 提取注意力后的特征
        attn_geno = attn_output[:, 0, :]
        attn_env = attn_output[:, 1, :]
        
        # 特征融合 - 拼接
        concat_features = torch.cat([attn_geno, attn_env], dim=1)
        interaction_features = self.interaction_layer(concat_features)
        
        # 显式基因型-环境交互
        cross_features = self.cross_layer(geno_features, env_features)
        
        # 融合所有特征
        final_features = torch.cat([interaction_features, cross_features], dim=1)
        
        # 预测
        output = self.predictor(final_features)
        
        return output 