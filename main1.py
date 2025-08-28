"""
实验1：训练集中训练12个环境数据，基因型数据只选择90%。
预测剩下10%的基因型数据在这12个环境下的性状。
做9：1十倍交叉验证选择品种，产生10组训练集和测试集组合。
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import KFold
from utils import prepare_data_experiment1, CropDataset
from models import AttentionGxE
from trainer import Trainer

def main():
    # 设置随机种子确保可复现性
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 设置数据路径
    data_folder = "data"
    
    # 准备数据 - 启用标准化
    geno_data, env_data, pheno_data, cv_splits = prepare_data_experiment1(data_folder, k=10, seed=seed, standardize=True)
    
    # 获取输入维度
    geno_dim = geno_data.shape[1]
    env_sample = list(env_data.values())[0]
    env_dim = env_sample.size  # 环境特征展平后的维度
    
    print(f"基因型维度: {geno_dim}, 环境特征维度: {env_dim}")
    
    # 存储每折交叉验证的结果
    cv_results = []
    
    # 进行交叉验证
    for fold, (train_idx, test_idx) in enumerate(cv_splits):
        print(f"\n开始交叉验证 Fold {fold+1}/10")
        
        # 创建训练集和测试集
        train_dataset = CropDataset(geno_data, env_data, pheno_data, genotype_indices=train_idx, standardize=True)
        test_dataset = CropDataset(geno_data, env_data, pheno_data, genotype_indices=test_idx, standardize=True)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        # 创建增强版多头注意力模型
        hidden_dim = 256  # 增加模型容量
        num_heads = 4    # 使用4头注意力机制
        dropout = 0.2    # 减小dropout以增加信息流
        model = AttentionGxE(geno_dim, env_dim, hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
        
        # 创建训练器 - 使用较小的学习率
        trainer = Trainer(model, device, lr=0.0003, weight_decay=1e-5)
        
        # 训练模型 - 增加epoch数和耐心值
        print(f"开始训练 Fold {fold+1}/10")
        best_metrics = trainer.train(train_loader, test_loader, num_epochs=100, patience=15)
        
        # 存储结果
        cv_results.append({
            'fold': fold + 1,
            'are': best_metrics[0],
            'mse': best_metrics[1],
            'pearson': best_metrics[2]
        })
        
        print(f"Fold {fold+1}/10 完成 - ARE: {best_metrics[0]:.4f}, MSE: {best_metrics[1]:.4f}, Pearson: {best_metrics[2]:.4f}")
    
    # 计算平均结果
    avg_are = np.mean([r['are'] for r in cv_results])
    avg_mse = np.mean([r['mse'] for r in cv_results])
    avg_pearson = np.mean([r['pearson'] for r in cv_results])
    
    # 打印最终结果
    print("\n=== 实验1最终结果 ===")
    print(f"平均 ARE: {avg_are:.4f}")
    print(f"平均 MSE: {avg_mse:.4f}")
    print(f"平均 Pearson 相关系数: {avg_pearson:.4f}")
    
    # 将结果保存到CSV文件
    results_df = pd.DataFrame(cv_results)
    results_df.to_csv("experiment1_results.csv", index=False)
    print("结果已保存到 experiment1_results.csv")

if __name__ == "__main__":
    main() 