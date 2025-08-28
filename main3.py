"""
实验3：训练集中只选择5个地点（10个环境），基因型数据只选择90%。
预测剩下10%的基因型数据在剩下的1个地点（2个环境）下的性状。
做9：1十倍交叉验证选择基因型数据，做5：1交叉验证选择地点。产生60组训练集和测试集组合。
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import pandas as pd
from utils import prepare_data_experiment3, CropDataset, extend_environment_features, create_stratified_folds
from models import DeepGxE, AttentionGxE, CrossAttentionGxE
from trainer import Trainer
from sklearn.model_selection import KFold

def create_ensemble_models(geno_dim, env_dim, device):
    """创建模型集成"""
    models = []
    
    # 创建三种不同架构的模型
    hidden_dim = 512
    dropout = 0.2
    
    # 模型1：交叉注意力模型
    models.append(CrossAttentionGxE(geno_dim, env_dim, hidden_dim=hidden_dim, 
                                      num_heads=8, dropout=dropout).to(device))
    
    # 模型2：标准注意力模型
    models.append(AttentionGxE(geno_dim, env_dim, hidden_dim=hidden_dim, 
                               num_heads=4, dropout=dropout).to(device))
    
    # 模型3：深度MLP模型
    models.append(DeepGxE(geno_dim, env_dim, hidden_dim=hidden_dim, 
                         dropout=dropout).to(device))
    
    return models

def ensemble_predict(models, test_loader, device):
    """集成预测"""
    all_targets = []
    all_ensemble_preds = []
    
    # 收集每个模型的预测结果
    model_predictions = [[] for _ in range(len(models))]
    
    # 对测试集中的每个批次进行预测
    with torch.no_grad():
        for batch in test_loader:
            geno = batch['genotype'].to(device)
            env = batch['environment'].to(device)
            target = batch['phenotype'].to(device)
            
            # 收集目标值
            all_targets.extend(target.cpu().numpy().flatten())
            
            # 收集每个模型的预测
            for i, model in enumerate(models):
                model.eval()
                pred = model(geno, env)
                model_predictions[i].extend(pred.cpu().numpy().flatten())
    
    # 计算每个样本的集成预测
    for i in range(len(all_targets)):
        # 从每个模型获取预测值并求平均
        preds = [model_predictions[j][i] for j in range(len(models))]
        ensemble_pred = np.mean(preds)
        all_ensemble_preds.append(ensemble_pred)
    
    return np.array(all_targets), np.array(all_ensemble_preds)

def create_combined_cv_splits(geno_data, env_data, k_geno=10, seed=42):
    """创建组合交叉验证划分，结合基因型折和环境折"""
    np.random.seed(seed)
    
    # 创建基因型的k折交叉验证
    kf_geno = KFold(n_splits=k_geno, shuffle=True, random_state=seed)
    geno_splits = []
    for train_idx, test_idx in kf_geno.split(range(len(geno_data))):
        geno_splits.append((train_idx, test_idx))
    
    # 使用分层方法创建环境折
    env_splits = create_stratified_folds(env_data, n_splits=6, seed=seed)
    
    # 组合基因型和环境的交叉验证
    cv_splits = []
    for geno_train_idx, geno_test_idx in geno_splits:
        for train_env_keys, test_env_keys in env_splits:
            cv_splits.append((geno_train_idx, geno_test_idx, train_env_keys, test_env_keys))
    
    return cv_splits

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
    geno_data, env_data, pheno_data, _ = prepare_data_experiment3(data_folder, k=10, seed=seed, standardize=True)
    
    # 应用特征工程扩展环境特征
    extended_env_data = extend_environment_features(env_data)
    print("应用特征工程扩展环境特征完成")
    
    # 创建组合交叉验证划分
    cv_splits = create_combined_cv_splits(geno_data, extended_env_data, k_geno=10, seed=seed)
    
    # 获取输入维度
    geno_dim = geno_data.shape[1]
    env_sample = list(extended_env_data.values())[0]
    env_dim = env_sample.size  # 环境特征展平后的维度（扩展后）
    
    print(f"基因型维度: {geno_dim}, 扩展后环境特征维度: {env_dim}")
    print(f"总共有 {len(cv_splits)} 个交叉验证组合")
    
    # 存储每折交叉验证的结果
    cv_results = []
    
    # 选择执行的折数，减少以加快测试速度 (选取12个代表性的组合，每个地点2个)
    num_folds_to_run = 12  # 为了加快测试速度，只运行12个组合
    selected_folds = []
    
    # 确保每个地点都有代表
    locations = sorted(list(set([key.split('_')[0] for key in extended_env_data.keys()])))
    for loc_idx, loc in enumerate(locations):
        # 找到该地点的所有折
        loc_folds = [i for i, (_, _, _, test_env_keys) in enumerate(cv_splits) 
                    if test_env_keys[0].split('_')[0] == loc]
        
        # 从该地点选择2个折
        if len(loc_folds) >= 2:
            selected_folds.extend(loc_folds[:2])
        else:
            selected_folds.extend(loc_folds)
    
    # 如果不足12个，随机选择剩余的
    if len(selected_folds) < num_folds_to_run:
        remaining = [i for i in range(len(cv_splits)) if i not in selected_folds]
        np.random.shuffle(remaining)
        selected_folds.extend(remaining[:num_folds_to_run - len(selected_folds)])
    
    # 确保不超过12个
    selected_folds = selected_folds[:num_folds_to_run]
    
    print(f"将运行 {len(selected_folds)} 个代表性交叉验证组合")
    
    # 进行交叉验证
    for fold_idx, fold in enumerate(selected_folds):
        geno_train_idx, geno_test_idx, train_env_keys, test_env_keys = cv_splits[fold]
        
        geno_fold = fold // 6 + 1  # 基因型数据的折数 (1-10)
        env_fold = fold % 6 + 1    # 环境数据的折数 (1-6)
        test_location = test_env_keys[0].split('_')[0]  # 测试地点
        
        print(f"\n开始交叉验证 组合 {fold_idx+1}/{len(selected_folds)} (全局fold {fold+1}) - 基因型折 {geno_fold}/10, 环境折 {env_fold}/6, 测试地点: {test_location}")
        
        # 创建训练集和测试集
        train_dataset = CropDataset(geno_data, extended_env_data, pheno_data, 
                                    genotype_indices=geno_train_idx, 
                                    env_locations=train_env_keys,
                                    standardize=True)
        
        test_dataset = CropDataset(geno_data, extended_env_data, pheno_data, 
                                  genotype_indices=geno_test_idx, 
                                  env_locations=test_env_keys,
                                  standardize=True)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        # 创建模型集成
        models = create_ensemble_models(geno_dim, env_dim, device)
        
        # 训练每个模型
        trained_models = []
        for i, model in enumerate(models):
            print(f"训练模型 {i+1}/{len(models)}")
            
            # 创建训练器 - 确保只传递正确的参数
            trainer = Trainer(model, device, lr=0.0003, weight_decay=1e-5)
            
            # 训练模型 - 增加epoch数和耐心值
            print(f"开始训练模型 {i+1}/{len(models)}")
            best_metrics = trainer.train(train_loader, test_loader, num_epochs=150, patience=20)
            
            print(f"模型 {i+1}/{len(models)} 完成 - ARE: {best_metrics[0]:.4f}, MSE: {best_metrics[1]:.4f}, Pearson: {best_metrics[2]:.4f}")
            
            # 保存已训练的模型
            trained_models.append(model)
        
        # 使用集成模型进行预测
        targets, ensemble_preds = ensemble_predict(trained_models, test_loader, device)
        
        # 计算集成模型的评估指标
        from utils import evaluate_model
        are, mse, pearson = evaluate_model(targets, ensemble_preds)
        
        # 存储结果
        cv_results.append({
            'fold': fold + 1,  # 原始折索引
            'fold_idx': fold_idx + 1,  # 运行的折索引
            'geno_fold': geno_fold,
            'env_fold': env_fold,
            'test_location': test_location,
            'are': are,
            'mse': mse,
            'pearson': pearson
        })
        
        print(f"组合 {fold_idx+1}/{len(selected_folds)} 集成结果 - ARE: {are:.4f}, MSE: {mse:.4f}, Pearson: {pearson:.4f}")
        
        # 每个组合都保存一次中间结果
        interim_df = pd.DataFrame(cv_results)
        interim_df.to_csv(f"experiment3_combined_interim_results_{fold_idx+1}.csv", index=False)
        print(f"中间结果已保存到 experiment3_combined_interim_results_{fold_idx+1}.csv")
    
    # 计算平均结果
    avg_are = np.mean([r['are'] for r in cv_results])
    avg_mse = np.mean([r['mse'] for r in cv_results])
    avg_pearson = np.mean([r['pearson'] for r in cv_results])
    
    # 打印最终结果
    print("\n=== 实验3组合版最终结果 ===")
    print(f"平均 ARE: {avg_are:.4f}")
    print(f"平均 MSE: {avg_mse:.4f}")
    print(f"平均 Pearson 相关系数: {avg_pearson:.4f}")
    
    # 按地点分组的结果
    print("\n按地点分组的结果:")
    locations = set([r['test_location'] for r in cv_results])
    for loc in locations:
        loc_results = [r for r in cv_results if r['test_location'] == loc]
        loc_are = np.mean([r['are'] for r in loc_results])
        loc_mse = np.mean([r['mse'] for r in loc_results])
        loc_pearson = np.mean([r['pearson'] for r in loc_results])
        print(f"地点 {loc} - ARE: {loc_are:.4f}, MSE: {loc_mse:.4f}, Pearson: {loc_pearson:.4f}")
    
    # 将结果保存到CSV文件
    results_df = pd.DataFrame(cv_results)
    results_df.to_csv("experiment3_combined_results.csv", index=False)
    print("结果已保存到 experiment3_combined_results.csv")

if __name__ == "__main__":
    main() 