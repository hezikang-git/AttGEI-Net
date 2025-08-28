import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from models import DeepGxE, CrossAttentionGxE, AttentionGxE
from utils import (load_genotype_data, load_environment_data, load_phenotype_data, 
                  evaluate_model, standardize_genotype, standardize_environment, 
                  CropDataset, normalize_phenotypes, inverse_normalize_predictions,
                  calculate_normalization_params)
import argparse
import logging
from datetime import datetime
from pathlib import Path

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class TrainEvaluator:
    """训练和评估每个性状模型的类"""
    def __init__(self, basedata_path, basedata1_path, testdata_path, output_dir, 
                 model_type='attention', hidden_dim=256, num_heads=4, 
                 batch_size=64, lr=0.0003, epochs=100, patience=15,
                 device=None, seed=42):
        self.basedata_path = basedata_path
        self.basedata1_path = basedata1_path
        self.testdata_path = testdata_path
        self.output_dir = output_dir
        self.model_type = model_type
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.seed = seed
        
        # 设置设备
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"使用设备: {self.device}")
        
        # 设置随机种子
        self.set_seed(seed)
        
        # 创建输出目录
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 获取所有性状目录
        self.trait_dirs = self.get_trait_directories()
        logging.info(f"找到以下性状目录: {self.trait_dirs}")
        
    def set_seed(self, seed):
        """设置随机种子确保可复现性"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
    
    def get_trait_directories(self):
        """获取所有性状目录列表"""
        # 从basedata获取所有性状目录
        trait_dirs = [d for d in os.listdir(self.basedata_path) 
                     if os.path.isdir(os.path.join(self.basedata_path, d))]
        return trait_dirs
    
    def create_model(self, geno_dim, env_dim):
        """创建模型"""
        if self.model_type.lower() == 'deepgxe':
            return DeepGxE(geno_dim, env_dim, hidden_dim=self.hidden_dim)
        elif self.model_type.lower() == 'crossattention':
            return CrossAttentionGxE(geno_dim, env_dim, hidden_dim=self.hidden_dim, num_heads=self.num_heads)
        else:  # default to AttentionGxE
            return AttentionGxE(geno_dim, env_dim, hidden_dim=self.hidden_dim, num_heads=self.num_heads)
    
    def load_combined_data(self, trait_dir):
        """加载并合并basedata和basedata1的数据"""
        # 加载basedata数据
        base_geno_path = os.path.join(self.basedata_path, trait_dir, 'genodata.txt')
        base_env_dir = os.path.join(self.basedata_path, trait_dir, 'environment')
        base_char_dir = os.path.join(self.basedata_path, trait_dir, 'characteristic')
        
        base_geno_data = load_genotype_data(base_geno_path)
        base_env_data = load_environment_data(base_env_dir)
        base_pheno_data = load_phenotype_data(base_char_dir)
        
        # 检查basedata1是否有此性状目录
        base1_trait_path = os.path.join(self.basedata1_path, trait_dir)
        if os.path.exists(base1_trait_path):
            # 加载basedata1数据
            base1_geno_path = os.path.join(base1_trait_path, 'genodata.txt')
            base1_env_dir = os.path.join(base1_trait_path, 'environment')
            base1_char_dir = os.path.join(base1_trait_path, 'characteristic')
            
            base1_geno_data = load_genotype_data(base1_geno_path)
            base1_env_data = load_environment_data(base1_env_dir)
            base1_pheno_data = load_phenotype_data(base1_char_dir)
            
            logging.info(f"性状 {trait_dir}: 合并basedata ({len(base_geno_data)} 样本)和basedata1 ({len(base1_geno_data)} 样本)")
            
            # 数据合并在训练数据集创建时进行，这里只返回两组数据
            return base_geno_data, base_env_data, base_pheno_data, base1_geno_data, base1_env_data, base1_pheno_data
        else:
            logging.info(f"性状 {trait_dir}: 仅使用basedata ({len(base_geno_data)} 样本)")
            return base_geno_data, base_env_data, base_pheno_data, None, None, None
    
    def load_test_data(self, trait_dir):
        """加载测试数据"""
        # 检查testdata中是否有此性状目录
        test_trait_path = os.path.join(self.testdata_path, trait_dir)
        if not os.path.exists(test_trait_path):
            logging.warning(f"测试数据中不存在性状目录 {trait_dir}")
            return None, None, None
        
        # 加载测试数据
        test_geno_path = os.path.join(test_trait_path, 'genodata.txt')
        test_env_dir = os.path.join(test_trait_path, 'environment')
        test_char_dir = os.path.join(test_trait_path, 'characteristic')
        
        test_geno_data = load_genotype_data(test_geno_path)
        test_env_data = load_environment_data(test_env_dir)
        test_pheno_data = load_phenotype_data(test_char_dir)
        
        return test_geno_data, test_env_data, test_pheno_data
    
    def create_datasets(self, base_geno_data, base_env_data, base_pheno_data, 
                      base1_geno_data, base1_env_data, base1_pheno_data,
                      test_geno_data, test_env_data, test_pheno_data):
        """创建训练、验证和测试数据集"""
        # 为basedata创建索引
        base_indices = np.arange(len(base_geno_data))
        
        # 拆分训练和验证集 (80/20 split)
        train_indices, val_indices = train_test_split(
            base_indices, test_size=0.2, random_state=self.seed
        )
        
        # 创建训练集子集表型数据（用于计算归一化参数）
        train_pheno_data = {}
        for env_key, pheno_values in base_pheno_data.items():
            # 获取对应环境的训练索引
            valid_train_indices = [idx for idx in train_indices if idx < len(pheno_values)]
            if valid_train_indices:
                train_pheno_data[env_key] = pheno_values[valid_train_indices]
        
        # 如果有basedata1，添加到训练集表型数据
        if base1_geno_data is not None and base1_pheno_data is not None:
            base1_indices = np.arange(len(base1_geno_data))
            base1_train_indices, _ = train_test_split(
                base1_indices, test_size=0.2, random_state=self.seed
            )
            
            for env_key, pheno_values in base1_pheno_data.items():
                valid_train_indices = [idx for idx in base1_train_indices if idx < len(pheno_values)]
                if valid_train_indices:
                    if env_key in train_pheno_data:
                        # 如果环境已存在，追加数据
                        train_pheno_data[env_key] = np.concatenate([
                            train_pheno_data[env_key], 
                            pheno_values[valid_train_indices]
                        ])
                    else:
                        train_pheno_data[env_key] = pheno_values[valid_train_indices]
        
        # 计算表型归一化参数（只使用训练数据）
        pheno_min, pheno_max = None, None
        if train_pheno_data:
            pheno_min, pheno_max = calculate_normalization_params(train_pheno_data)
            logging.info(f"使用训练数据计算表型归一化参数: 最小值={pheno_min:.4f}, 最大值={pheno_max:.4f}")
        
        # 创建训练集 - 包含basedata的训练部分（is_train=True）
        train_dataset = CropDataset(
            base_geno_data, base_env_data, base_pheno_data, 
            genotype_indices=train_indices, standardize=True,
            normalize_pheno=True, is_train=True
        )
        
        # 如果有basedata1，添加到训练集
        if base1_geno_data is not None:
            # 为basedata1创建索引
            base1_indices = np.arange(len(base1_geno_data))
            
            # 将basedata1分为训练和验证
            base1_train_indices, base1_val_indices = train_test_split(
                base1_indices, test_size=0.2, random_state=self.seed
            )
            
            # 创建basedata1训练集
            base1_train_dataset = CropDataset(
                base1_geno_data, base1_env_data, base1_pheno_data, 
                genotype_indices=base1_train_indices, standardize=True,
                normalize_pheno=True, is_train=True
            )
            
            # 创建basedata1验证集（使用训练集的归一化参数）
            base1_val_dataset = CropDataset(
                base1_geno_data, base1_env_data, base1_pheno_data, 
                genotype_indices=base1_val_indices, standardize=True,
                normalize_pheno=True, pheno_min=pheno_min, pheno_max=pheno_max, is_train=False
            )
        else:
            base1_train_dataset = None
            base1_val_dataset = None
        
        # 创建验证集 - 仅包含basedata的验证部分（使用训练集的归一化参数）
        val_dataset = CropDataset(
            base_geno_data, base_env_data, base_pheno_data, 
            genotype_indices=val_indices, standardize=True,
            normalize_pheno=True, pheno_min=pheno_min, pheno_max=pheno_max, is_train=False
        )
        
        # 创建测试集（使用训练集的归一化参数）
        if test_geno_data is not None:
            test_indices = np.arange(len(test_geno_data))
            test_dataset = CropDataset(
                test_geno_data, test_env_data, test_pheno_data, 
                genotype_indices=test_indices, standardize=True,
                normalize_pheno=True, pheno_min=pheno_min, pheno_max=pheno_max, is_train=False
            )
        else:
            test_dataset = None
        
        return train_dataset, val_dataset, test_dataset, base1_train_dataset, base1_val_dataset
    
    def train_model(self, model, train_loader, val_loader):
        """训练模型"""
        # 设置优化器和损失函数
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=0.01)
        
        # 使用Huber损失函数，对异常值更加鲁棒
        criterion = torch.nn.HuberLoss(delta=1.0)
        
        # 添加学习率调度器 - 更强的衰减策略
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-6
        )
        
        # 追踪最佳模型
        best_val_loss = float('inf')
        best_epoch = 0
        best_model_state = None
        no_improve = 0
        
        # 训练循环
        for epoch in range(self.epochs):
            # 训练阶段
            model.train()
            train_loss = 0
            
            for batch in train_loader:
                # 获取数据
                geno = batch['genotype'].to(self.device)
                env = batch['environment'].to(self.device)
                target = batch['phenotype'].to(self.device)
                
                # 前向传播
                optimizer.zero_grad()
                output = model(geno, env)
                loss = criterion(output, target)
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # 验证阶段
            model.eval()
            val_loss = 0
            val_preds = []
            val_targets = []
            
            with torch.no_grad():
                for batch in val_loader:
                    geno = batch['genotype'].to(self.device)
                    env = batch['environment'].to(self.device)
                    target = batch['phenotype'].to(self.device)
                    
                    output = model(geno, env)
                    loss = criterion(output, target)
                    
                    val_loss += loss.item()
                    val_preds.extend(output.cpu().numpy().flatten())
                    val_targets.extend(target.cpu().numpy().flatten())
            
            val_loss /= len(val_loader)
            
            # 计算验证集评估指标
            val_are, val_mse, val_pearson = evaluate_model(
                np.array(val_targets), np.array(val_preds)
            )
            
            # 更新学习率
            scheduler.step(val_loss)
            
            # 打印进度
            logging.info(
                f"Epoch {epoch+1}/{self.epochs} - Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val ARE: {val_are:.4f}, "
                f"Val MSE: {val_mse:.4f}, Val Pearson: {val_pearson:.4f}"
            )
            
            # 检查是否是最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_model_state = model.state_dict().copy()
                no_improve = 0
            else:
                no_improve += 1
            
            # 早停检查
            if no_improve >= self.patience:
                logging.info(f"早停 - {no_improve} 个epochs没有改善")
                break
        
        # 恢复最佳模型
        model.load_state_dict(best_model_state)
        logging.info(f"训练完成 - 最佳epoch: {best_epoch+1}, 最佳验证损失: {best_val_loss:.4f}")
        
        return model, best_epoch, best_val_loss
    
    def test_model(self, model, test_loader):
        """测试模型并返回评估指标"""
        model.eval()
        test_preds = []
        test_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                geno = batch['genotype'].to(self.device)
                env = batch['environment'].to(self.device)
                target = batch['phenotype'].to(self.device)
                
                # 获取5次预测并平均，增强稳定性
                outputs = []
                for _ in range(5):  
                    output = model(geno, env)
                    outputs.append(output)
                    
                # 平均多次预测结果
                avg_output = torch.mean(torch.stack(outputs), dim=0)
                
                test_preds.extend(avg_output.cpu().numpy().flatten())
                test_targets.extend(target.cpu().numpy().flatten())
        
        # 获取原始比例的预测值和目标值
        # 检查test_loader中是否有归一化信息
        if hasattr(test_loader.dataset, 'normalize_pheno') and test_loader.dataset.normalize_pheno:
            pheno_min = test_loader.dataset.pheno_min
            pheno_max = test_loader.dataset.pheno_max
            
            # 转换回原始尺度
            test_preds = inverse_normalize_predictions(np.array(test_preds), pheno_min, pheno_max)
            test_targets = inverse_normalize_predictions(np.array(test_targets), pheno_min, pheno_max)
            
            logging.info(f"预测值已转换回原始尺度: 最小值={pheno_min:.4f}, 最大值={pheno_max:.4f}")
        else:
            test_preds = np.array(test_preds)
            test_targets = np.array(test_targets)
        
        # 计算测试指标
        test_are, test_mse, test_pearson = evaluate_model(test_targets, test_preds)
        
        return test_are, test_mse, test_pearson, test_preds, test_targets
    
    def run_trait_evaluation(self, trait_dir):
        """执行单个性状的训练、验证和测试流程"""
        logging.info(f"处理性状目录: {trait_dir}")
        
        # 1. 加载数据
        base_geno_data, base_env_data, base_pheno_data, base1_geno_data, base1_env_data, base1_pheno_data = self.load_combined_data(trait_dir)
        test_geno_data, test_env_data, test_pheno_data = self.load_test_data(trait_dir)
        
        if test_geno_data is None:
            logging.warning(f"无测试数据，跳过性状 {trait_dir}")
            return None
        
        # 2. 创建数据集
        train_dataset, val_dataset, test_dataset, base1_train_dataset, base1_val_dataset = self.create_datasets(
            base_geno_data, base_env_data, base_pheno_data,
            base1_geno_data, base1_env_data, base1_pheno_data,
            test_geno_data, test_env_data, test_pheno_data
        )
        
        # 合并basedata和basedata1的训练集（如果存在）
        if base1_train_dataset:
            from torch.utils.data import ConcatDataset
            combined_train_dataset = ConcatDataset([train_dataset, base1_train_dataset])
            combined_val_dataset = ConcatDataset([val_dataset, base1_val_dataset])
            
            logging.info(f"合并数据集: 训练集 {len(train_dataset)} + {len(base1_train_dataset)} = {len(combined_train_dataset)} 样本")
            logging.info(f"合并数据集: 验证集 {len(val_dataset)} + {len(base1_val_dataset)} = {len(combined_val_dataset)} 样本")
            
            train_dataset = combined_train_dataset
            val_dataset = combined_val_dataset
        
        # 3. 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        
        # 4. 获取模型输入维度
        geno_dim = base_geno_data.shape[1]
        env_sample = list(base_env_data.values())[0].flatten()
        env_dim = env_sample.size
        
        logging.info(f"模型输入维度: 基因型维度={geno_dim}, 环境维度={env_dim}")
        
        # 5. 创建模型
        model = self.create_model(geno_dim, env_dim)
        model = model.to(self.device)
        
        # 6. 训练模型
        logging.info(f"开始训练 {trait_dir} 性状模型")
        trained_model, best_epoch, best_val_loss = self.train_model(model, train_loader, val_loader)
        
        # 7. 测试模型
        logging.info(f"开始测试 {trait_dir} 性状模型")
        test_are, test_mse, test_pearson, test_preds, test_targets = self.test_model(
            trained_model, test_loader
        )
        
        logging.info(f"测试结果 - ARE: {test_are:.4f}, MSE: {test_mse:.4f}, Pearson: {test_pearson:.4f}")
        
        # 8. 保存模型和结果
        results = {
            'trait': trait_dir,
            'model_type': self.model_type,
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'test_are': test_are,
            'test_mse': test_mse,
            'test_pearson': test_pearson,
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'test_samples': len(test_dataset)
        }
        
        # 保存模型
        model_save_path = os.path.join(self.output_dir, f"{trait_dir}_model.pth")
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'model_type': self.model_type,
            'geno_dim': geno_dim,
            'env_dim': env_dim,
            'hidden_dim': self.hidden_dim,
            'num_heads': self.num_heads,
            'results': results
        }, model_save_path)
        
        # 保存预测结果
        pred_df = pd.DataFrame({
            'true': test_targets,
            'pred': test_preds
        })
        pred_save_path = os.path.join(self.output_dir, f"{trait_dir}_predictions.csv")
        pred_df.to_csv(pred_save_path, index=False)
        
        return results
    
    def run_all_traits(self):
        """执行所有性状的评估"""
        all_results = []
        
        for trait_dir in self.trait_dirs:
            try:
                result = self.run_trait_evaluation(trait_dir)
                if result:
                    all_results.append(result)
            except Exception as e:
                logging.error(f"处理性状 {trait_dir} 时出错: {str(e)}", exc_info=True)
        
        # 保存所有结果
        if all_results:
            results_df = pd.DataFrame(all_results)
            results_save_path = os.path.join(self.output_dir, "all_traits_results.csv")
            results_df.to_csv(results_save_path, index=False)
            
            # 打印汇总结果
            logging.info("\n====== 所有性状评估结果 ======")
            logging.info(f"平均 ARE: {results_df['test_are'].mean():.4f}")
            logging.info(f"平均 MSE: {results_df['test_mse'].mean():.4f}")
            logging.info(f"平均 Pearson: {results_df['test_pearson'].mean():.4f}")
            logging.info(f"结果已保存到: {results_save_path}")
        
        return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练和评估作物性状预测模型")
    parser.add_argument("--basedata", type=str, required=True, help="basedata目录路径")
    parser.add_argument("--basedata1", type=str, required=True, help="basedata1目录路径")
    parser.add_argument("--testdata", type=str, required=True, help="testdata目录路径")
    parser.add_argument("--output", type=str, required=True, help="输出目录路径")
    parser.add_argument("--model", type=str, default="attention",
                      choices=["deepgxe", "crossattention", "attention"],
                      help="模型类型: deepgxe, crossattention, attention")
    parser.add_argument("--hidden_dim", type=int, default=128, help="隐藏层维度")
    parser.add_argument("--num_heads", type=int, default=4, help="注意力头数量")
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
    parser.add_argument("--lr", type=float, default=0.0001, help="学习率")
    parser.add_argument("--epochs", type=int, default=200, help="训练轮数")
    parser.add_argument("--patience", type=int, default=35, help="早停耐心值")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--trait", type=str, default=None, help="指定单个性状进行训练(可选)")
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = TrainEvaluator(
        basedata_path=args.basedata,
        basedata1_path=args.basedata1,
        testdata_path=args.testdata,
        output_dir=args.output,
        model_type=args.model,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        patience=args.patience,
        seed=args.seed
    )
    
    # 开始评估
    if args.trait:
        # 仅评估指定性状
        if args.trait in evaluator.trait_dirs:
            logging.info(f"仅评估指定性状: {args.trait}")
            evaluator.run_trait_evaluation(args.trait)
        else:
            logging.error(f"指定的性状 {args.trait} 不在可用性状列表中")
    else:
        # 评估所有性状
        evaluator.run_all_traits() 