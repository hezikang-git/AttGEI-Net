import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from utils import evaluate_model
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR, CosineAnnealingWarmRestarts

class Trainer:
    """模型训练器类"""
    def __init__(self, model, device, lr=0.001, weight_decay=1e-5):
        self.model = model
        self.device = device
        self.model.to(device)
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # 学习率调度器将在train方法中设置，因为需要知道总步数
        self.scheduler = None
        
        self.mse_criterion = torch.nn.MSELoss()
        self.mae_criterion = torch.nn.L1Loss()  # 添加MAE损失，对异常值不敏感
        
        # 增加损失函数权重 - 大幅提高相关系数损失的权重
        self.pearson_weight = 2.0  # 增加相关系数损失的权重
        self.mae_weight = 0.3      # 添加MAE损失的权重
        
        # 学习率预热阶段的步数
        self.warmup_steps = 0
        
        # 添加梯度积累步数，实现更大的等效批次大小
        self.gradient_accumulation_steps = 1
        self.step_count = 0
    
    def pearson_corr_loss(self, y_pred, y_true, epsilon=1e-8):
        """优化版皮尔逊相关系数损失函数 - 数值更稳定"""
        # 将张量转换为一维向量
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        
        # 1. 计算均值
        mean_pred = torch.mean(y_pred)
        mean_true = torch.mean(y_true)
        
        # 2. 计算各自与均值的差
        vx = y_pred - mean_pred
        vy = y_true - mean_true
        
        # 3. 计算分子
        numerator = torch.sum(vx * vy)
        
        # 4. 计算分母，添加epsilon防止除零
        denominator = (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + epsilon)
        
        # 5. 计算相关系数
        corr = numerator / denominator
        
        # 6. 应用平滑函数，使优化更稳定
        # 针对较低的相关性进行更激进的惩罚
        loss = 1.0 - corr * corr  # 使用平方相关系数，对高相关性给予更大奖励
        
        return loss
    
    def combined_loss(self, y_pred, y_true):
        """组合损失函数 - 结合MSE、MAE和皮尔逊相关系数损失"""
        mse_loss = self.mse_criterion(y_pred, y_true)
        mae_loss = self.mae_criterion(y_pred, y_true)
        pearson_loss = self.pearson_corr_loss(y_pred, y_true)
        
        # 动态调整权重 - 如果皮尔逊相关系数低，增加其权重
        if self.step_count > 1000:  # 训练初期使用固定权重
            with torch.no_grad():
                current_corr = 1.0 - pearson_loss.item()
                # 如果相关系数低，增加其权重
                dynamic_pearson_weight = self.pearson_weight * (1.0 + max(0, 0.7 - current_corr))
        else:
            dynamic_pearson_weight = self.pearson_weight
        
        # 组合损失
        loss = mse_loss + self.mae_weight * mae_loss + dynamic_pearson_weight * pearson_loss
        
        return loss, mse_loss.item(), pearson_loss.item()
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        epoch_loss = 0
        epoch_mse_loss = 0
        epoch_pearson_loss = 0
        
        self.optimizer.zero_grad()  # 在每个epoch前清零梯度
        
        for batch_idx, batch in enumerate(train_loader):
            # 获取数据
            geno = batch['genotype'].to(self.device)
            env = batch['environment'].to(self.device)
            target = batch['phenotype'].to(self.device)
            
            # 前向传播
            output = self.model(geno, env)
            
            # 计算损失 - 使用组合损失
            loss, mse_loss, pearson_loss = self.combined_loss(output, target)
            
            # 梯度累积实现更大的批次大小
            loss = loss / self.gradient_accumulation_steps
            
            # 反向传播
            loss.backward()
            
            # 累计统计
            epoch_loss += loss.item() * self.gradient_accumulation_steps
            epoch_mse_loss += mse_loss
            epoch_pearson_loss += pearson_loss
            
            # 梯度累积
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # 自适应梯度裁剪 - 根据梯度范数动态调整裁剪阈值
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                
                # 步数累加
                self.step_count += 1
                
                # 优化器更新
                self.optimizer.step()
                
                # 学习率调度器更新
                if self.scheduler is not None:
                    self.scheduler.step()
                
                # 清空梯度
                self.optimizer.zero_grad()
        
        # 计算平均损失
        num_batches = len(train_loader)
        return epoch_loss / num_batches, epoch_mse_loss / num_batches, epoch_pearson_loss / num_batches
    
    def evaluate(self, val_loader):
        """评估模型"""
        self.model.eval()
        val_loss = 0
        val_mse_loss = 0
        val_pearson_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                # 获取数据
                geno = batch['genotype'].to(self.device)
                env = batch['environment'].to(self.device)
                target = batch['phenotype'].to(self.device)
                
                # 前向传播
                output = self.model(geno, env)
                
                # 计算损失 - 使用组合损失
                loss, mse_loss, pearson_loss = self.combined_loss(output, target)
                
                val_loss += loss.item()
                val_mse_loss += mse_loss
                val_pearson_loss += pearson_loss
                
                # 收集预测和目标值
                all_preds.extend(output.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy().flatten())
        
        # 计算评估指标
        are, mse, pearson = evaluate_model(np.array(all_targets), np.array(all_preds))
        
        num_batches = len(val_loader)
        return val_loss / num_batches, val_mse_loss / num_batches, val_pearson_loss / num_batches, are, mse, pearson
    
    def train(self, train_loader, val_loader, num_epochs=150, patience=20):
        """训练模型并进行早停"""
        # 设置余弦退火学习率调度器，带热重启
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,             # 每10个epoch一个周期
            T_mult=2,           # 每次重启后周期长度翻倍
            eta_min=1e-6        # 最小学习率
        )
        
        best_val_loss = float('inf')
        best_metrics = None
        patience_counter = 0
        
        # 设置梯度累积步数
        batch_size = next(iter(train_loader))['genotype'].size(0)
        self.gradient_accumulation_steps = max(1, 256 // batch_size)  # 目标等效批次大小为256
        print(f"使用梯度累积步数: {self.gradient_accumulation_steps}, 等效批次大小: {batch_size * self.gradient_accumulation_steps}")
        
        # 记录训练历史
        train_history = []
        val_history = []
        
        # 记录最佳模型状态
        best_model_state = None
        
        for epoch in range(num_epochs):
            # 训练一个epoch
            train_loss, train_mse, train_pearson = self.train_epoch(train_loader)
            
            # 评估模型
            val_loss, val_mse, val_pearson, are, mse, pearson = self.evaluate(val_loader)
            
            # 获取当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录训练历史
            train_history.append((train_loss, train_mse, train_pearson))
            val_history.append((val_loss, val_mse, val_pearson, are, mse, pearson))
            
            # 打印每个epoch的结果 - 更详细的信息
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f} (MSE: {train_mse:.4f}, Pearson Loss: {train_pearson:.4f}), "
                  f"Val Loss: {val_loss:.4f} (MSE: {val_mse:.4f}, Pearson Loss: {val_pearson:.4f}), "
                  f"ARE: {are:.4f}, MSE: {mse:.4f}, Pearson: {pearson:.4f}, LR: {current_lr:.6f}")
            
            # 检查是否需要保存最佳模型
            # 使用验证集上的皮尔逊相关系数作为选择标准
            if pearson > 0 and (best_metrics is None or pearson > best_metrics[2]):
                best_val_loss = val_loss
                best_metrics = (are, mse, pearson)
                patience_counter = 0
                
                # 保存最佳模型状态
                best_model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # 恢复最佳模型状态
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            
        return best_metrics 