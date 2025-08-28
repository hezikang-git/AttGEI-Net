import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from utils import evaluate_model
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR, CosineAnnealingWarmRestarts

class Trainer:
    """Model trainer class"""
    def __init__(self, model, device, lr=0.001, weight_decay=1e-5):
        self.model = model
        self.device = device
        self.model.to(device)
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Learning rate scheduler will be set in the train method, as total steps are needed
        self.scheduler = None
        
        self.mse_criterion = torch.nn.MSELoss()
        self.mae_criterion = torch.nn.L1Loss()  # Add MAE loss, less sensitive to outliers
        
        # Increase loss function weights - significantly increase correlation coefficient loss weight
        self.pearson_weight = 2.0  # Increase correlation coefficient loss weight
        self.mae_weight = 0.3      # Add MAE loss weight
        
        # Number of steps for learning rate warm-up phase
        self.warmup_steps = 0
        
        # Add gradient accumulation steps to achieve larger effective batch size
        self.gradient_accumulation_steps = 1
        self.step_count = 0
    
    def pearson_corr_loss(self, y_pred, y_true, epsilon=1e-8):
        """Optimized Pearson correlation coefficient loss function - more numerically stable"""
        # Convert tensors to 1D vectors
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        
        # 1. Calculate means
        mean_pred = torch.mean(y_pred)
        mean_true = torch.mean(y_true)
        
        # 2. Calculate differences from means
        vx = y_pred - mean_pred
        vy = y_true - mean_true
        
        # 3. Calculate numerator
        numerator = torch.sum(vx * vy)
        
        # 4. Calculate denominator, add epsilon to prevent division by zero
        denominator = (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + epsilon)
        
        # 5. Calculate correlation coefficient
        corr = numerator / denominator
        
        # 6. Apply smoothing function for more stable optimization
        # More aggressive penalty for lower correlations
        loss = 1.0 - corr * corr  # Use squared correlation coefficient, give greater rewards to high correlations
        
        return loss
    
    def combined_loss(self, y_pred, y_true):
        """Combined loss function - integrates MSE, MAE, and Pearson correlation coefficient losses"""
        mse_loss = self.mse_criterion(y_pred, y_true)
        mae_loss = self.mae_criterion(y_pred, y_true)
        pearson_loss = self.pearson_corr_loss(y_pred, y_true)
        
        # Dynamically adjust weights - if Pearson correlation is low, increase its weight
        if self.step_count > 1000:  # Use fixed weights during early training
            with torch.no_grad():
                current_corr = 1.0 - pearson_loss.item()
                # If correlation coefficient is low, increase its weight
                dynamic_pearson_weight = self.pearson_weight * (1.0 + max(0, 0.7 - current_corr))
        else:
            dynamic_pearson_weight = self.pearson_weight
        
        # Combined loss
        loss = mse_loss + self.mae_weight * mae_loss + dynamic_pearson_weight * pearson_loss
        
        return loss, mse_loss.item(), pearson_loss.item()
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        epoch_mse_loss = 0
        epoch_pearson_loss = 0
        
        self.optimizer.zero_grad()  # Zero gradients before each epoch
        
        for batch_idx, batch in enumerate(train_loader):
            # Get data
            geno = batch['genotype'].to(self.device)
            env = batch['environment'].to(self.device)
            target = batch['phenotype'].to(self.device)
            
            # Forward pass
            output = self.model(geno, env)
            
            # Calculate loss - use combined loss
            loss, mse_loss, pearson_loss = self.combined_loss(output, target)
            
            # Gradient accumulation to achieve larger batch size
            loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Accumulate statistics
            epoch_loss += loss.item() * self.gradient_accumulation_steps
            epoch_mse_loss += mse_loss
            epoch_pearson_loss += pearson_loss
            
            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Adaptive gradient clipping - dynamically adjust clipping threshold based on gradient norm
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                
                # Step count increment
                self.step_count += 1
                
                # Optimizer update
                self.optimizer.step()
                
                # Learning rate scheduler update
                if self.scheduler is not None:
                    self.scheduler.step()
                
                # Clear gradients
                self.optimizer.zero_grad()
        
        # Calculate average loss
        num_batches = len(train_loader)
        return epoch_loss / num_batches, epoch_mse_loss / num_batches, epoch_pearson_loss / num_batches
    
    def evaluate(self, val_loader):
        """Evaluate the model"""
        self.model.eval()
        val_loss = 0
        val_mse_loss = 0
        val_pearson_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Get data
                geno = batch['genotype'].to(self.device)
                env = batch['environment'].to(self.device)
                target = batch['phenotype'].to(self.device)
                
                # Forward pass
                output = self.model(geno, env)
                
                # Calculate loss - use combined loss
                loss, mse_loss, pearson_loss = self.combined_loss(output, target)
                
                val_loss += loss.item()
                val_mse_loss += mse_loss
                val_pearson_loss += pearson_loss
                
                # Collect predictions and target values
                all_preds.extend(output.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy().flatten())
        
        # Calculate evaluation metrics
        are, mse, pearson = evaluate_model(np.array(all_targets), np.array(all_preds))
        
        num_batches = len(val_loader)
        return val_loss / num_batches, val_mse_loss / num_batches, val_pearson_loss / num_batches, are, mse, pearson
    
    def train(self, train_loader, val_loader, num_epochs=150, patience=20):
        """Train the model with early stopping"""
        # Set up cosine annealing learning rate scheduler with warm restarts
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,             # 10 epochs per cycle
            T_mult=2,           # Double period length after each restart
            eta_min=1e-6        # Minimum learning rate
        )
        
        best_val_loss = float('inf')
        best_metrics = None
        patience_counter = 0
        
        # Set gradient accumulation steps
        batch_size = next(iter(train_loader))['genotype'].size(0)
        self.gradient_accumulation_steps = max(1, 256 // batch_size)  # Target effective batch size of 256
        print(f"Using gradient accumulation steps: {self.gradient_accumulation_steps}, effective batch size: {batch_size * self.gradient_accumulation_steps}")
        
        # Track training history
        train_history = []
        val_history = []
        
        # Record best model state
        best_model_state = None
        
        for epoch in range(num_epochs):
            # Train for one epoch
            train_loss, train_mse, train_pearson = self.train_epoch(train_loader)
            
            # Evaluate the model
            val_loss, val_mse, val_pearson, are, mse, pearson = self.evaluate(val_loader)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Record training history
            train_history.append((train_loss, train_mse, train_pearson))
            val_history.append((val_loss, val_mse, val_pearson, are, mse, pearson))
            
            # Print results for each epoch - more detailed information
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f} (MSE: {train_mse:.4f}, Pearson Loss: {train_pearson:.4f}), "
                  f"Val Loss: {val_loss:.4f} (MSE: {val_mse:.4f}, Pearson Loss: {val_pearson:.4f}), "
                  f"ARE: {are:.4f}, MSE: {mse:.4f}, Pearson: {pearson:.4f}, LR: {current_lr:.6f}")
            
            # Check if we need to save the best model
            # Use Pearson correlation coefficient on validation set as selection criterion
            if pearson > 0 and (best_metrics is None or pearson > best_metrics[2]):
                best_val_loss = val_loss
                best_metrics = (are, mse, pearson)
                patience_counter = 0
                
                # Save best model state
                best_model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Restore best model state
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            
        return best_metrics