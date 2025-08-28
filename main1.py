"""
Experiment 1: Train on all 12 environment data with 90% of genotype data.
Predict phenotypes for the remaining 10% of genotypes across all 12 environments.
Use 9:1 10-fold cross-validation for genotype selection, producing 10 training and test set combinations.
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
    # Set random seed for reproducibility
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set data path
    data_folder = "data"
    
    # Prepare data - enable standardization
    geno_data, env_data, pheno_data, cv_splits = prepare_data_experiment1(data_folder, k=10, seed=seed, standardize=True)
    
    # Get input dimensions
    geno_dim = geno_data.shape[1]
    env_sample = list(env_data.values())[0]
    env_dim = env_sample.size  # Flattened dimension of environmental features
    
    print(f"Genotype dimension: {geno_dim}, Environment dimension: {env_dim}")
    
    # Store results for each cross-validation fold
    cv_results = []
    
    # Perform cross-validation
    for fold, (train_idx, test_idx) in enumerate(cv_splits):
        print(f"\nStarting cross-validation Fold {fold+1}/10")
        
        # Create training and test datasets
        train_dataset = CropDataset(geno_data, env_data, pheno_data, genotype_indices=train_idx, standardize=True)
        test_dataset = CropDataset(geno_data, env_data, pheno_data, genotype_indices=test_idx, standardize=True)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        # Create enhanced multi-head attention model
        hidden_dim = 256  # Increase model capacity
        num_heads = 4     # Use 4 attention heads
        dropout = 0.2     # Reduce dropout to increase information flow
        model = AttentionGxE(geno_dim, env_dim, hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
        
        # Create trainer - use smaller learning rate
        trainer = Trainer(model, device, lr=0.0003, weight_decay=1e-5)
        
        # Train model - increase epochs and patience
        print(f"Starting training Fold {fold+1}/10")
        best_metrics = trainer.train(train_loader, test_loader, num_epochs=100, patience=15)
        
        # Store results
        cv_results.append({
            'fold': fold + 1,
            'are': best_metrics[0],
            'mse': best_metrics[1],
            'pearson': best_metrics[2]
        })
        
        print(f"Fold {fold+1}/10 completed - ARE: {best_metrics[0]:.4f}, MSE: {best_metrics[1]:.4f}, Pearson: {best_metrics[2]:.4f}")
    
    # Calculate average results
    avg_are = np.mean([r['are'] for r in cv_results])
    avg_mse = np.mean([r['mse'] for r in cv_results])
    avg_pearson = np.mean([r['pearson'] for r in cv_results])
    
    # Print final results
    print("\n=== Experiment 1 Final Results ===")
    print(f"Average ARE: {avg_are:.4f}")
    print(f"Average MSE: {avg_mse:.4f}")
    print(f"Average Pearson correlation: {avg_pearson:.4f}")
    
    # Save results to CSV file
    results_df = pd.DataFrame(cv_results)
    results_df.to_csv("experiment1_results.csv", index=False)
    print("Results saved to experiment1_results.csv")

if __name__ == "__main__":
    main()