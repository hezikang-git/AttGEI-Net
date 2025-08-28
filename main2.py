"""
Experiment 2: Select only 5 locations (10 environments) for the training set, using all genotype data.
Predict phenotypes for all genotypes in the remaining 1 location (2 environments).
Perform 5:1 cross-validation by location. Generate 6 training and test set combinations.
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import pandas as pd
from utils import prepare_data_experiment2, CropDataset, extend_environment_features, create_stratified_folds
from models import DeepGxE, AttentionGxE, CrossAttentionGxE
from trainer import Trainer

def create_ensemble_models(geno_dim, env_dim, device):
    """Create model ensemble"""
    models = []
    
    # Create three different model architectures
    hidden_dim = 512
    dropout = 0.2
    
    # Model 1: Cross-attention model
    models.append(CrossAttentionGxE(geno_dim, env_dim, hidden_dim=hidden_dim, 
                                      num_heads=8, dropout=dropout).to(device))
    
    # Model 2: Deep MLP model
    models.append(DeepGxE(geno_dim, env_dim, hidden_dim=hidden_dim, 
                         dropout=dropout).to(device))
    
    # Model 3: Standard attention model
    models.append(AttentionGxE(geno_dim, env_dim, hidden_dim=hidden_dim, 
                              num_heads=4, dropout=dropout).to(device))
    
    return models

def ensemble_predict(models, test_loader, device):
    """Ensemble prediction"""
    all_targets = []
    all_ensemble_preds = []
    
    # Collect predictions from each model
    model_predictions = [[] for _ in range(len(models))]
    
    # Predict for each batch in the test set
    with torch.no_grad():
        for batch in test_loader:
            geno = batch['genotype'].to(device)
            env = batch['environment'].to(device)
            target = batch['phenotype'].to(device)
            
            # Collect target values
            all_targets.extend(target.cpu().numpy().flatten())
            
            # Collect predictions from each model
            for i, model in enumerate(models):
                model.eval()
                pred = model(geno, env)
                model_predictions[i].extend(pred.cpu().numpy().flatten())
    
    # Calculate ensemble prediction for each sample
    for i in range(len(all_targets)):
        # Get predictions from each model and average them
        preds = [model_predictions[j][i] for j in range(len(models))]
        ensemble_pred = np.mean(preds)
        all_ensemble_preds.append(ensemble_pred)
    
    return np.array(all_targets), np.array(all_ensemble_preds)

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
    geno_data, env_data, pheno_data, _ = prepare_data_experiment2(data_folder, seed=seed, standardize=True)
    
    # Apply feature engineering to extend environmental features
    extended_env_data = extend_environment_features(env_data)
    print("Feature engineering for environmental features completed")
    
    # Use stratified cross-validation
    cv_splits = create_stratified_folds(extended_env_data, n_splits=6, seed=seed)
    print(f"Created {len(cv_splits)} stratified cross-validation folds")
    
    # Get input dimensions
    geno_dim = geno_data.shape[1]
    env_sample = list(extended_env_data.values())[0]
    env_dim = env_sample.size  # Flattened dimension of environmental features (after extension)
    
    print(f"Genotype dimension: {geno_dim}, Extended environment feature dimension: {env_dim}")
    
    # Store results for each cross-validation fold
    cv_results = []
    
    # Perform cross-validation
    for fold, (train_env_keys, test_env_keys) in enumerate(cv_splits):
        print(f"\nStarting cross-validation Fold {fold+1}/6 - Test location: {test_env_keys[0].split('_')[0]}")
        
        # Create training and test datasets
        train_dataset = CropDataset(geno_data, extended_env_data, pheno_data, env_locations=train_env_keys, standardize=True)
        test_dataset = CropDataset(geno_data, extended_env_data, pheno_data, env_locations=test_env_keys, standardize=True)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        # Create model ensemble
        models = create_ensemble_models(geno_dim, env_dim, device)
        
        # Train each model
        trained_models = []
        for i, model in enumerate(models):
            print(f"Training model {i+1}/{len(models)}")
            
            # Create trainer
            trainer = Trainer(model, device, lr=0.0003, weight_decay=1e-5)
            
            # Train model - increase epochs and patience
            print(f"Starting training model {i+1}/{len(models)}")
            best_metrics = trainer.train(train_loader, test_loader, num_epochs=150, patience=20)
            
            print(f"Model {i+1}/{len(models)} completed - ARE: {best_metrics[0]:.4f}, MSE: {best_metrics[1]:.4f}, Pearson: {best_metrics[2]:.4f}")
            
            # Save trained model
            trained_models.append(model)
        
        # Use ensemble model for prediction
        targets, ensemble_preds = ensemble_predict(trained_models, test_loader, device)
        
        # Calculate evaluation metrics for ensemble model
        from utils import evaluate_model
        are, mse, pearson = evaluate_model(targets, ensemble_preds)
        
        # Store results
        test_location = test_env_keys[0].split('_')[0]  # Extract test location
        cv_results.append({
            'fold': fold + 1,
            'test_location': test_location,
            'are': are,
            'mse': mse,
            'pearson': pearson
        })
        
        print(f"Fold {fold+1}/6 ensemble results - Test location: {test_location}, ARE: {are:.4f}, MSE: {mse:.4f}, Pearson: {pearson:.4f}")
    
    # Calculate average results
    avg_are = np.mean([r['are'] for r in cv_results])
    avg_mse = np.mean([r['mse'] for r in cv_results])
    avg_pearson = np.mean([r['pearson'] for r in cv_results])
    
    # Print final results
    print("\n=== Experiment 2 Final Results ===")
    print(f"Average ARE: {avg_are:.4f}")
    print(f"Average MSE: {avg_mse:.4f}")
    print(f"Average Pearson correlation: {avg_pearson:.4f}")
    
    # Print results for each location
    print("\nTest results by location:")
    for r in cv_results:
        print(f"Location {r['test_location']} - ARE: {r['are']:.4f}, MSE: {r['mse']:.4f}, Pearson: {r['pearson']:.4f}")
    
    # Save results to CSV file
    results_df = pd.DataFrame(cv_results)
    results_df.to_csv("experiment2_extended_results.csv", index=False)
    print("Results saved to experiment2_extended_results.csv")

if __name__ == "__main__":
    main()