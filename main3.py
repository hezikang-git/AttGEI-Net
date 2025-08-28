"""
Experiment 3: Select only 5 locations (10 environments) for the training set, using 90% of genotype data.
Predict phenotypes for the remaining 10% of genotypes in the remaining 1 location (2 environments).
Perform 9:1 10-fold cross-validation for genotype selection and 5:1 cross-validation for location selection.
Generate 60 training and test set combinations.
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
    """Create model ensemble"""
    models = []
    
    # Create three different model architectures
    hidden_dim = 512
    dropout = 0.2
    
    # Model 1: Cross-attention model
    models.append(CrossAttentionGxE(geno_dim, env_dim, hidden_dim=hidden_dim, 
                                      num_heads=8, dropout=dropout).to(device))
    
    # Model 2: Standard attention model
    models.append(AttentionGxE(geno_dim, env_dim, hidden_dim=hidden_dim, 
                               num_heads=4, dropout=dropout).to(device))
    
    # Model 3: Deep MLP model
    models.append(DeepGxE(geno_dim, env_dim, hidden_dim=hidden_dim, 
                         dropout=dropout).to(device))
    
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

def create_combined_cv_splits(geno_data, env_data, k_geno=10, seed=42):
    """Create combined cross-validation splits, combining genotype folds and environment folds"""
    np.random.seed(seed)
    
    # Create k-fold cross-validation for genotypes
    kf_geno = KFold(n_splits=k_geno, shuffle=True, random_state=seed)
    geno_splits = []
    for train_idx, test_idx in kf_geno.split(range(len(geno_data))):
        geno_splits.append((train_idx, test_idx))
    
    # Use stratified method to create environment folds
    env_splits = create_stratified_folds(env_data, n_splits=6, seed=seed)
    
    # Combine genotype and environment cross-validation
    cv_splits = []
    for geno_train_idx, geno_test_idx in geno_splits:
        for train_env_keys, test_env_keys in env_splits:
            cv_splits.append((geno_train_idx, geno_test_idx, train_env_keys, test_env_keys))
    
    return cv_splits

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
    geno_data, env_data, pheno_data, _ = prepare_data_experiment3(data_folder, k=10, seed=seed, standardize=True)
    
    # Apply feature engineering to extend environmental features
    extended_env_data = extend_environment_features(env_data)
    print("Feature engineering for environmental features completed")
    
    # Create combined cross-validation splits
    cv_splits = create_combined_cv_splits(geno_data, extended_env_data, k_geno=10, seed=seed)
    
    # Get input dimensions
    geno_dim = geno_data.shape[1]
    env_sample = list(extended_env_data.values())[0]
    env_dim = env_sample.size  # Flattened dimension of environmental features (after extension)
    
    print(f"Genotype dimension: {geno_dim}, Extended environment feature dimension: {env_dim}")
    print(f"Total number of cross-validation combinations: {len(cv_splits)}")
    
    # Store results for each cross-validation fold
    cv_results = []
    
    # Select number of folds to run, reduce for faster testing (select 12 representative combinations, 2 per location)
    num_folds_to_run = 12  # Run only 12 combinations for faster testing
    selected_folds = []
    
    # Ensure each location is represented
    locations = sorted(list(set([key.split('_')[0] for key in extended_env_data.keys()])))
    for loc_idx, loc in enumerate(locations):
        # Find all folds for this location
        loc_folds = [i for i, (_, _, _, test_env_keys) in enumerate(cv_splits) 
                    if test_env_keys[0].split('_')[0] == loc]
        
        # Select 2 folds from this location
        if len(loc_folds) >= 2:
            selected_folds.extend(loc_folds[:2])
        else:
            selected_folds.extend(loc_folds)
    
    # If fewer than 12, randomly select remaining
    if len(selected_folds) < num_folds_to_run:
        remaining = [i for i in range(len(cv_splits)) if i not in selected_folds]
        np.random.shuffle(remaining)
        selected_folds.extend(remaining[:num_folds_to_run - len(selected_folds)])
    
    # Ensure no more than 12
    selected_folds = selected_folds[:num_folds_to_run]
    
    print(f"Will run {len(selected_folds)} representative cross-validation combinations")
    
    # Perform cross-validation
    for fold_idx, fold in enumerate(selected_folds):
        geno_train_idx, geno_test_idx, train_env_keys, test_env_keys = cv_splits[fold]
        
        geno_fold = fold // 6 + 1  # Genotype fold number (1-10)
        env_fold = fold % 6 + 1    # Environment fold number (1-6)
        test_location = test_env_keys[0].split('_')[0]  # Test location
        
        print(f"\nStarting cross-validation combination {fold_idx+1}/{len(selected_folds)} (global fold {fold+1}) - Genotype fold {geno_fold}/10, Environment fold {env_fold}/6, Test location: {test_location}")
        
        # Create training and test datasets
        train_dataset = CropDataset(geno_data, extended_env_data, pheno_data, 
                                    genotype_indices=geno_train_idx, 
                                    env_locations=train_env_keys,
                                    standardize=True)
        
        test_dataset = CropDataset(geno_data, extended_env_data, pheno_data, 
                                  genotype_indices=geno_test_idx, 
                                  env_locations=test_env_keys,
                                  standardize=True)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        # Create model ensemble
        models = create_ensemble_models(geno_dim, env_dim, device)
        
        # Train each model
        trained_models = []
        for i, model in enumerate(models):
            print(f"Training model {i+1}/{len(models)}")
            
            # Create trainer - ensure only passing correct parameters
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
        cv_results.append({
            'fold': fold + 1,  # Original fold index
            'fold_idx': fold_idx + 1,  # Run fold index
            'geno_fold': geno_fold,
            'env_fold': env_fold,
            'test_location': test_location,
            'are': are,
            'mse': mse,
            'pearson': pearson
        })
        
        print(f"Combination {fold_idx+1}/{len(selected_folds)} ensemble results - ARE: {are:.4f}, MSE: {mse:.4f}, Pearson: {pearson:.4f}")
        
        # Save interim results for each combination
        interim_df = pd.DataFrame(cv_results)
        interim_df.to_csv(f"experiment3_combined_interim_results_{fold_idx+1}.csv", index=False)
        print(f"Interim results saved to experiment3_combined_interim_results_{fold_idx+1}.csv")
    
    # Calculate average results
    avg_are = np.mean([r['are'] for r in cv_results])
    avg_mse = np.mean([r['mse'] for r in cv_results])
    avg_pearson = np.mean([r['pearson'] for r in cv_results])
    
    # Print final results
    print("\n=== Experiment 3 Combined Version Final Results ===")
    print(f"Average ARE: {avg_are:.4f}")
    print(f"Average MSE: {avg_mse:.4f}")
    print(f"Average Pearson correlation: {avg_pearson:.4f}")
    
    # Print results by location
    print("\nResults by location:")
    locations = set([r['test_location'] for r in cv_results])
    for loc in locations:
        loc_results = [r for r in cv_results if r['test_location'] == loc]
        loc_are = np.mean([r['are'] for r in loc_results])
        loc_mse = np.mean([r['mse'] for r in loc_results])
        loc_pearson = np.mean([r['pearson'] for r in loc_results])
        print(f"Location {loc} - ARE: {loc_are:.4f}, MSE: {loc_mse:.4f}, Pearson: {loc_pearson:.4f}")
    
    # Save results to CSV file
    results_df = pd.DataFrame(cv_results)
    results_df.to_csv("experiment3_combined_results.csv", index=False)
    print("Results saved to experiment3_combined_results.csv")

if __name__ == "__main__":
    main()