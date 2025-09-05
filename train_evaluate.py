import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from models import AttGEINet
from utils import (load_genotype_data, load_environment_data, load_phenotype_data, 
                  evaluate_model, standardize_genotype, standardize_environment, 
                  CropDataset, normalize_phenotypes, inverse_normalize_predictions,
                  calculate_normalization_params)
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class TrainEvaluator:
    """Class for training and evaluating models for each trait"""
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
        
        # Set device
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")
        
        # Set random seed
        self.set_seed(seed)
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Get all trait directories
        self.trait_dirs = self.get_trait_directories()
        logging.info(f"Found the following trait directories: {self.trait_dirs}")
        
    def set_seed(self, seed):
        """Set random seed for reproducibility"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
    
    def get_trait_directories(self):
        """Get list of all trait directories"""
        # Get all trait directories from basedata
        trait_dirs = [d for d in os.listdir(self.basedata_path) 
                     if os.path.isdir(os.path.join(self.basedata_path, d))]
        return trait_dirs
    
    def create_model(self, geno_dim, env_dim):
        """Create model"""
        # Always return AttGEINet regardless of model_type parameter
        return AttGEINet(geno_dim, env_dim, hidden_dim=self.hidden_dim, num_heads=self.num_heads)
    
    def load_combined_data(self, trait_dir):
        """Load and combine data from basedata and basedata1"""
        # Load basedata
        base_geno_path = os.path.join(self.basedata_path, trait_dir, 'genodata.txt')
        base_env_dir = os.path.join(self.basedata_path, trait_dir, 'environment')
        base_char_dir = os.path.join(self.basedata_path, trait_dir, 'characteristic')
        
        base_geno_data = load_genotype_data(base_geno_path)
        base_env_data = load_environment_data(base_env_dir)
        base_pheno_data = load_phenotype_data(base_char_dir)
        
        # Check if this trait directory exists in basedata1
        base1_trait_path = os.path.join(self.basedata1_path, trait_dir)
        if os.path.exists(base1_trait_path):
            # Load basedata1
            base1_geno_path = os.path.join(base1_trait_path, 'genodata.txt')
            base1_env_dir = os.path.join(base1_trait_path, 'environment')
            base1_char_dir = os.path.join(base1_trait_path, 'characteristic')
            
            base1_geno_data = load_genotype_data(base1_geno_path)
            base1_env_data = load_environment_data(base1_env_dir)
            base1_pheno_data = load_phenotype_data(base1_char_dir)
            
            logging.info(f"Trait {trait_dir}: Combining basedata ({len(base_geno_data)} samples) and basedata1 ({len(base1_geno_data)} samples)")
            
            # Data merging is done during dataset creation, just return both datasets here
            return base_geno_data, base_env_data, base_pheno_data, base1_geno_data, base1_env_data, base1_pheno_data
        else:
            logging.info(f"Trait {trait_dir}: Using only basedata ({len(base_geno_data)} samples)")
            return base_geno_data, base_env_data, base_pheno_data, None, None, None
    
    def load_test_data(self, trait_dir):
        """Load test data"""
        # Check if this trait directory exists in testdata
        test_trait_path = os.path.join(self.testdata_path, trait_dir)
        if not os.path.exists(test_trait_path):
            logging.warning(f"Trait directory {trait_dir} not found in test data")
            return None, None, None
        
        # Load test data
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
        """Create training, validation and test datasets"""
        # Create indices for basedata
        base_indices = np.arange(len(base_geno_data))
        
        # Split into training and validation sets (80/20 split)
        train_indices, val_indices = train_test_split(
            base_indices, test_size=0.2, random_state=self.seed
        )
        
        # Create subset of phenotype data for the training set (for normalization parameter calculation)
        train_pheno_data = {}
        for env_key, pheno_values in base_pheno_data.items():
            # Get valid training indices for this environment
            valid_train_indices = [idx for idx in train_indices if idx < len(pheno_values)]
            if valid_train_indices:
                train_pheno_data[env_key] = pheno_values[valid_train_indices]
        
        # If basedata1 exists, add to training phenotype data
        if base1_geno_data is not None and base1_pheno_data is not None:
            base1_indices = np.arange(len(base1_geno_data))
            base1_train_indices, _ = train_test_split(
                base1_indices, test_size=0.2, random_state=self.seed
            )
            
            for env_key, pheno_values in base1_pheno_data.items():
                valid_train_indices = [idx for idx in base1_train_indices if idx < len(pheno_values)]
                if valid_train_indices:
                    if env_key in train_pheno_data:
                        # If environment already exists, append data
                        train_pheno_data[env_key] = np.concatenate([
                            train_pheno_data[env_key], 
                            pheno_values[valid_train_indices]
                        ])
                    else:
                        train_pheno_data[env_key] = pheno_values[valid_train_indices]
        
        # Calculate phenotype normalization parameters (using only training data)
        pheno_min, pheno_max = None, None
        if train_pheno_data:
            pheno_min, pheno_max = calculate_normalization_params(train_pheno_data)
            logging.info(f"Calculated phenotype normalization parameters from training data: min={pheno_min:.4f}, max={pheno_max:.4f}")
        
        # Create training dataset - including training portion of basedata (is_train=True)
        train_dataset = CropDataset(
            base_geno_data, base_env_data, base_pheno_data, 
            genotype_indices=train_indices, standardize=True,
            normalize_pheno=True, is_train=True
        )
        
        # If basedata1 exists, add to training set
        if base1_geno_data is not None:
            # Create indices for basedata1
            base1_indices = np.arange(len(base1_geno_data))
            
            # Split basedata1 into training and validation
            base1_train_indices, base1_val_indices = train_test_split(
                base1_indices, test_size=0.2, random_state=self.seed
            )
            
            # Create basedata1 training dataset
            base1_train_dataset = CropDataset(
                base1_geno_data, base1_env_data, base1_pheno_data, 
                genotype_indices=base1_train_indices, standardize=True,
                normalize_pheno=True, is_train=True
            )
            
            # Create basedata1 validation dataset (using normalization parameters from training)
            base1_val_dataset = CropDataset(
                base1_geno_data, base1_env_data, base1_pheno_data, 
                genotype_indices=base1_val_indices, standardize=True,
                normalize_pheno=True, pheno_min=pheno_min, pheno_max=pheno_max, is_train=False
            )
        else:
            base1_train_dataset = None
            base1_val_dataset = None
        
        # Create validation dataset - only validation portion of basedata (using normalization parameters from training)
        val_dataset = CropDataset(
            base_geno_data, base_env_data, base_pheno_data, 
            genotype_indices=val_indices, standardize=True,
            normalize_pheno=True, pheno_min=pheno_min, pheno_max=pheno_max, is_train=False
        )
        
        # Create test dataset (using normalization parameters from training)
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
        """Train the model"""
        # Set up optimizer and loss function
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=0.01)
        
        # Use Huber loss function, more robust to outliers
        criterion = torch.nn.HuberLoss(delta=1.0)
        
        # Add learning rate scheduler - stronger decay strategy
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-6
        )
        
        # Track best model
        best_val_loss = float('inf')
        best_epoch = 0
        best_model_state = None
        no_improve = 0
        
        # Training loop
        for epoch in range(self.epochs):
            # Training phase
            model.train()
            train_loss = 0
            
            for batch in train_loader:
                # Get data
                geno = batch['genotype'].to(self.device)
                env = batch['environment'].to(self.device)
                target = batch['phenotype'].to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                output = model(geno, env)
                loss = criterion(output, target)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping to prevent explosion
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation phase
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
            
            # Calculate validation metrics
            val_are, val_mse, val_pearson = evaluate_model(
                np.array(val_targets), np.array(val_preds)
            )
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Print progress
            logging.info(
                f"Epoch {epoch+1}/{self.epochs} - Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val ARE: {val_are:.4f}, "
                f"Val MSE: {val_mse:.4f}, Val Pearson: {val_pearson:.4f}"
            )
            
            # Check if this is the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_model_state = model.state_dict().copy()
                no_improve = 0
            else:
                no_improve += 1
            
            # Early stopping check
            if no_improve >= self.patience:
                logging.info(f"Early stopping - No improvement for {no_improve} epochs")
                break
        
        # Restore best model
        model.load_state_dict(best_model_state)
        logging.info(f"Training completed - Best epoch: {best_epoch+1}, Best validation loss: {best_val_loss:.4f}")
        
        return model, best_epoch, best_val_loss
    
    def test_model(self, model, test_loader):
        """Test the model and return evaluation metrics"""
        model.eval()
        test_preds = []
        test_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                geno = batch['genotype'].to(self.device)
                env = batch['environment'].to(self.device)
                target = batch['phenotype'].to(self.device)
                
                # Get 5 predictions and average for enhanced stability
                outputs = []
                for _ in range(5):  
                    output = model(geno, env)
                    outputs.append(output)
                    
                # Average multiple predictions
                avg_output = torch.mean(torch.stack(outputs), dim=0)
                
                test_preds.extend(avg_output.cpu().numpy().flatten())
                test_targets.extend(target.cpu().numpy().flatten())
        
        # Get predictions and targets in original scale
        # Check if test_loader has normalization information
        if hasattr(test_loader.dataset, 'normalize_pheno') and test_loader.dataset.normalize_pheno:
            pheno_min = test_loader.dataset.pheno_min
            pheno_max = test_loader.dataset.pheno_max
            
            # Convert back to original scale
            test_preds = inverse_normalize_predictions(np.array(test_preds), pheno_min, pheno_max)
            test_targets = inverse_normalize_predictions(np.array(test_targets), pheno_min, pheno_max)
            
            logging.info(f"Predictions converted back to original scale: min={pheno_min:.4f}, max={pheno_max:.4f}")
        else:
            test_preds = np.array(test_preds)
            test_targets = np.array(test_targets)
        
        # Calculate test metrics
        test_are, test_mse, test_pearson = evaluate_model(test_targets, test_preds)
        
        return test_are, test_mse, test_pearson, test_preds, test_targets
    
    def run_trait_evaluation(self, trait_dir):
        """Execute training, validation and testing workflow for a single trait"""
        logging.info(f"Processing trait directory: {trait_dir}")
        
        # 1. Load data
        base_geno_data, base_env_data, base_pheno_data, base1_geno_data, base1_env_data, base1_pheno_data = self.load_combined_data(trait_dir)
        test_geno_data, test_env_data, test_pheno_data = self.load_test_data(trait_dir)
        
        if test_geno_data is None:
            logging.warning(f"No test data available, skipping trait {trait_dir}")
            return None
        
        # 2. Create datasets
        train_dataset, val_dataset, test_dataset, base1_train_dataset, base1_val_dataset = self.create_datasets(
            base_geno_data, base_env_data, base_pheno_data,
            base1_geno_data, base1_env_data, base1_pheno_data,
            test_geno_data, test_env_data, test_pheno_data
        )
        
        # Merge training sets from basedata and basedata1 (if exists)
        if base1_train_dataset:
            from torch.utils.data import ConcatDataset
            combined_train_dataset = ConcatDataset([train_dataset, base1_train_dataset])
            combined_val_dataset = ConcatDataset([val_dataset, base1_val_dataset])
            
            logging.info(f"Combined datasets: Training {len(train_dataset)} + {len(base1_train_dataset)} = {len(combined_train_dataset)} samples")
            logging.info(f"Combined datasets: Validation {len(val_dataset)} + {len(base1_val_dataset)} = {len(combined_val_dataset)} samples")
            
            train_dataset = combined_train_dataset
            val_dataset = combined_val_dataset
        
        # 3. Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        
        # 4. Get model input dimensions
        geno_dim = base_geno_data.shape[1]
        env_sample = list(base_env_data.values())[0].flatten()
        env_dim = env_sample.size
        
        logging.info(f"Model input dimensions: Genotype dimension={geno_dim}, Environment dimension={env_dim}")
        
        # 5. Create model
        model = self.create_model(geno_dim, env_dim)
        model = model.to(self.device)
        
        # 6. Train model
        logging.info(f"Starting training for trait {trait_dir}")
        trained_model, best_epoch, best_val_loss = self.train_model(model, train_loader, val_loader)
        
        # 7. Test model
        logging.info(f"Starting testing for trait {trait_dir}")
        test_are, test_mse, test_pearson, test_preds, test_targets = self.test_model(
            trained_model, test_loader
        )
        
        logging.info(f"Test results - ARE: {test_are:.4f}, MSE: {test_mse:.4f}, Pearson: {test_pearson:.4f}")
        
        # 8. Save model and results
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
        
        # Save model
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
        
        # Save prediction results
        pred_df = pd.DataFrame({
            'true': test_targets,
            'pred': test_preds
        })
        pred_save_path = os.path.join(self.output_dir, f"{trait_dir}_predictions.csv")
        pred_df.to_csv(pred_save_path, index=False)
        
        return results
    
    def run_all_traits(self):
        """Execute evaluation for all traits"""
        all_results = []
        
        for trait_dir in self.trait_dirs:
            try:
                result = self.run_trait_evaluation(trait_dir)
                if result:
                    all_results.append(result)
            except Exception as e:
                logging.error(f"Error processing trait {trait_dir}: {str(e)}", exc_info=True)
        
        # Save all results
        if all_results:
            results_df = pd.DataFrame(all_results)
            results_save_path = os.path.join(self.output_dir, "all_traits_results.csv")
            results_df.to_csv(results_save_path, index=False)
            
            # Print summary results
            logging.info("\n====== All Traits Evaluation Results ======")
            logging.info(f"Average ARE: {results_df['test_are'].mean():.4f}")
            logging.info(f"Average MSE: {results_df['test_mse'].mean():.4f}")
            logging.info(f"Average Pearson: {results_df['test_pearson'].mean():.4f}")
            logging.info(f"Results saved to: {results_save_path}")
        
        return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate crop trait prediction models")
    parser.add_argument("--basedata", type=str, required=True, help="Path to basedata directory")
    parser.add_argument("--basedata1", type=str, required=True, help="Path to basedata1 directory")
    parser.add_argument("--testdata", type=str, required=True, help="Path to testdata directory")
    parser.add_argument("--output", type=str, required=True, help="Path to output directory")
    parser.add_argument("--model", type=str, default="attention",
                      choices=["deepgxe", "crossattention", "attention"],
                      help="Model type: deepgxe, crossattention, attention")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden layer dimension")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=35, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--trait", type=str, default=None, help="Specify a single trait for training (optional)")
    args = parser.parse_args()
    
    # Create evaluator
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
    
    # Start evaluation
    if args.trait:
        # Evaluate only the specified trait
        if args.trait in evaluator.trait_dirs:
            logging.info(f"Evaluating only the specified trait: {args.trait}")
            evaluator.run_trait_evaluation(args.trait)
        else:
            logging.error(f"The specified trait {args.trait} is not in the available trait list")
    else:
        # Evaluate all traits
        evaluator.run_all_traits() 