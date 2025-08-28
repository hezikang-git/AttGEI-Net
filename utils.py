import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

def load_genotype_data(file_path):
    """Load genotype data"""
    return np.loadtxt(file_path)

def load_environment_data(env_folder):
    """Load environmental data"""
    env_files = os.listdir(env_folder)
    env_data = {}
    
    for file in env_files:
        if file.endswith('.xlsx'):
            location = file.split('.')[0]  # Get location and year information
            file_path = os.path.join(env_folder, file)
            df = pd.read_excel(file_path)
            # Remove the first column (location number), keep environmental indicators
            env_features = df.iloc[:, 1:].values
            env_data[location] = env_features
    
    return env_data

def load_phenotype_data(pheno_folder):
    """Load phenotype data"""
    pheno_files = os.listdir(pheno_folder)
    pheno_data = {}
    
    for file in pheno_files:
        if file.endswith('.txt'):
            location = file.split('.')[0]  # Get location and year information
            file_path = os.path.join(pheno_folder, file)
            # Read single column phenotype data
            pheno_values = np.loadtxt(file_path)
            pheno_data[location] = pheno_values
    
    return pheno_data

def standardize_genotype(genotype_data):
    """Standardize genotype data"""
    # Create standardizer
    scaler = StandardScaler()
    # Perform standardization and return
    return scaler.fit_transform(genotype_data)

def standardize_environment(env_data):
    """Standardize environment data"""
    # Create a new dictionary to store standardized environment data
    standardized_env_data = {}
    
    # First flatten and merge all environment data to calculate global mean and standard deviation
    all_env_features = np.concatenate([env.flatten() for env in env_data.values()])
    
    # Calculate global mean and standard deviation
    mean = np.mean(all_env_features)
    std = np.std(all_env_features)
    
    # Standardize each environment using the same mean and standard deviation
    for key, env in env_data.items():
        standardized_env_data[key] = (env - mean) / (std + 1e-8)
    
    return standardized_env_data

class CropDataset(Dataset):
    """Crop dataset class"""
    def __init__(self, genotype, environment, phenotype, genotype_indices=None, env_locations=None, standardize=True):
        # Standardize genotype data if needed
        if standardize:
            self.genotype = standardize_genotype(genotype)
            self.environment = standardize_environment(environment)
        else:
            self.genotype = genotype
            self.environment = environment
        
        self.phenotype = phenotype
        self.env_keys = list(environment.keys()) if env_locations is None else env_locations
        self.genotype_indices = genotype_indices if genotype_indices is not None else np.arange(len(genotype))
        
        # Generate all genotype and environment combinations
        self.combinations = []
        for idx in self.genotype_indices:
            for env_key in self.env_keys:
                self.combinations.append((idx, env_key))
    
    def __len__(self):
        return len(self.combinations)
    
    def __getitem__(self, idx):
        geno_idx, env_key = self.combinations[idx]
        geno_features = self.genotype[geno_idx]
        env_features = self.environment[env_key]
        pheno_value = self.phenotype[env_key][geno_idx]
        
        # Convert environment features to a 1D vector
        env_features_flat = env_features.flatten()
        
        return {
            'genotype': torch.FloatTensor(geno_features),
            'environment': torch.FloatTensor(env_features_flat),
            'phenotype': torch.FloatTensor([pheno_value])
        }

def evaluate_model(y_true, y_pred):
    """Calculate model evaluation metrics: ARE, MSE, and Pearson correlation coefficient"""
    # Filter out any invalid values
    valid_indices = ~np.isnan(y_true) & ~np.isnan(y_pred) & ~np.isinf(y_true) & ~np.isinf(y_pred)
    y_true = y_true[valid_indices]
    y_pred = y_pred[valid_indices]
    
    # Ensure there are enough data points
    if len(y_true) < 2:
        return np.nan, np.nan, np.nan
    
    # Calculate Absolute Relative Error (ARE), add small value to prevent division by zero
    are = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8)))
    
    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_true, y_pred)
    
    # Calculate Pearson correlation coefficient
    pearson_corr, _ = pearsonr(y_true, y_pred)
    
    return are, mse, pearson_corr

def prepare_data_experiment1(data_folder, k=10, seed=42, standardize=True):
    """Prepare data for Experiment 1"""
    # Load data
    geno_data = load_genotype_data(os.path.join(data_folder, 'genodata.txt'))
    env_data = load_environment_data(os.path.join(data_folder, 'environment'))
    pheno_data = load_phenotype_data(os.path.join(data_folder, 'characteristic'))
    
    # Set random seed
    np.random.seed(seed)
    
    # Create 10-fold cross-validation
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    
    # Store all training and testing combinations
    cv_splits = []
    for train_idx, test_idx in kf.split(range(len(geno_data))):
        cv_splits.append((train_idx, test_idx))
    
    return geno_data, env_data, pheno_data, cv_splits

def prepare_data_experiment2(data_folder, seed=42, standardize=True):
    """Prepare data for Experiment 2"""
    # Load data
    geno_data = load_genotype_data(os.path.join(data_folder, 'genodata.txt'))
    env_data = load_environment_data(os.path.join(data_folder, 'environment'))
    pheno_data = load_phenotype_data(os.path.join(data_folder, 'characteristic'))
    
    # Set random seed
    np.random.seed(seed)
    
    # Extract location information (excluding year)
    locations = set([key.split('_')[0] for key in env_data.keys()])
    
    # Create 6-fold cross-validation (leave one location out each time)
    cv_splits = []
    for test_loc in locations:
        train_locs = [loc for loc in locations if loc != test_loc]
        
        # Get complete training and testing environment keys
        train_env_keys = []
        for loc in train_locs:
            train_env_keys.extend([key for key in env_data.keys() if key.startswith(loc)])
        
        test_env_keys = [key for key in env_data.keys() if key.startswith(test_loc)]
        
        cv_splits.append((train_env_keys, test_env_keys))
    
    return geno_data, env_data, pheno_data, cv_splits

def prepare_data_experiment3(data_folder, k=10, seed=42, standardize=True):
    """Prepare data for Experiment 3"""
    # Load data
    geno_data = load_genotype_data(os.path.join(data_folder, 'genodata.txt'))
    env_data = load_environment_data(os.path.join(data_folder, 'environment'))
    pheno_data = load_phenotype_data(os.path.join(data_folder, 'characteristic'))
    
    # Set random seed
    np.random.seed(seed)
    
    # Extract location information (excluding year)
    locations = set([key.split('_')[0] for key in env_data.keys()])
    
    # Create k-fold cross-validation for genotypes
    kf_geno = KFold(n_splits=k, shuffle=True, random_state=seed)
    geno_splits = []
    for train_idx, test_idx in kf_geno.split(range(len(geno_data))):
        geno_splits.append((train_idx, test_idx))
    
    # Create 6-fold cross-validation for locations
    env_splits = []
    for test_loc in locations:
        train_locs = [loc for loc in locations if loc != test_loc]
        
        # Get complete training and testing environment keys
        train_env_keys = []
        for loc in train_locs:
            train_env_keys.extend([key for key in env_data.keys() if key.startswith(loc)])
        
        test_env_keys = [key for key in env_data.keys() if key.startswith(test_loc)]
        
        env_splits.append((train_env_keys, test_env_keys))
    
    # Combine genotype and environment cross-validation
    cv_splits = []
    for geno_train_idx, geno_test_idx in geno_splits:
        for train_env_keys, test_env_keys in env_splits:
            cv_splits.append((geno_train_idx, geno_test_idx, train_env_keys, test_env_keys))
    
    return geno_data, env_data, pheno_data, cv_splits

def extend_environment_features(env_data):
    """
    Extend environmental features by adding statistical and non-linear features
    
    Args:
        env_data: Original environment data dictionary
        
    Returns:
        Extended environment data dictionary
    """
    extended_env_data = {}
    
    for key, env in env_data.items():
        # Original features
        original_features = env
        
        # 1. Add statistical features
        mean_features = np.mean(original_features, axis=0, keepdims=True)
        std_features = np.std(original_features, axis=0, keepdims=True)
        max_features = np.max(original_features, axis=0, keepdims=True)
        min_features = np.min(original_features, axis=0, keepdims=True)
        
        # 2. Add ranking features - reflect relative position
        rank_features = np.zeros_like(original_features)
        for i in range(original_features.shape[1]):
            rank_features[:, i] = np.argsort(np.argsort(original_features[:, i])) / original_features.shape[0]
        
        # 3. Add non-linear transformation features
        # Log transformation - handle potential negative values
        log_features = np.log1p(np.abs(original_features)) * np.sign(original_features)
        # Square transformation - preserve sign
        square_features = original_features ** 2 * np.sign(original_features)
        
        # Merge all features
        extended_features = np.concatenate([
            original_features,
            mean_features.repeat(original_features.shape[0], axis=0),
            std_features.repeat(original_features.shape[0], axis=0),
            rank_features,
            log_features,
            square_features
        ], axis=1)
        
        extended_env_data[key] = extended_features
    
    return extended_env_data

def create_stratified_folds(env_data, n_splits=6, seed=42):
    """
    Create stratified cross-validation folds for more representative test sets
    
    Args:
        env_data: Environment data dictionary
        n_splits: Number of splits
        seed: Random seed
        
    Returns:
        List of cross-validation folds, each containing training and testing environment keys
    """
    # Extract statistical information from environmental features for stratification
    np.random.seed(seed)
    
    # Set random seed
    np.random.seed(seed)
    
    # Extract location information (excluding year)
    locations = sorted(list(set([key.split('_')[0] for key in env_data.keys()])))
    
    # Calculate environmental feature statistics for each location
    loc_features = []
    for loc in locations:
        # Get all environment data for this location
        loc_env_data = [env_data[key] for key in env_data.keys() if key.startswith(loc)]
        
        # Merge all environment data for this location
        combined_env = np.concatenate(loc_env_data, axis=0)
        
        # Calculate statistical features
        mean_val = np.mean(combined_env)
        std_val = np.std(combined_env)
        max_val = np.max(combined_env)
        min_val = np.min(combined_env)
        
        # Create feature vector
        loc_feature = [mean_val, std_val, max_val - min_val]
        loc_features.append(loc_feature)
    
    # Use K-means clustering to create pseudo-labels
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_splits, random_state=seed)
    labels = kmeans.fit_predict(loc_features)
    
    # Create stratified folds based on cluster labels
    folds = []
    for i in range(n_splits):
        test_locs = [locations[j] for j in range(len(locations)) if labels[j] == i]
        train_locs = [locations[j] for j in range(len(locations)) if labels[j] != i]
        
        # Get complete training and testing environment keys
        train_env_keys = []
        for loc in train_locs:
            train_env_keys.extend([key for key in env_data.keys() if key.startswith(loc)])
        
        test_env_keys = []
        for loc in test_locs:
            test_env_keys.extend([key for key in env_data.keys() if key.startswith(loc)])
        
        folds.append((train_env_keys, test_env_keys))
    
    return folds