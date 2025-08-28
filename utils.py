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
    """加载基因型数据"""
    return np.loadtxt(file_path)

def load_environment_data(env_folder):
    """加载环境型数据"""
    env_files = os.listdir(env_folder)
    env_data = {}
    
    for file in env_files:
        if file.endswith('.xlsx'):
            location = file.split('.')[0]  # 获取地点和年份信息
            file_path = os.path.join(env_folder, file)
            df = pd.read_excel(file_path)
            # 去掉第一列（地点编号），保留环境指标
            env_features = df.iloc[:, 1:].values
            env_data[location] = env_features
    
    return env_data

def load_phenotype_data(pheno_folder):
    """加载表型数据"""
    pheno_files = os.listdir(pheno_folder)
    pheno_data = {}
    
    for file in pheno_files:
        if file.endswith('.txt'):
            location = file.split('.')[0]  # 获取地点和年份信息
            file_path = os.path.join(pheno_folder, file)
            # 读取单列表型数据
            pheno_values = np.loadtxt(file_path)
            pheno_data[location] = pheno_values
    
    return pheno_data

def standardize_genotype(genotype_data):
    """标准化基因型数据"""
    # 创建标准化器
    scaler = StandardScaler()
    # 执行标准化并返回
    return scaler.fit_transform(genotype_data)

def standardize_environment(env_data):
    """标准化环境数据"""
    # 创建一个新的字典存储标准化后的环境数据
    standardized_env_data = {}
    
    # 首先将所有环境数据展平并合并，以便计算全局均值和标准差
    all_env_features = np.concatenate([env.flatten() for env in env_data.values()])
    
    # 计算全局均值和标准差
    mean = np.mean(all_env_features)
    std = np.std(all_env_features)
    
    # 对每个环境使用相同的均值和标准差进行标准化
    for key, env in env_data.items():
        standardized_env_data[key] = (env - mean) / (std + 1e-8)
    
    return standardized_env_data

class CropDataset(Dataset):
    """作物数据集类"""
    def __init__(self, genotype, environment, phenotype, genotype_indices=None, env_locations=None, standardize=True):
        # 如果需要标准化，则对基因型数据进行标准化
        if standardize:
            self.genotype = standardize_genotype(genotype)
            self.environment = standardize_environment(environment)
        else:
            self.genotype = genotype
            self.environment = environment
        
        self.phenotype = phenotype
        self.env_keys = list(environment.keys()) if env_locations is None else env_locations
        self.genotype_indices = genotype_indices if genotype_indices is not None else np.arange(len(genotype))
        
        # 生成所有基因型和环境组合
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
        
        # 将环境特征转换为一维向量
        env_features_flat = env_features.flatten()
        
        return {
            'genotype': torch.FloatTensor(geno_features),
            'environment': torch.FloatTensor(env_features_flat),
            'phenotype': torch.FloatTensor([pheno_value])
        }

def evaluate_model(y_true, y_pred):
    """计算模型评估指标：ARE、MSE和皮尔逊相关系数"""
    # 过滤掉任何无效值
    valid_indices = ~np.isnan(y_true) & ~np.isnan(y_pred) & ~np.isinf(y_true) & ~np.isinf(y_pred)
    y_true = y_true[valid_indices]
    y_pred = y_pred[valid_indices]
    
    # 确保有足够的数据点
    if len(y_true) < 2:
        return np.nan, np.nan, np.nan
    
    # 计算绝对相对误差（ARE），添加小值防止除零
    are = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8)))
    
    # 计算均方误差（MSE）
    mse = mean_squared_error(y_true, y_pred)
    
    # 计算皮尔逊相关系数
    pearson_corr, _ = pearsonr(y_true, y_pred)
    
    return are, mse, pearson_corr

def prepare_data_experiment1(data_folder, k=10, seed=42, standardize=True):
    """准备第一个实验的数据"""
    # 加载数据
    geno_data = load_genotype_data(os.path.join(data_folder, 'genodata.txt'))
    env_data = load_environment_data(os.path.join(data_folder, 'enviroment'))
    pheno_data = load_phenotype_data(os.path.join(data_folder, 'characteristic'))
    
    # 设置随机种子
    np.random.seed(seed)
    
    # 创建10折交叉验证
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    
    # 存储所有训练和测试组合
    cv_splits = []
    for train_idx, test_idx in kf.split(range(len(geno_data))):
        cv_splits.append((train_idx, test_idx))
    
    return geno_data, env_data, pheno_data, cv_splits

def prepare_data_experiment2(data_folder, seed=42, standardize=True):
    """准备第二个实验的数据"""
    # 加载数据
    geno_data = load_genotype_data(os.path.join(data_folder, 'genodata.txt'))
    env_data = load_environment_data(os.path.join(data_folder, 'enviroment'))
    pheno_data = load_phenotype_data(os.path.join(data_folder, 'characteristic'))
    
    # 设置随机种子
    np.random.seed(seed)
    
    # 提取地点信息（不包括年份）
    locations = set([key.split('_')[0] for key in env_data.keys()])
    
    # 创建6折交叉验证（每次留出一个地点）
    cv_splits = []
    for test_loc in locations:
        train_locs = [loc for loc in locations if loc != test_loc]
        
        # 获取完整的训练和测试环境键
        train_env_keys = []
        for loc in train_locs:
            train_env_keys.extend([key for key in env_data.keys() if key.startswith(loc)])
        
        test_env_keys = [key for key in env_data.keys() if key.startswith(test_loc)]
        
        cv_splits.append((train_env_keys, test_env_keys))
    
    return geno_data, env_data, pheno_data, cv_splits

def prepare_data_experiment3(data_folder, k=10, seed=42, standardize=True):
    """准备第三个实验的数据"""
    # 加载数据
    geno_data = load_genotype_data(os.path.join(data_folder, 'genodata.txt'))
    env_data = load_environment_data(os.path.join(data_folder, 'enviroment'))
    pheno_data = load_phenotype_data(os.path.join(data_folder, 'characteristic'))
    
    # 设置随机种子
    np.random.seed(seed)
    
    # 提取地点信息（不包括年份）
    locations = set([key.split('_')[0] for key in env_data.keys()])
    
    # 创建基因型的10折交叉验证
    kf_geno = KFold(n_splits=k, shuffle=True, random_state=seed)
    geno_splits = []
    for train_idx, test_idx in kf_geno.split(range(len(geno_data))):
        geno_splits.append((train_idx, test_idx))
    
    # 创建地点的6折交叉验证
    env_splits = []
    for test_loc in locations:
        train_locs = [loc for loc in locations if loc != test_loc]
        
        # 获取完整的训练和测试环境键
        train_env_keys = []
        for loc in train_locs:
            train_env_keys.extend([key for key in env_data.keys() if key.startswith(loc)])
        
        test_env_keys = [key for key in env_data.keys() if key.startswith(test_loc)]
        
        env_splits.append((train_env_keys, test_env_keys))
    
    # 组合基因型和环境的交叉验证
    cv_splits = []
    for geno_train_idx, geno_test_idx in geno_splits:
        for train_env_keys, test_env_keys in env_splits:
            cv_splits.append((geno_train_idx, geno_test_idx, train_env_keys, test_env_keys))
    
    return geno_data, env_data, pheno_data, cv_splits

def extend_environment_features(env_data):
    """
    扩展环境特征，添加统计特征和非线性特征
    
    Args:
        env_data: 原始环境数据字典
        
    Returns:
        扩展后的环境数据字典
    """
    extended_env_data = {}
    
    for key, env in env_data.items():
        # 原始特征
        original_features = env
        
        # 1. 添加统计特征
        mean_features = np.mean(original_features, axis=0, keepdims=True)
        std_features = np.std(original_features, axis=0, keepdims=True)
        max_features = np.max(original_features, axis=0, keepdims=True)
        min_features = np.min(original_features, axis=0, keepdims=True)
        
        # 2. 添加排名特征 - 反映相对位置
        rank_features = np.zeros_like(original_features)
        for i in range(original_features.shape[1]):
            rank_features[:, i] = np.argsort(np.argsort(original_features[:, i])) / original_features.shape[0]
        
        # 3. 添加非线性变换特征
        # 对数变换 - 处理可能的负值
        log_features = np.log1p(np.abs(original_features)) * np.sign(original_features)
        # 平方变换 - 保留符号
        square_features = original_features ** 2 * np.sign(original_features)
        
        # 合并所有特征
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
    创建分层交叉验证折，使测试集更具代表性
    
    Args:
        env_data: 环境数据字典
        n_splits: 分割数量
        seed: 随机种子
        
    Returns:
        交叉验证折列表，每一项包含训练环境键和测试环境键
    """
    # 提取环境特征的统计信息作为分层依据
    np.random.seed(seed)
    
    # 设置随机种子
    np.random.seed(seed)
    
    # 提取地点信息（不包括年份）
    locations = sorted(list(set([key.split('_')[0] for key in env_data.keys()])))
    
    # 计算每个位置的环境特征统计信息
    loc_features = []
    for loc in locations:
        # 获取该位置的所有环境数据
        loc_env_data = [env_data[key] for key in env_data.keys() if key.startswith(loc)]
        
        # 合并该位置的所有环境数据
        combined_env = np.concatenate(loc_env_data, axis=0)
        
        # 计算统计特征
        mean_val = np.mean(combined_env)
        std_val = np.std(combined_env)
        max_val = np.max(combined_env)
        min_val = np.min(combined_env)
        
        # 创建特征向量
        loc_feature = [mean_val, std_val, max_val - min_val]
        loc_features.append(loc_feature)
    
    # 使用K-means进行聚类，创建伪标签
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_splits, random_state=seed)
    labels = kmeans.fit_predict(loc_features)
    
    # 基于聚类标签创建分层折
    folds = []
    for i in range(n_splits):
        test_locs = [locations[j] for j in range(len(locations)) if labels[j] == i]
        train_locs = [locations[j] for j in range(len(locations)) if labels[j] != i]
        
        # 获取完整的训练和测试环境键
        train_env_keys = []
        for loc in train_locs:
            train_env_keys.extend([key for key in env_data.keys() if key.startswith(loc)])
        
        test_env_keys = []
        for loc in test_locs:
            test_env_keys.extend([key for key in env_data.keys() if key.startswith(loc)])
        
        folds.append((train_env_keys, test_env_keys))
    
    return folds 