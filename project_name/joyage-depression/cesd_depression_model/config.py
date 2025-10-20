"""
Configuration settings for CESD Depression Prediction Model
"""

# 解决Windows上的CPU核心检测警告
import os
os.environ['LOKY_MAX_CPU_COUNT'] = '24'

# 分类变量列表 (🔑 原始高性能版本的实际配置)
CATEGORICAL_VARS = [
    'ragender', 'raeducl', 'mstath', 'rural', 'shlta', 'hlthlm', 'hibpe', 'diabe',
    'cancre', 'lunge', 'hearte', 'stroke', 'arthre', 'livere', 'drinkev', 'smokev',
    'stayhospital', 'hipriv', 'momliv', 'dadliv', 'ramomeducl', 'radadeducl',
    'lvnear', 'kcntf', 'socwk', 'work', 'pubpen', 'peninc',
    'ftrsp', 'ftrkids', 'painfr', 'fall',
    'adlfive',  # 新增分类变量
]

# 需要排除的ID相关变量
EXCLUDED_VARS = ['ID', 'householdID', 'interview_year']

# 数值变量列表 (🔑 原始高性能版本的实际配置)
NUMERICAL_VARS = [
    'agey', 'child', 'comparable_hexp', 'hhres',
    'comparable_exp', 'comparable_itearn', 'comparable_frec', 
    'comparable_tgiv', 'comparable_ipubpen'  # 新增数值变量
]

# 机器学习模型超参数搜索空间定义
MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    },
    'gradient_boosting': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    # 🚫 用户要求：完全禁用SVC模型（避免长时间训练）
    # 'svc': {
    #     'base_estimator__C': [0.1, 1, 10],  # LinearSVC的C参数
    #     'base_estimator__max_iter': [5000]  # 固定迭代次数
    # },
    'logistic_regression': {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'max_iter': [1000, 2000]
    },
    'xgboost': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    },
    'lightgbm': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, -1],
        'learning_rate': [0.01, 0.1, 0.2],
        'num_leaves': [31, 50, 100],
        'subsample': [0.8, 0.9, 1.0]
    }
}

# 交叉验证设置（使用10折获得最准确评估）
CV_SETTINGS = {
    'outer_splits': 10,  # 10折交叉验证
    'inner_splits': 10,
    'random_state': 42,
    'n_splits': 10  # 统一使用10折
}

# 文件路径设置
FILE_PATHS = {
    'model_save_dir': 'saved_models',
    'results_dir': 'results',
    'plots_dir': 'plots',
    'logs_dir': 'logs'
}

# SMOTE参数
SMOTE_PARAMS = {
    'random_state': 42,
    'k_neighbors': 5,
    'sampling_strategy': 'auto'
}

# 评估指标设置
METRICS = [
    'accuracy',
    'precision',
    'recall',
    'f1',
    'roc_auc',
    'average_precision',
    'brier_score'
]

# ========== 资源自适应并行参数 ==========
import os
try:
    import psutil
    def get_total_memory_gb():
        mem = psutil.virtual_memory()
        return mem.total / (1024 ** 3)
except ImportError:
    def get_total_memory_gb():
        # psutil不可用时，返回一个保守值
        return 8

def get_optimal_n_jobs(per_job_gb=3, max_ratio=0.5):
    """
    计算最优并行作业数
    
    参数:
    ----
    per_job_gb : float, 默认3
        每个作业预估内存需求(GB)
    max_ratio : float, 默认0.5
        最大使用CPU核心数的比例
    """
    total_memory_gb = get_total_memory_gb()
    cpu_count = os.cpu_count() or 1
    
    # 基于内存限制的最大作业数
    memory_based_jobs = int(total_memory_gb // per_job_gb)
    
    # 基于CPU限制的最大作业数（不超过CPU核心数的50%）
    cpu_based_jobs = max(1, int(cpu_count * max_ratio))
    
    # 为大数据集设置更保守的上限
    conservative_limit = min(8, cpu_count // 2) if cpu_count > 8 else min(4, cpu_count)
    
    # 取三者最小值
    optimal_jobs = max(1, min(memory_based_jobs, cpu_based_jobs, conservative_limit))
    
    return optimal_jobs

# 使用更保守的参数
N_JOBS = get_optimal_n_jobs(per_job_gb=3, max_ratio=0.4)

# ========== 优化完成 ==========
# 方案1：智能N_JOBS计算已启用
# 当前N_JOBS值会根据系统配置自动计算
# ========================================= 