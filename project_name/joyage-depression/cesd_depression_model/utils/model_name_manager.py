"""
模型名称管理器 - 统一处理模型名称映射逻辑
"""

class ModelNameManager:
    """统一的模型名称管理器，消除重复的映射逻辑"""
    
    # 统一的模型名称映射配置
    MODEL_MAPPING = {
        'rf': 'rf',
        'gb': 'gb', 
        'xgb': 'xgb',
        'lgb': 'lgb',
        'lr': 'lr',
        'LinearSVC': 'LinearSVC',
        'extra_trees': 'extra_trees',
        'adaboost': 'adaboost',
        'catboost': 'catboost',
        'voting': 'voting',
        'stacking': 'stacking'
    }
    
    @classmethod
    def map_to_tuning_names(cls, models_dict):
        """
        将模型字典的键映射到调优使用的名称
        
        参数:
        ----
        models_dict : dict
            原始模型字典 {model_name: model_instance}
            
        返回:
        ----
        dict : 映射后的模型字典
        """
        mapped_models = {}
        for name, model in models_dict.items():
            mapped_name = cls.MODEL_MAPPING.get(name, name)
            mapped_models[mapped_name] = model
        return mapped_models
    
    @classmethod
    def map_to_original_names(cls, models_dict):
        """
        将调优后的模型字典的键映射回原始名称
        
        参数:
        ----
        models_dict : dict
            调优后的模型字典 {tuned_name: model_instance}
            
        返回:
        ----
        dict : 映射回原始名称的模型字典
        """
        reverse_map = {v: k for k, v in cls.MODEL_MAPPING.items()}
        original_models = {}
        for name, model in models_dict.items():
            original_name = reverse_map.get(name, name)
            original_models[original_name] = model
        return original_models
    
    @classmethod
    def get_original_name(cls, tuned_name):
        """
        获取调优名称对应的原始名称
        
        参数:
        ----
        tuned_name : str
            调优使用的名称
            
        返回:
        ----
        str : 对应的原始名称
        """
        reverse_map = {v: k for k, v in cls.MODEL_MAPPING.items()}
        return reverse_map.get(tuned_name, tuned_name)
    
    @classmethod
    def get_tuning_name(cls, original_name):
        """
        获取原始名称对应的调优名称
        
        参数:
        ----
        original_name : str
            原始模型名称
            
        返回:
        ----
        str : 对应的调优名称
        """
        return cls.MODEL_MAPPING.get(original_name, original_name)
    
    @classmethod
    def is_valid_model_name(cls, name):
        """
        检查是否为有效的模型名称
        
        参数:
        ----
        name : str
            模型名称
            
        返回:
        ----
        bool : 是否为有效名称
        """
        return name in cls.MODEL_MAPPING or name in cls.MODEL_MAPPING.values() 