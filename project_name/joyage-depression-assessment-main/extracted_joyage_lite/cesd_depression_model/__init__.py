"""
CESD Depression Prediction Model Package
"""

# 简化导入，避免循环导入问题
__version__ = '1.0.0'
__author__ = "CESD Research Team"

# 延迟导入，只在需要时导入
def get_pipeline():
    """获取主流水线类"""
    from cesd_depression_model.core.main_pipeline import CESDPredictionPipeline
    return CESDPredictionPipeline

def get_data_processor():
    """获取数据处理器类"""
    from cesd_depression_model.preprocessing.data_processor import DataProcessor
    return DataProcessor

def get_model_builder():
    """获取模型构建器类"""
    from cesd_depression_model.models.model_builder import ModelBuilder
    return ModelBuilder

def get_evaluator():
    """获取模型评估器类"""
    from cesd_depression_model.evaluation.model_evaluator import ModelEvaluator
    return ModelEvaluator

__all__ = [
    'get_pipeline',
    'get_data_processor', 
    'get_model_builder',
    'get_evaluator'
] 