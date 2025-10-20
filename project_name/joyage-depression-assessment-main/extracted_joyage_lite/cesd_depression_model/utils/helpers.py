"""
Utility functions for CESD Depression Prediction Model
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from ..config import FILE_PATHS

def create_directories():
    """创建必要的目录"""
    for dir_name in FILE_PATHS.values():
        os.makedirs(dir_name, exist_ok=True)
        
def save_model(model, filename, model_dir=None):
    """保存模型"""
    if model_dir is None:
        model_dir = FILE_PATHS['model_save_dir']
        
    os.makedirs(model_dir, exist_ok=True)
    
    filepath = os.path.join(model_dir, filename)
    joblib.dump(model, filepath)
    print(f"✓ 模型已保存: {filepath}")
    
def load_model(filename, model_dir=None):
    """加载模型"""
    if model_dir is None:
        model_dir = FILE_PATHS['model_save_dir']
        
    filepath = os.path.join(model_dir, filename)
    if os.path.exists(filepath):
        model = joblib.load(filepath)
        print(f"✓ 模型已加载: {filepath}")
        return model
    else:
        print(f"✗ 模型文件不存在: {filepath}")
        return None
        
def save_results(results, filename, results_dir=None):
    """保存结果"""
    if results_dir is None:
        results_dir = FILE_PATHS['results_dir']
        
    os.makedirs(results_dir, exist_ok=True)
    
    filepath = os.path.join(results_dir, filename)
    
    # 根据文件扩展名选择保存格式
    if filename.endswith('.json'):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    elif filename.endswith('.csv'):
        if isinstance(results, pd.DataFrame):
            results.to_csv(filepath, index=False, encoding='utf-8-sig')
        else:
            pd.DataFrame(results).to_csv(filepath, index=False, encoding='utf-8-sig')
    else:
        # 默认使用joblib保存
        joblib.dump(results, filepath)
        
    print(f"✓ 结果已保存: {filepath}")
    
def load_results(filename, results_dir=None):
    """加载结果"""
    if results_dir is None:
        results_dir = FILE_PATHS['results_dir']
        
    filepath = os.path.join(results_dir, filename)
    
    if not os.path.exists(filepath):
        print(f"✗ 结果文件不存在: {filepath}")
        return None
        
    # 根据文件扩展名选择加载格式
    if filename.endswith('.json'):
        with open(filepath, 'r', encoding='utf-8') as f:
            results = json.load(f)
    elif filename.endswith('.csv'):
        results = pd.read_csv(filepath)
    else:
        results = joblib.load(filepath)
        
    print(f"✓ 结果已加载: {filepath}")
    return results
    
def generate_timestamp():
    """生成时间戳"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
    
def validate_data(X, y):
    """验证数据"""
    # 检查形状匹配
    if len(X) != len(y):
        raise ValueError(f"X和y的样本数不匹配: {len(X)} vs {len(y)}")
        
    # 检查缺失值
    if hasattr(X, 'isnull'):
        null_count = X.isnull().sum().sum()
        if null_count > 0:
            print(f"⚠️ X中有 {null_count} 个缺失值")
            
    if hasattr(y, 'isnull'):
        null_count = y.isnull().sum()
        if null_count > 0:
            print(f"⚠️ y中有 {null_count} 个缺失值")
            
    # 检查目标变量分布
    unique_vals = np.unique(y)
    print(f"目标变量类别: {unique_vals}")
    
    if len(unique_vals) < 2:
        raise ValueError("目标变量只有一个类别，无法进行分类")
        
    return True
    
def calculate_class_weights(y):
    """计算类别权重"""
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    
    class_weight_dict = dict(zip(classes, weights))
    print(f"类别权重: {class_weight_dict}")
    
    return class_weight_dict
    
def format_results_table(results):
    """格式化结果表格"""
    formatted_data = []
    
    for model_name, metrics in results.items():
        row = {'Model': model_name}
        
        for metric_name, metric_data in metrics.items():
            if isinstance(metric_data, dict) and 'value' in metric_data:
                # 带置信区间的格式
                value = metric_data['value']
                ci_lower = metric_data['ci_lower']
                ci_upper = metric_data['ci_upper']
                row[metric_name] = f"{value:.3f} ({ci_lower:.3f}-{ci_upper:.3f})"
            else:
                # 简单数值格式
                row[metric_name] = f"{metric_data:.3f}"
                
        formatted_data.append(row)
        
    return pd.DataFrame(formatted_data)
    
def print_summary(results):
    """打印结果摘要"""
    print(f"\n{'='*60}")
    print("模型性能摘要")
    print(f"{'='*60}")
    
    # 找到最佳模型
    best_model = None
    best_auroc = -1
    
    for model_name, metrics in results.items():
        auroc = metrics.get('AUROC', {})
        if isinstance(auroc, dict):
            auroc_value = auroc.get('value', 0)
        else:
            auroc_value = auroc
            
        if auroc_value > best_auroc:
            best_auroc = auroc_value
            best_model = model_name
            
    print(f"🏆 最佳模型: {best_model} (AUROC: {best_auroc:.3f})")
    
    # 显示所有模型的主要指标
    print(f"\n所有模型性能:")
    print("-" * 60)
    print(f"{'Model':<15} {'AUROC':<8} {'AUPRC':<8} {'F1':<8} {'Accuracy':<8}")
    print("-" * 60)
    
    for model_name, metrics in results.items():
        auroc = metrics.get('AUROC', {})
        auprc = metrics.get('AUPRC', {})
        f1 = metrics.get('F1_Score', {})
        accuracy = metrics.get('Accuracy', {})
        
        # 提取数值
        if isinstance(auroc, dict):
            auroc_val = auroc.get('value', 0.0)
        else:
            auroc_val = float(auroc) if auroc is not None else 0.0
            
        if isinstance(auprc, dict):
            auprc_val = auprc.get('value', 0.0)
        else:
            auprc_val = float(auprc) if auprc is not None else 0.0
            
        if isinstance(f1, dict):
            f1_val = f1.get('value', 0.0)
        else:
            f1_val = float(f1) if f1 is not None else 0.0
            
        if isinstance(accuracy, dict):
            accuracy_val = accuracy.get('value', 0.0)
        else:
            accuracy_val = float(accuracy) if accuracy is not None else 0.0
        
        try:
            print(f"{model_name:<15} {auroc_val:<8.3f} {auprc_val:<8.3f} {f1_val:<8.3f} {accuracy_val:<8.3f}")
        except Exception as e:
            print(f"{model_name} - 格式化错误: {str(e)}")
            print(f"原始值: AUROC={auroc}, AUPRC={auprc}, F1={f1}, Accuracy={accuracy}")
        
def setup_logging(log_file='model_training.log'):
    """设置日志"""
    import logging
    
    logs_dir = FILE_PATHS['logs_dir']
    os.makedirs(logs_dir, exist_ok=True)
    
    log_path = os.path.join(logs_dir, log_file)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__) 