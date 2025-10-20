"""
统一的分类变量编码器 - 消除重复编码逻辑
"""

from sklearn.preprocessing import LabelEncoder
import pandas as pd

class UnifiedCategoricalEncoder:
    """统一的分类变量编码器，消除数据处理中的重复逻辑"""
    
    @staticmethod
    def encode_categorical_column(features_processed, col, encoder_dict, is_training):
        """
        统一的分类变量编码方法
        
        参数:
        ----
        features_processed : pd.DataFrame
            要处理的特征数据
        col : str
            列名
        encoder_dict : dict
            编码器字典
        is_training : bool
            是否为训练阶段
            
        返回:
        ----
        pd.DataFrame : 处理后的特征数据
        """
        # 预处理：填充缺失值并转换为字符串
        features_processed[col] = features_processed[col].fillna('missing').astype(str)
        
        if is_training:
            # 训练阶段：创建并拟合编码器
            if col not in encoder_dict:
                encoder_dict[col] = LabelEncoder()
            
            unique_values = sorted(features_processed[col].unique())
            encoder_dict[col].fit(unique_values)
            features_processed[col] = encoder_dict[col].transform(features_processed[col])
            
            print(f"    ✓ 训练编码器 {col}: {len(unique_values)} 个类别")
            
        else:
            # 验证阶段：使用已有编码器并验证类别
            if col in encoder_dict:
                encoder = encoder_dict[col]
                UnifiedCategoricalEncoder._validate_categories(col, features_processed[col], encoder)
                features_processed[col] = encoder.transform(features_processed[col])
                print(f"    ✓ 应用编码器 {col}")
            else:
                raise ValueError(f"❌ 缺失编码器: {col}")
        
        return features_processed
    
    @staticmethod
    def _validate_categories(col, data, encoder):
        """
        验证类别是否在训练集中存在
        
        参数:
        ----
        col : str
            列名
        data : pd.Series
            数据列
        encoder : LabelEncoder
            已训练的编码器
        """
        known_categories = set(encoder.classes_)
        current_categories = set(data.unique())
        new_categories = current_categories - known_categories
        
        if new_categories:
            raise ValueError(f"❌ 严格映射错误：{col} 发现训练数据中不存在的新类别: {new_categories}")
    
    @staticmethod
    def batch_encode_columns(features_processed, columns, encoder_dict, is_training, description=""):
        """
        批量编码多个分类列
        
        参数:
        ----
        features_processed : pd.DataFrame
            要处理的特征数据
        columns : list
            要编码的列名列表
        encoder_dict : dict
            编码器字典
        is_training : bool
            是否为训练阶段
        description : str
            描述信息
            
        返回:
        ----
        pd.DataFrame : 处理后的特征数据
        """
        if columns:
            print(f"\n{description}: {len(columns)} 个")
            for col in columns:
                features_processed = UnifiedCategoricalEncoder.encode_categorical_column(
                    features_processed, col, encoder_dict, is_training
                )
        
        return features_processed
    
    @staticmethod
    def sync_encoders_efficiently(data_processor):
        """
        高效地同步编码器到所有存储位置
        
        参数:
        ----
        data_processor : DataProcessor
            数据处理器实例
        """
        print("🔄 高效同步编码器...")
        
        # 定义所有编码器属性
        encoder_attrs = [
            'tree_label_encoders', 
            'svm_nominal_label_encoders', 
            'svm_label_encoders', 
            'ensemble_label_encoders'
        ]
        
        # 确保所有编码器字典存在
        for attr in encoder_attrs:
            if not hasattr(data_processor, attr):
                setattr(data_processor, attr, {})
        
        # 从主编码器同步到所有其他编码器
        if hasattr(data_processor, 'label_encoders') and data_processor.label_encoders:
            synced_count = 0
            for var, encoder in data_processor.label_encoders.items():
                for attr in encoder_attrs:
                    encoder_dict = getattr(data_processor, attr)
                    if var not in encoder_dict:
                        encoder_dict[var] = encoder
                        synced_count += 1
            
            print(f"✅ 编码器同步完成，同步了 {synced_count} 个编码器")
        else:
            print("⚠️ 主编码器为空，无法同步")
    
    @staticmethod
    def validate_encoder_consistency(data_processor, categorical_vars):
        """
        验证编码器一致性
        
        参数:
        ----
        data_processor : DataProcessor
            数据处理器实例
        categorical_vars : list
            分类变量列表
            
        返回:
        ----
        dict : 验证结果
        """
        print("\n🔍 验证编码器一致性...")
        
        encoder_attrs = ['tree_label_encoders', 'svm_nominal_label_encoders', 'svm_label_encoders']
        consistency_report = {}
        
        for var in categorical_vars:
            var_report = {'available_in': [], 'missing_in': []}
            
            for attr in encoder_attrs:
                if hasattr(data_processor, attr):
                    encoder_dict = getattr(data_processor, attr)
                    if var in encoder_dict:
                        var_report['available_in'].append(attr)
                    else:
                        var_report['missing_in'].append(attr)
            
            consistency_report[var] = var_report
            
            # 打印结果
            if var_report['missing_in']:
                print(f"  ⚠️ {var}: 缺失于 {var_report['missing_in']}")
            else:
                print(f"  ✅ {var}: 所有编码器都可用")
        
        return consistency_report 