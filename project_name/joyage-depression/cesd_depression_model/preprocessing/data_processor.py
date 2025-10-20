"""
Data preprocessing module for CESD Depression Prediction Model
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, RobustScaler
from imblearn.over_sampling import SMOTE
from ..config import CATEGORICAL_VARS, NUMERICAL_VARS, SMOTE_PARAMS, EXCLUDED_VARS

class DataProcessor:
    """数据预处理类 - 完全按照原始版本逻辑"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.label_encoders = {}
        self.scaler = None
        self.imputation_values = {}
        
        # 模型特定的编码器
        self.svm_label_encoders = {}
        self.svm_onehot_encoder = None
        self.svm_nominal_label_encoders = {}
        self.svm_scaler = None
        
        self.tree_label_encoders = {}
        
        self.ensemble_label_encoders = {}
        self.ensemble_scaler = None
        
        # 🆕 预定义关键分类变量的所有可能类别
        self.adlfive_all_categories = [
            '0.Fully Independent',
            '1.Mild Dependence', 
            '2.Moderate Dependence',
            '3.Significant Dependence',
            '4.Severe Dependence',
            '5.Total Dependence'
        ]
        
        # 其他分类变量的预定义类别
        self.categorical_all_categories = {
            'child': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # CHARLS有10，KLOSA没有
            'ragender': ['1.man', '2.woman'],
            'work': ['0.Not working for pay', '1.Working for pay'],
            'diabe': ['0.No', '1.Yes'],
            'stroke': ['0.No', '1.Yes'],
            'livere': ['0.No', '1.Yes']
        }
        
    def load_data(self, file_path, dataset_name="Dataset"):
        """加载数据"""
        print(f"\n{'='*60}")
        print(f"加载{dataset_name}数据")
        print(f"{'='*60}")
        
        try:
            data = pd.read_csv(file_path)
            print(f"✓ 数据加载成功: {data.shape}")
            
            # 检查目标变量
            if 'depressed' not in data.columns:
                print("✗ 未找到目标变量 'depressed'")
                return None
                
            # 处理目标变量
            if data['depressed'].dtype == 'object':
                data['depressed'] = data['depressed'].map({'0.No': 0, '1.Yes': 1})
                print("  ✓ 已将depressed从字符串格式(0.No/1.Yes)转换为数值格式(0/1)")
            
            # 🔧 修复：确保depressed变量为整数类型，处理NaN值
            if data['depressed'].dtype == 'float64':
                # 删除depressed为NaN的样本
                missing_count = data['depressed'].isnull().sum()
                if missing_count > 0:
                    print(f"  ⚠️ 发现 {missing_count} 个depressed缺失值，删除这些样本")
                    data = data.dropna(subset=['depressed'])
                    print(f"  删除后样本数: {len(data)}")
                
                # 转换为整数类型
                data['depressed'] = data['depressed'].astype(int)
                print("  ✓ 已将depressed转换为整数类型")
            
            # 🔍 数据完整性检查（仅对训练数据）
            if dataset_name == "CHARLS":
                self._check_data_completeness(data)
            
            # 显示基本统计信息
            self._display_basic_stats(data)
            
            return data
            
        except Exception as e:
            print(f"✗ 数据加载失败: {e}")
            return None
    
    def _check_data_completeness(self, data):
        """检查训练数据的完整性，确保包含所有可能的类别"""
        print(f"\n🔍 数据完整性检查:")
        print("-" * 40)
        
        # 定义关键分类变量及其预期类别
        expected_categories = {
            'adlfive': ['0.Fully Independent', '1.Mild Dependence', '2.Moderate Dependence', 
                       '3.Significant Dependence', '4.Severe Dependence', '5.Total Dependence'],  # 修正：使用实际字符串格式
            'child': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # CHARLS有10，KLOSA没有
            'ragender': ['1.man', '2.woman'],
            'work': ['0.Not working for pay', '1.Working for pay'],
            'diabe': ['0.No', '1.Yes'],
            'stroke': ['0.No', '1.Yes'],
            'livere': ['0.No', '1.Yes']
        }
        
        missing_categories = {}
        
        for var, expected in expected_categories.items():
            if var in data.columns:
                actual_values = set(data[var].dropna().unique())
                expected_set = set(expected)
                
                # 检查是否有缺失的类别
                missing = expected_set - actual_values
                if missing:
                    missing_categories[var] = missing
                    print(f"  ⚠️  {var}: 缺少类别 {sorted(missing)}")
                else:
                    print(f"  ✅  {var}: 类别完整")
        
        if missing_categories:
            print(f"\n⚠️  警告：以下变量缺少某些类别，可能影响外部验证:")
            for var, missing in missing_categories.items():
                print(f"    {var}: 缺少 {sorted(missing)}")
            print(f"\n💡 建议：")
            print(f"    1. 检查数据采样是否合理")
            print(f"    2. 考虑使用分层采样确保类别完整性")
            print(f"    3. 或者使用扩展映射策略处理新类别")
        else:
            print(f"\n✅ 所有关键变量类别完整，适合严格映射策略")
            
    def preprocess_data_before_split(self, data):
        """数据分割前预处理 - 完全按照原始版本"""
        print(f"\n{'='*60}")
        print("数据预处理 - 新的缺失值处理顺序")
        print(f"{'='*60}")
        
        if data is None:
            raise ValueError("输入数据为空")
            
        processed_data = data.copy()
        
        # 1. 删除ID变量
        vars_found = []
        for var in EXCLUDED_VARS:
            if var in processed_data.columns:
                processed_data = processed_data.drop(columns=[var])
                vars_found.append(var)
        
        if vars_found:
            print(f"✅ 已删除ID变量: {', '.join(vars_found)}")
        
        # 2. 处理目标变量缺失值
        depressed_missing = processed_data['depressed'].isnull().sum()
        if depressed_missing > 0:
            print(f"⚠️ 目标变量有 {depressed_missing} 个缺失值，删除这些样本")
            valid_indices = ~processed_data['depressed'].isnull()
            processed_data = processed_data[valid_indices]
            print(f"删除后样本数: {len(processed_data)}")
        
        # ✅ 优化：删除冗余的depression标签创建
        # processed_data['depression'] = processed_data['depressed'].astype(int)
        
        return processed_data
        
    def prepare_features_by_model_type(self, data, model_type='tree', is_training=True):
        """根据模型类型准备特征数据 - 完全按照原始版本"""
        print(f"\n{'-'*50}")
        print(f"根据模型类型准备特征数据: {model_type.upper()}")
        print(f"{'-'*50}")
        
        try:
            # 1. 分离特征和目标变量 (优化：简化逻辑)
            target_col = 'depressed'
            
            if target_col not in data.columns:
                raise ValueError(f"目标变量 '{target_col}' 不存在")
            
            features = data.drop(columns=[target_col])
            target = data[target_col]
            
            # 确保目标变量为数值类型（在需要时转换）
            if target.dtype == 'object':
                target = target.astype(int)
            
            print(f"原始特征数量: {features.shape[1]}")
            print(f"目标变量分布:\n{target.value_counts()}")
            
            # 2. 处理缺失值
            features = self.impute_features(features, is_training)
            
            # 3. 识别特征类型
            print(f"🔍 开始特征类型识别...")
            print(f"  配置中的分类变量数量: {len(CATEGORICAL_VARS)}")
            print(f"  配置中的数值变量数量: {len(NUMERICAL_VARS)}")
            
            numeric_columns = []
            binary_columns = []
            ordinal_columns = []
            nominal_columns = []
            
            for column in features.columns:
                try:
                    # 🔍 特殊调试：检查hhres变量的处理
                    if column == 'hhres':
                        if column in CATEGORICAL_VARS:
                            print(f"  ❌ 错误: hhres被识别为分类变量!")
                        else:
                            print(f"  ✅ 正确: hhres被识别为数值变量")
                    
                    if column in CATEGORICAL_VARS:
                        unique_vals = features[column].dropna().unique()
                        if len(unique_vals) == 2:
                            binary_columns.append(column)
                        elif self._is_ordinal_categorical(column, unique_vals):
                            ordinal_columns.append(column)
                        else:
                            nominal_columns.append(column)
                    else:
                        numeric_columns.append(column)
                except:
                    if column in CATEGORICAL_VARS:
                        nominal_columns.append(column)
                    else:
                        numeric_columns.append(column)
            
            print(f"特征类型统计:")
            print(f"  数值变量: {len(numeric_columns)} 个")
            print(f"  二分类变量: {len(binary_columns)} 个") 
            print(f"  有序分类变量: {len(ordinal_columns)} 个")
            print(f"  无序分类变量: {len(nominal_columns)} 个")
            
            # 4. 根据模型类型处理特征
            if model_type.lower() == 'svm':
                print(f"\n🔧 SVM模型特征处理:")
                print("  - 连续型变量: 标准化")
                print("  - 二分类变量: 标签编码 (0/1)")
                print("  - 有序分类变量: 标签编码 (保持顺序)")
                print("  - 无序分类变量: 独热编码")
                
                features_processed = self._process_features_for_svm(
                    features, numeric_columns, binary_columns, 
                    ordinal_columns, nominal_columns, is_training
                )
                
            elif model_type.lower() in ['tree', 'forest', 'xgboost', 'lightgbm', 'catboost']:
                print(f"\n🌳 树模型特征处理:")
                print("  - 连续型变量: 保持原样")
                print("  - 所有分类变量: 标签编码")
                
                features_processed = self._process_features_for_tree(
                    features, numeric_columns, binary_columns, 
                    ordinal_columns, nominal_columns, is_training
                )
                
            elif model_type.lower() in ['linear', 'logistic', 'ridge', 'lasso']:
                print(f"\n📈 线性模型特征处理:")
                print("  - 连续型变量: 标准化")
                print("  - 二分类变量: 标签编码")
                print("  - 有序分类变量: 标签编码")
                print("  - 无序分类变量: 独热编码")
                
                features_processed = self._process_features_for_linear(
                    features, numeric_columns, binary_columns, 
                    ordinal_columns, nominal_columns, is_training
                )
                
            elif model_type.lower() == 'ensemble':
                print(f"\n🎯 集成模型特征处理:")
                print("  - 使用混合策略，兼顾不同基础模型需求")
                
                features_processed = self._process_features_for_ensemble(
                    features, numeric_columns, binary_columns, 
                    ordinal_columns, nominal_columns, is_training
                )
                
            else:
                print(f"\n⚠️ 未知模型类型 {model_type}，使用默认处理")
                features_processed = self._process_features_for_tree(
                    features, numeric_columns, binary_columns, 
                    ordinal_columns, nominal_columns, is_training
                )
            
            # 5. 最终检查
            final_null_count = features_processed.isnull().sum().sum()
            if final_null_count > 0:
                print(f"🚨 警告: {model_type}模型特征处理完成后仍有 {final_null_count} 个缺失值!")
                # 紧急修复
                for col in features_processed.columns:
                    if features_processed[col].isnull().any():
                        if pd.api.types.is_numeric_dtype(features_processed[col]):
                            features_processed[col].fillna(0, inplace=True)
                        else:
                            features_processed[col].fillna('Unknown', inplace=True)
            
            print(f"\n✅ {model_type.upper()} 模型特征处理完成")
            print(f"   最终特征数: {features_processed.shape[1]}")
            print(f"   样本数: {features_processed.shape[0]}")
            
            return features_processed, target
            
        except Exception as e:
            print(f"❌ {model_type} 特征处理失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def _is_ordinal_categorical(self, column, unique_vals):
        """判断是否为有序分类变量"""
        if all(isinstance(val, (int, float)) for val in unique_vals):
            sorted_vals = sorted(unique_vals)
            if len(sorted_vals) <= 10 and sorted_vals == list(range(int(min(sorted_vals)), int(max(sorted_vals)) + 1)):
                return True
        return False
    
    def _process_features_for_svm(self, features, numeric_cols, binary_cols, ordinal_cols, nominal_cols, is_training):
        """SVM模型的特征处理 - 使用直接编码策略"""
        features_processed = features.copy()
        
        # SVM模型：二元变量保持原样，序数变量直接编码，名义变量独热编码
        # 🚀 优化：使用统一的编码器处理二元和序数变量
        from ..utils.categorical_encoder import UnifiedCategoricalEncoder
        
        # 批量处理二元变量
        features_processed = UnifiedCategoricalEncoder.batch_encode_columns(
            features_processed, binary_cols, self.svm_label_encoders, is_training, 
            "二元变量直接编码"
        )
        
        # 批量处理序数变量  
        features_processed = UnifiedCategoricalEncoder.batch_encode_columns(
            features_processed, ordinal_cols, self.svm_label_encoders, is_training,
            "序数变量直接编码"
        )
        
        if nominal_cols:
            print(f"\n名义变量独热编码: {len(nominal_cols)} 个")
            # 先对名义变量进行直接编码
            for col in nominal_cols:
                features_processed[col] = features_processed[col].fillna('missing').astype(str)
                
                if is_training:
                    if col not in self.svm_nominal_label_encoders:
                        self.svm_nominal_label_encoders[col] = LabelEncoder()
                    unique_values = sorted(features_processed[col].unique())
                    self.svm_nominal_label_encoders[col].fit(unique_values)
                    features_processed[col] = self.svm_nominal_label_encoders[col].transform(features_processed[col])
                else:
                    if col in self.svm_nominal_label_encoders:
                        encoder = self.svm_nominal_label_encoders[col]
                        known_categories = set(encoder.classes_)
                        current_categories = set(features_processed[col].unique())
                        new_categories = current_categories - known_categories
                        
                        if new_categories:
                            raise ValueError(f"❌ 严格映射错误：{col} 发现训练数据中不存在的新类别: {new_categories}")
                        else:
                            features_processed[col] = encoder.transform(features_processed[col])
                    else:
                        # 临时编码器
                        temp_encoder = LabelEncoder()
                        unique_vals = sorted(features_processed[col].unique())
                        temp_encoder.fit(unique_vals)
                        features_processed[col] = temp_encoder.transform(features_processed[col])
            
            # 🆕 改进的独热编码：使用预定义类别
            # 注意：OneHotEncoder需要原始字符串数据，不是LabelEncoder后的数字
            nominal_encoded_original = features[nominal_cols].fillna('missing')
            
            if is_training:
                # 训练时：创建使用预定义类别的编码器
                categories_list = []
                for col in nominal_cols:
                    if col == 'adlfive':
                        # 特殊处理adlfive，使用预定义类别
                        categories_list.append(self.adlfive_all_categories)
                    elif col in self.categorical_all_categories:
                        # 其他预定义分类变量
                        categories_list.append(self.categorical_all_categories[col])
                    else:
                        # 对于未预定义的变量，使用数据中的实际类别
                        categories_list.append(sorted(nominal_encoded_original[col].unique()))
                
                self.svm_onehot_encoder = OneHotEncoder(
                    categories=categories_list,
                    sparse_output=False, 
                    handle_unknown='ignore'
                )
                nominal_onehot = self.svm_onehot_encoder.fit_transform(nominal_encoded_original)
                
                # 保存训练时的特征名称
                self.svm_onehot_feature_names = []
                for i, col in enumerate(nominal_cols):
                    categories = self.svm_onehot_encoder.categories_[i]
                    for cat in categories:
                        self.svm_onehot_feature_names.append(f"{col}_{cat}")
                
                print(f"✅ 独热编码完成，生成 {len(self.svm_onehot_feature_names)} 个特征")
                print(f"✅ adlfive特征: {[f for f in self.svm_onehot_feature_names if 'adlfive_' in f]}")
                
            else:
                # 预测时：使用训练好的编码器
                if self.svm_onehot_encoder is not None:
                    nominal_onehot = self.svm_onehot_encoder.transform(nominal_encoded_original)
                else:
                    # 如果没有训练好的独热编码器，跳过独热编码
                    nominal_onehot = nominal_encoded_original.values
            
            # 创建独热编码的列名
            if is_training and self.svm_onehot_encoder is not None:
                feature_names = self.svm_onehot_feature_names
            elif hasattr(self, 'svm_onehot_feature_names') and self.svm_onehot_feature_names:
                # 使用训练时保存的特征名称
                feature_names = self.svm_onehot_feature_names
            else:
                feature_names = [f"onehot_{i}" for i in range(nominal_onehot.shape[1])]
            
            # 删除原始名义变量，添加独热编码变量
            features_processed = features_processed.drop(columns=nominal_cols)
            nominal_df = pd.DataFrame(nominal_onehot, columns=feature_names, index=features_processed.index)
            features_processed = pd.concat([features_processed, nominal_df], axis=1)
        
        # 数值变量标准化
        if numeric_cols:
            print(f"\n数值变量标准化: {len(numeric_cols)} 个")
            if is_training:
                self.svm_scaler = StandardScaler()
                features_processed[numeric_cols] = self.svm_scaler.fit_transform(features_processed[numeric_cols])
            else:
                if self.svm_scaler is not None:
                    features_processed[numeric_cols] = self.svm_scaler.transform(features_processed[numeric_cols])
        
        return features_processed
    
    def _process_features_for_tree(self, features, numeric_cols, binary_cols, ordinal_cols, nominal_cols, is_training):
        """树模型的特征处理 - 使用直接编码策略"""
        features_processed = features.copy()
        
        # 树模型：所有分类变量都用直接编码
        categorical_cols = binary_cols + ordinal_cols + nominal_cols
        if categorical_cols:
            print(f"\n直接编码处理 {len(categorical_cols)} 个分类变量:")
            for col in categorical_cols:
                # 先填充缺失值
                features_processed[col] = features_processed[col].fillna('missing').astype(str)
                
                if is_training:
                    # 训练阶段：创建编码器并保存
                    if col not in self.tree_label_encoders:
                        self.tree_label_encoders[col] = LabelEncoder()
                    
                    # 获取所有唯一值并排序，确保编码一致性
                    unique_values = sorted(features_processed[col].unique())
                    self.tree_label_encoders[col].fit(unique_values)
                    
                    # 编码数据
                    features_processed[col] = self.tree_label_encoders[col].transform(features_processed[col])
                    
                    print(f"  📊 训练编码器: {col}")
                    print(f"    类别数量: {len(unique_values)}")
                    print(f"    编码映射: {dict(zip(unique_values, range(len(unique_values))))}")
                    
                    # 🔧 修复：确保编码器同时保存到主编码器字典
                    if col not in self.label_encoders:
                        self.label_encoders[col] = self.tree_label_encoders[col]
                    
                else:
                    # 测试阶段：使用训练好的编码器
                    if col in self.tree_label_encoders:
                        encoder = self.tree_label_encoders[col]
                        known_categories = set(encoder.classes_)
                        
                        # 检查是否有新类别
                        current_categories = set(features_processed[col].unique())
                        new_categories = current_categories - known_categories
                        
                        if new_categories:
                            raise ValueError(f"❌ 严格映射错误：{col} 发现训练数据中不存在的新类别: {new_categories}")
                        else:
                            # 没有新类别，使用原始编码器
                            features_processed[col] = encoder.transform(features_processed[col])
                            print(f"  ✅ 使用原始编码器: {col}")
                            
                    else:
                        # 🔧 修复：尝试从主编码器字典获取编码器
                        if col in self.label_encoders:
                            encoder = self.label_encoders[col]
                            print(f"  ✅ 从主编码器获取: {col}")
                            features_processed[col] = encoder.transform(features_processed[col])
                        else:
                            print(f"  ! 严重警告: {col} 缺少训练好的标签编码器")
                            print(f"    这会导致编码不一致！建议检查编码器保存/加载流程")
                            
                            # 🚨 紧急处理：尝试使用数值编码
                            try:
                                # 先尝试转换为数值（可能数据已经是数值编码）
                                features_processed[col] = pd.to_numeric(features_processed[col], errors='coerce')
                                # 填充转换失败的值
                                if features_processed[col].isna().any():
                                    features_processed[col] = features_processed[col].fillna(0)
                                print(f"    已转换为数值编码")
                            except:
                                # 最后手段：临时标签编码（会有编码不一致问题）
                                temp_encoder = LabelEncoder()
                                unique_vals = features_processed[col].astype(str).unique()
                                temp_encoder.fit(unique_vals)
                                features_processed[col] = temp_encoder.transform(features_processed[col].astype(str))
                                print(f"    使用临时编码器（可能不一致）")
        
        # 数值变量保持原样（树模型不需要标准化）
        if numeric_cols:
            print(f"\n数值变量保持原样: {len(numeric_cols)} 个")
        
        return features_processed
    
    def _process_features_for_linear(self, features, numeric_cols, binary_cols, ordinal_cols, nominal_cols, is_training):
        """线性模型的特征处理（类似SVM）"""
        return self._process_features_for_svm(features, numeric_cols, binary_cols, ordinal_cols, nominal_cols, is_training)
    
    def _process_features_for_ensemble(self, features, numeric_cols, binary_cols, ordinal_cols, nominal_cols, is_training):
        """集成模型的特征处理 - 使用直接编码策略"""
        features_processed = features.copy()
        
        # 集成模型使用直接编码（适合树模型）+ 部分标准化（适合线性模型）
        categorical_cols = binary_cols + ordinal_cols + nominal_cols
        if categorical_cols:
            print(f"\n直接编码处理 {len(categorical_cols)} 个分类变量:")
            for col in categorical_cols:
                # 先填充缺失值
                features_processed[col] = features_processed[col].fillna('missing').astype(str)
                
                if is_training:
                    # 训练阶段：创建编码器并保存
                    if col not in self.ensemble_label_encoders:
                        self.ensemble_label_encoders[col] = LabelEncoder()
                    
                    # 获取所有唯一值并排序，确保编码一致性
                    unique_values = sorted(features_processed[col].unique())
                    self.ensemble_label_encoders[col].fit(unique_values)
                    
                    # 编码数据
                    features_processed[col] = self.ensemble_label_encoders[col].transform(features_processed[col])
                    
                    print(f"  📊 训练编码器: {col}")
                    print(f"    类别数量: {len(unique_values)}")
                    print(f"    编码映射: {dict(zip(unique_values, range(len(unique_values))))}")
                    
                else:
                    # 测试阶段：使用训练好的编码器
                    if col in self.ensemble_label_encoders:
                        encoder = self.ensemble_label_encoders[col]
                        known_categories = set(encoder.classes_)
                        
                        # 检查是否有新类别
                        current_categories = set(features_processed[col].unique())
                        new_categories = current_categories - known_categories
                        
                        if new_categories:
                            raise ValueError(f"❌ 严格映射错误：{col} 发现训练数据中不存在的新类别: {new_categories}")
                        else:
                            # 没有新类别，使用原始编码器
                            features_processed[col] = encoder.transform(features_processed[col])
                            print(f"  ✅ 使用原始编码器: {col}")
                            
                    else:
                        raise ValueError(f"❌ 错误：{col} 缺少训练好的编码器，无法进行严格映射")
        
        # 数值变量轻度标准化（RobustScaler，对异常值不敏感）
        if numeric_cols:
            print(f"\n轻度标准化 {len(numeric_cols)} 个数值变量:")
            if is_training:
                self.ensemble_scaler = RobustScaler()
                features_processed[numeric_cols] = self.ensemble_scaler.fit_transform(features_processed[numeric_cols])
                print(f"  - 已使用RobustScaler标准化")
            else:
                if hasattr(self, 'ensemble_scaler') and self.ensemble_scaler is not None:
                    features_processed[numeric_cols] = self.ensemble_scaler.transform(features_processed[numeric_cols])
        
        return features_processed
    
    def impute_features(self, features, is_training):
        """修复版缺失值填充 - 避免数据泄露和Pandas警告"""
        print("\n处理缺失值...")
        
        # 🔧 修复：创建副本避免SettingWithCopyWarning
        features = features.copy()
        
        null_counts = features.isnull().sum()
        columns_with_nulls = null_counts[null_counts > 0]
        
        if not columns_with_nulls.empty:
            print("发现缺失值:")
            for col, count in columns_with_nulls.items():
                print(f"  - {col}: {count} 个缺失值 ({count/len(features)*100:.2f}%)")
                
                # 🔧 特殊处理：comparable_hexp直接使用中位数
                if col == 'comparable_hexp':
                    if is_training:
                        fill_value = features[col].median()
                        if pd.isna(fill_value):
                            fill_value = 0.0
                        self.imputation_values[col] = fill_value
                    else:
                        fill_value = self.imputation_values.get(col, 0.0)
                    
                    features[col] = features[col].fillna(fill_value)
                    print(f"    已填充: {fill_value}")
                    continue
                
                if col in NUMERICAL_VARS:
                    if is_training:
                        # 🔧 修复：确保计算填充值时数据有效
                        valid_values = features[col].dropna()
                        if len(valid_values) > 0:
                            fill_value = valid_values.median()
                        else:
                            fill_value = 0.0  # 默认值
                        self.imputation_values[col] = fill_value
                    else:
                        # 🔧 修复：避免数据泄露 - 仅使用训练时保存的填充值
                        fill_value = self.imputation_values.get(col, 0.0)
                    
                    # 🔧 修复：避免inplace警告
                    features[col] = features[col].fillna(fill_value)
                else:
                    if is_training:
                        # 🔧 修复：安全的众数计算
                        valid_values = features[col].dropna()
                        if len(valid_values) > 0:
                            mode_values = valid_values.mode()
                            fill_value = mode_values[0] if len(mode_values) > 0 else '0.Default'
                        else:
                            fill_value = '0.Default'
                        self.imputation_values[col] = fill_value
                    else:
                        # 🔧 修复：避免数据泄露 - 仅使用训练时保存的填充值
                        fill_value = self.imputation_values.get(col, '0.Default')
                    
                    # 🔧 修复：避免inplace警告
                    features[col] = features[col].fillna(fill_value)
                    
                print(f"    已填充: {fill_value}")
        else:
            print("✓ 没有发现缺失值")
            
        return features
    
    def apply_smote(self, X, y):
        """应用SMOTE进行数据平衡"""
        print("\n应用SMOTE进行数据平衡...")
        
        try:
            smote = SMOTE(**SMOTE_PARAMS)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            # 显示平衡后的分布
            unique_new, counts_new = np.unique(y_resampled, return_counts=True)
            print("\nSMOTE处理后数据分布:")
            for label, count in zip(unique_new, counts_new):
                print(f"  类别 {label}: {count} 样本 ({count/len(y_resampled)*100:.1f}%)")
                
            return X_resampled, y_resampled
            
        except Exception as e:
            print(f"✗ SMOTE处理失败: {str(e)}")
            return X, y
    
    def _display_basic_stats(self, data):
        """显示数据基本统计信息"""
        print(f"\n数据基本统计信息:")
        print(f"样本数量: {len(data)}")
        print(f"特征数量: {data.shape[1]}")
        
        if 'depressed' in data.columns:
            print(f"\n目标变量分布:")
            print(data['depressed'].value_counts(normalize=True))
        
        # 显示数值变量统计
        numerical_columns = [col for col in NUMERICAL_VARS if col in data.columns]
        if numerical_columns:
            print(f"\n数值变量统计:")
            print(data[numerical_columns].describe())

    # 保持向后兼容的旧方法
    def preprocess_data(self, data, is_training=True):
        """预处理数据（向后兼容）"""
        if data is None:
            return None
            
        # 检查和创建缺失变量
        if 'coresd' not in data.columns:
            data['coresd'] = '0.Default'
        if 'ftrhlp' not in data.columns:
            data['ftrhlp'] = '0.Default'
            
        # 排除ID变量
        for var in EXCLUDED_VARS:
            if var in data.columns:
                data = data.drop(columns=[var])
        
        # 处理缺失值
        data = self.impute_features(data, is_training)
        
        # 编码分类变量
        for col in CATEGORICAL_VARS:
            if col in data.columns:
                if is_training:
                    self.label_encoders[col] = LabelEncoder()
                    data[col] = self.label_encoders[col].fit_transform(data[col].astype(str))
                else:
                    if col in self.label_encoders:
                        data[col] = self._direct_encode_with_consistency(data[col], col, is_training)
        
        # 标准化数值变量
        numerical_columns = [col for col in NUMERICAL_VARS if col in data.columns]
        if numerical_columns:
            if is_training:
                self.scaler = StandardScaler()
                data[numerical_columns] = self.scaler.fit_transform(data[numerical_columns])
            else:
                if self.scaler is not None:
                    data[numerical_columns] = self.scaler.transform(data[numerical_columns])
                    
        return data 

    def _direct_encode_with_consistency(self, data, column_name, is_training=True):
        """对分类变量进行严格编码 - 只允许训练数据中存在的类别"""
        if is_training:
            # 训练阶段：创建编码器
            print(f"    📊 训练编码器: {column_name}")
            
            # 处理缺失值
            data_clean = data.fillna('missing')
            
            # 🔧 修复：对于adlfive变量，使用特殊的数据类型统一处理
            if column_name == 'adlfive':
                # adlfive实际是字符串类型，直接使用，不需要数值转换
                data_str = data_clean.astype(str)
            else:
                data_str = data_clean.astype(str)
            
            # 创建编码器
            unique_values = sorted(data_str.unique())
            encoder = LabelEncoder()
            encoder.fit(unique_values)
            
            # 编码数据
            encoded_data = encoder.transform(data_str)
            
            # 保存编码器
            self.label_encoders[column_name] = encoder
            
            # 显示编码映射
            mapping = dict(zip(encoder.classes_, range(len(encoder.classes_))))
            print(f"      类别数量: {len(unique_values)}")
            print(f"      编码映射: {mapping}")
            
            return pd.Series(encoded_data, index=data.index, name=column_name)
            
        else:
            # 测试阶段：使用训练好的编码器，严格映射
            print(f"    ✅ 使用严格编码器: {column_name}")
            
            if column_name not in self.label_encoders:
                raise ValueError(f"❌ 错误：{column_name} 缺少训练好的编码器，无法进行严格映射")
            
            encoder = self.label_encoders[column_name]
            
            # 处理缺失值
            data_clean = data.fillna('missing')
            
            # 🔧 修复：对于adlfive变量，使用特殊的数据类型统一处理
            if column_name == 'adlfive':
                # adlfive实际是字符串类型，直接使用，不需要数值转换
                data_str = data_clean.astype(str)
            else:
                data_str = data_clean.astype(str)
            
            # 检查是否有新类别
            known_categories = set(encoder.classes_)
            current_categories = set(data_str.unique())
            new_categories = current_categories - known_categories
            
            if new_categories:
                raise ValueError(f"❌ 严格映射错误：{column_name} 发现训练数据中不存在的新类别: {new_categories}")
            
            # 严格映射：只允许已知类别
            try:
                encoded_data = encoder.transform(data_str)
                print(f"      ✅ 严格映射成功，无新类别")
                return pd.Series(encoded_data, index=data.index, name=column_name)
            except ValueError as e:
                raise ValueError(f"❌ 严格映射失败：{column_name} 编码错误 - {e}")
    
    def _process_features_direct_encoding(self, features, is_training=True):
        """
        使用直接编码处理所有分类变量，确保一致性
        """
        from ..config import CATEGORICAL_VARS
        
        features_processed = features.copy()
        
        # 识别分类变量
        categorical_cols = [col for col in features.columns if col in CATEGORICAL_VARS]
        
        if categorical_cols:
            print(f"\n🔤 直接编码处理 {len(categorical_cols)} 个分类变量:")
            
            for col in categorical_cols:
                features_processed[col] = self._direct_encode_with_consistency(
                    features_processed[col], col, is_training
                )
        
        return features_processed 
    
    def validate_preprocessing_consistency(self, train_data, test_data):
        """验证训练和测试数据预处理的一致性"""
        print("🔍 验证数据预处理一致性...")
        
        # 检查分类变量
        categorical_vars = ['adlfive', 'ragender', 'work', 'diabe', 'stroke', 'livere', 'child']
        
        for col in categorical_vars:
            if col in train_data.columns and col in test_data.columns:
                train_cats = set(train_data[col].dropna().unique())
                test_cats = set(test_data[col].dropna().unique())
                
                missing_in_test = train_cats - test_cats
                new_in_test = test_cats - train_cats
                
                if missing_in_test:
                    print(f"  ⚠️ {col}: 测试数据缺少类别 {missing_in_test}")
                if new_in_test:
                    print(f"  ⚠️ {col}: 测试数据有新类别 {new_in_test}")
                
                if not missing_in_test and not new_in_test:
                    print(f"  ✅ {col}: 类别完全一致")
        
        print("✅ 数据预处理一致性检查完成")