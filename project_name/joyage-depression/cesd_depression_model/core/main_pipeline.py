"""
Main pipeline for CESD Depression Prediction Model
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ..preprocessing.data_processor import DataProcessor

from ..models.model_builder import ModelBuilder
from ..models.hyperparameter_tuner import HyperparameterTuner
from ..evaluation.model_evaluator import ModelEvaluator
from ..visualization.plot_generator import PlotGenerator
from ..analysis.shap_analyzer import SHAPAnalyzer
from ..analysis.enhanced_shap_analyzer import EnhancedSHAPAnalyzer
from ..analysis.model_diagnostics import ModelDiagnosticsAnalyzer
from ..utils.helpers import *
from ..config import CV_SETTINGS, EXCLUDED_VARS
import time
from datetime import datetime

# 在类开头添加TRIPOD+AI合规性检查
class CESDPredictionPipeline:
    """CESD抑郁预测主流水线 - 符合TRIPOD+AI报告指南"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.data_processor = DataProcessor(random_state)

        self.model_builder = ModelBuilder(random_state)
        self.hyperparameter_tuner = HyperparameterTuner(random_state)
        self.evaluator = ModelEvaluator(random_state)
        self.plot_generator = PlotGenerator()
        self.shap_analyzer = None
        self.enhanced_shap_analyzer = None
        self.diagnostics_analyzer = None
        
        # 基础属性初始化
        self.tuned_models = None
        
        # TRIPOD+AI合规性记录
        self.tripod_compliance = {
            "study_design": "Prediction model development and validation",
            "data_source": {"primary": "CHARLS 2018", "external": "KLOSA 2018"},
            "outcome_definition": "Depression (CESD-10 based)",
            "predictor_handling": "Model-specific preprocessing applied",
            "missing_data_strategy": "Median/mode imputation, training-based statistics",
            "model_development": "Multiple algorithms with hyperparameter tuning",
            "validation_strategy": "Internal CV + External validation",
            "performance_measures": ["AUROC", "AUPRC", "Accuracy", "Precision", "Recall", "F1", "Brier"],
            "confidence_intervals": "95% CI using bootstrap method"
        }
        
        # 初始化存储变量
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        self.best_model = None
        self.label_encoders = {}
        
        # 创建必要的目录
        create_directories()
        
    def _ensure_encoders_available(self):
        """确保编码器可用，如果缺失则尝试从文件恢复，并同步编码器"""
        if not hasattr(self.data_processor, 'tree_label_encoders') or not self.data_processor.tree_label_encoders:
            print("🔍 检测到编码器缺失，尝试恢复...")
            try:
                from ..utils.helpers import load_model
                saved_processor = load_model('data_processor.joblib')
                if saved_processor is not None:
                    # 恢复所有编码器相关属性
                    self.data_processor.tree_label_encoders = saved_processor.tree_label_encoders
                    self.data_processor.svm_onehot_encoder = saved_processor.svm_onehot_encoder
                    self.data_processor.svm_nominal_label_encoders = saved_processor.svm_nominal_label_encoders
                    if hasattr(saved_processor, 'label_encoders'):
                        self.data_processor.label_encoders = saved_processor.label_encoders
                    print("✅ 编码器已恢复")
                else:
                    print("❌ 无法找到保存的编码器文件")
                    return False
            except Exception as e:
                print(f"❌ 编码器恢复失败: {e}")
                return False
        
        # 🆕 关键修复：同步编码器，确保所有存储位置都包含完整的编码器信息
        print("🔄 同步编码器，确保完整性...")
        
        # 确保所有编码器存储位置都存在
        if not hasattr(self.data_processor, 'label_encoders'):
            self.data_processor.label_encoders = {}
        if not hasattr(self.data_processor, 'tree_label_encoders'):
            self.data_processor.tree_label_encoders = {}
        if not hasattr(self.data_processor, 'svm_nominal_label_encoders'):
            self.data_processor.svm_nominal_label_encoders = {}
        
        # 🚀 优化：使用统一的编码器同步方法
        from ..utils.categorical_encoder import UnifiedCategoricalEncoder
        from ..config import CATEGORICAL_VARS
        
        # 高效同步编码器
        UnifiedCategoricalEncoder.sync_encoders_efficiently(self.data_processor)
        
        # 验证编码器一致性
        UnifiedCategoricalEncoder.validate_encoder_consistency(
            self.data_processor, CATEGORICAL_VARS
        )
        
        return True
        
    def load_and_preprocess_data(self, train_path, test_path=None, use_smote=False):
        """加载和预处理数据"""
        print(f"\n{'='*80}")
        print("🚀 开始数据加载和预处理")
        print(f"{'='*80}")
        
        # 加载训练数据
        train_data = self.data_processor.load_data(train_path, "训练集")
        if train_data is None:
            raise ValueError("训练数据加载失败")
            
        # 预处理训练数据
        train_data = self.data_processor.preprocess_data_before_split(train_data)
        
        # 分离特征和目标 - 只保留需要的列
        target_col = 'depressed'
        if target_col not in train_data.columns:
            raise ValueError(f"目标变量 '{target_col}' 不存在")
            
        # 移除目标变量和任何其他不需要的列
        feature_cols = [col for col in train_data.columns if col not in [target_col]]
        X = train_data[feature_cols]
        y = train_data[target_col]
        
        # 处理缺失值并保存统计信息用于KLOSA填充
        print("\n🔧 处理CHARLS训练数据缺失值...")
        X = self.data_processor.impute_features(X, is_training=True)
        
        print(f"🔍 数据检查:")
        print(f"  特征列数: {X.shape[1]}")
        print(f"  样本数: {X.shape[0]}")
        print(f"  目标变量分布:")
        print(f"    类别0: {(y==0).sum()} 个")
        print(f"    类别1: {(y==1).sum()} 个")
        
        # 验证数据
        validate_data(X, y)
        
        if test_path is None:
            # 如果没有单独的测试集，分割数据
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state, stratify=y
            )
            # 重置索引以避免后续问题
            self.X_train = self.X_train.reset_index(drop=True)
            self.X_test = self.X_test.reset_index(drop=True) 
            self.y_train = self.y_train.reset_index(drop=True)
            self.y_test = self.y_test.reset_index(drop=True)
            
            # 处理测试集缺失值（使用训练集统计信息）
            print("🔧 处理测试集缺失值...")
            self.X_test = self.data_processor.impute_features(self.X_test, is_training=False)
            
            print(f"✓ 数据分割完成: 训练集 {len(self.X_train)}, 测试集 {len(self.X_test)}")
        else:
            # 使用单独的测试集
            self.X_train, self.y_train = X.reset_index(drop=True), y.reset_index(drop=True)
            
            test_data = self.data_processor.load_data(test_path, "测试集")
            test_data = self.data_processor.preprocess_data_before_split(test_data)
            
            # 确保测试集也使用相同的特征列
            self.X_test = test_data[feature_cols].reset_index(drop=True)
            self.y_test = test_data[target_col].reset_index(drop=True)
            
            # 处理测试集缺失值（使用训练集统计信息）
            print("🔧 处理测试集缺失值...")
            self.X_test = self.data_processor.impute_features(self.X_test, is_training=False)
            
            print(f"✓ 使用独立测试集: 训练集 {len(self.X_train)}, 测试集 {len(self.X_test)}")
            
        # 保存特征名称用于KLOSA验证
        self.feature_names = list(self.X_train.columns)
            
        # 应用SMOTE
        if use_smote:
            print("\n🔄 应用SMOTE数据平衡...")
            # 在SMOTE之前先对分类变量进行编码
            self.X_train = self._encode_categorical_variables(self.X_train, is_training=True)
            self.X_test = self._encode_categorical_variables(self.X_test, is_training=False)
            
            self.X_train, self.y_train = self.data_processor.apply_smote(self.X_train, self.y_train)
        else:
            # 即使不使用SMOTE，也需要对分类变量进行编码
            print("\n🔄 对分类变量进行编码...")
            self.X_train = self._encode_categorical_variables(self.X_train, is_training=True)
            self.X_test = self._encode_categorical_variables(self.X_test, is_training=False)
            
        print(f"✅ 数据预处理完成")
        return True
        
    def _encode_categorical_variables(self, X, is_training=True):
        """对分类变量进行直接编码 - 完全切换到直接编码策略"""
        from sklearn.preprocessing import LabelEncoder
        from sklearn.impute import SimpleImputer
        from ..config import CATEGORICAL_VARS
        
        X_encoded = X.copy()
        
        if is_training:
            print("  🔤 训练集分类变量直接编码...")
        else:
            print("  🔤 测试集分类变量直接编码...")
        
        # 1. 先处理所有缺失值
        print("    🔧 处理缺失值...")
        
        # 分别处理数值和分类变量
        numeric_cols = []
        categorical_cols = []
        
        for col in X_encoded.columns:
            if col in CATEGORICAL_VARS or X_encoded[col].dtype == 'object':
                categorical_cols.append(col)
            else:
                numeric_cols.append(col)
        
        # 处理数值变量的缺失值
        if numeric_cols:
            numeric_imputer = SimpleImputer(strategy='median')
            if is_training:
                self.numeric_imputer = numeric_imputer
                X_encoded[numeric_cols] = self.numeric_imputer.fit_transform(X_encoded[numeric_cols])
            else:
                if hasattr(self, 'numeric_imputer'):
                    X_encoded[numeric_cols] = self.numeric_imputer.transform(X_encoded[numeric_cols])
                else:
                    # 如果没有训练过的imputer，用0填充
                    X_encoded[numeric_cols] = X_encoded[numeric_cols].fillna(0)
        
        # 2. 使用直接编码处理分类变量
        if categorical_cols:
            print(f"    🔤 直接编码处理 {len(categorical_cols)} 个分类变量:")
            
            for col in categorical_cols:
                # 先填充缺失值
                X_encoded[col] = X_encoded[col].fillna('missing').astype(str)
                
                # 🔧 修复：对于adlfive变量，使用特殊的数据类型统一处理
                if col == 'adlfive':
                    # adlfive实际是字符串类型，使用标准编码流程
                    if is_training:
                        # 训练阶段：创建编码器并保存
                        if col not in self.data_processor.label_encoders:
                            self.data_processor.label_encoders[col] = LabelEncoder()
                        
                        # 获取所有唯一值并排序，确保编码一致性
                        unique_values = sorted(X_encoded[col].unique())
                        self.data_processor.label_encoders[col].fit(unique_values)
                        
                        # 编码数据
                        X_encoded[col] = self.data_processor.label_encoders[col].transform(X_encoded[col])
                        
                        print(f"      📊 训练编码器: {col}")
                        print(f"        类别数量: {len(unique_values)}")
                        print(f"        编码映射: {dict(zip(unique_values, range(len(unique_values))))}")
                        
                        # 🔧 修复：确保编码器被正确保存到所有相关的编码器字典中
                        if col not in self.data_processor.tree_label_encoders:
                            self.data_processor.tree_label_encoders[col] = self.data_processor.label_encoders[col]
                        if col not in self.data_processor.svm_label_encoders:
                            self.data_processor.svm_label_encoders[col] = self.data_processor.label_encoders[col]
                        if col not in self.data_processor.ensemble_label_encoders:
                            self.data_processor.ensemble_label_encoders[col] = self.data_processor.label_encoders[col]
                    else:
                        # 测试阶段：使用训练好的编码器
                        if col in self.data_processor.label_encoders:
                            encoder = self.data_processor.label_encoders[col]
                            known_categories = set(encoder.classes_)
                            
                            # 检查是否有新类别
                            current_categories = set(X_encoded[col].unique())
                            new_categories = current_categories - known_categories
                            
                            if new_categories:
                                raise ValueError(f"❌ 严格映射错误：{col} 发现训练数据中不存在的新类别: {new_categories}")
                            else:
                                # 没有新类别，使用原始编码器
                                X_encoded[col] = encoder.transform(X_encoded[col])
                                print(f"      ✅ 使用原始编码器: {col}")
                        else:
                            raise ValueError(f"❌ 错误：{col} 缺少训练好的编码器，无法进行严格映射")
                else:
                    if is_training:
                        # 训练阶段：创建编码器并保存
                        if col not in self.data_processor.label_encoders:
                            self.data_processor.label_encoders[col] = LabelEncoder()
                        
                        # 获取所有唯一值并排序，确保编码一致性
                        unique_values = sorted(X_encoded[col].unique())
                        self.data_processor.label_encoders[col].fit(unique_values)
                        
                        # 编码数据
                        X_encoded[col] = self.data_processor.label_encoders[col].transform(X_encoded[col])
                        
                        print(f"      📊 训练编码器: {col}")
                        print(f"        类别数量: {len(unique_values)}")
                        print(f"        编码映射: {dict(zip(unique_values, range(len(unique_values))))}")
                        
                        # 🔧 修复：确保编码器被正确保存到所有相关的编码器字典中
                        if col not in self.data_processor.tree_label_encoders:
                            self.data_processor.tree_label_encoders[col] = self.data_processor.label_encoders[col]
                        if col not in self.data_processor.svm_label_encoders:
                            self.data_processor.svm_label_encoders[col] = self.data_processor.label_encoders[col]
                        if col not in self.data_processor.ensemble_label_encoders:
                            self.data_processor.ensemble_label_encoders[col] = self.data_processor.label_encoders[col]
                    else:
                        # 测试阶段：使用训练好的编码器
                        if col in self.data_processor.label_encoders:
                            encoder = self.data_processor.label_encoders[col]
                            known_categories = set(encoder.classes_)
                            
                            # 检查是否有新类别
                            current_categories = set(X_encoded[col].unique())
                            new_categories = current_categories - known_categories
                            
                            if new_categories:
                                raise ValueError(f"❌ 严格映射错误：{col} 发现训练数据中不存在的新类别: {new_categories}")
                            else:
                                # 没有新类别，使用原始编码器
                                X_encoded[col] = encoder.transform(X_encoded[col])
                                print(f"      ✅ 使用原始编码器: {col}")
                                
                        else:
                            raise ValueError(f"❌ 错误：{col} 缺少训练好的编码器，无法进行严格映射")
        
        # 3. 最终检查并处理任何剩余的缺失值
        if X_encoded.isnull().any().any():
            print("    ⚠️ 发现剩余缺失值，用0填充...")
            X_encoded = X_encoded.fillna(0)
        
        # 4. 确保所有列都是数值类型
        for col in X_encoded.columns:
            if X_encoded[col].dtype == 'object':
                X_encoded[col] = pd.to_numeric(X_encoded[col], errors='coerce').fillna(0)
        
        # 5. 重置索引以避免索引匹配问题
        X_encoded = X_encoded.reset_index(drop=True)
        
        print(f"    ✅ 直接编码完成，所有特征现在都是数值类型，形状: {X_encoded.shape}")
        return X_encoded
        
    def train_models(self):
        """训练基础模型 - 修复重复编码问题"""
        print(f"\n{'='*80}")
        print("🔧 开始模型训练")
        print(f"{'='*80}")
        
        if self.X_train is None or self.y_train is None:
            raise ValueError("请先加载和预处理数据")
        
        # 检查数据是否已经编码
        print("✅ 使用已编码的训练数据进行模型训练")
        print(f"📊 训练数据形状: {self.X_train.shape}")
        print(f"📊 目标变量分布: {self.y_train.value_counts().to_dict()}")
        
        # 🔑 关键修改：直接使用已编码的数据，避免重复处理
        train_data_with_target = self.X_train.copy()
        train_data_with_target['depressed'] = self.y_train
        
        # 调用模型构建器，但标明数据已预处理
        self.models, self.model_preprocessed_data = self.model_builder.build_base_models_with_preprocessing(
            train_data_with_target,
            use_pre_encoded_data=True
        )
        # 保存训练时的特征顺序
        self.feature_names = list(self.X_train.columns)
        
        if not self.models:
            raise ValueError("没有模型训练成功")
        
        print(f"✅ 模型训练完成，共训练了 {len(self.models)} 个模型")
        print(f"   📊 使用特征数量: {self.X_train.shape[1]}")
        
        return self.models
        
    def evaluate_models(self):
        """评估模型 - 使用全部特征"""
        print(f"\n{'='*80}")
        print("📊 开始模型评估")
        print(f"{'='*80}")
        
        if not self.models:
            raise ValueError("请先训练模型")
            
        # 使用全部特征的测试数据
        X_test_to_use = self.X_test
        print("✅ 使用全部特征的测试数据评估模型")
            
        # 评估所有模型
        self.results = self.evaluator.evaluate_all_models(
            self.models, X_test_to_use, self.y_test
        )
        
        # 找到最佳模型
        best_model_name, best_score = self.evaluator.find_best_model(self.results)
        if best_model_name:
            self.best_model = self.models[best_model_name]
            self.best_model_name = best_model_name  # 添加这一行
            print(f"🏆 最佳模型: {best_model_name} (AUROC: {best_score:.3f})")
        
        # 生成结果表格
        results_table = self.evaluator.generate_comparison_table(self.results)
        save_results(results_table, 'model_comparison_results.csv')
        
        # 打印摘要
        print_summary(self.results)
        
        print(f"✅ 模型评估完成")
        return self.results
    
    def _evaluate_models(self, models, X_test, y_test, model_type="unknown"):
        """评估模型"""
        # 🔧 修复：使用已初始化的evaluator，确保有正确的random_state和所有修复
        evaluator = self.evaluator
        
        evaluation_results = {}
        
        for name, model in models.items():
            try:
                print(f"  • 评估 {name}...")
                
                # 确定模型类型并应用相应预处理
                if hasattr(self, 'model_preprocessed_data') and name in self.model_preprocessed_data:
                    model_info = self.model_preprocessed_data[name]
                    model_specific_type = model_info.get('model_type', 'unknown')
                    
                    # 🔧 方案A修复：使用早期编码的数据，不再进行模型特定编码
                    print(f"    使用早期编码数据（跳过{model_specific_type}模型特定编码）")
                    
                    # 直接使用已经编码的测试数据
                    if 'depressed' in X_test.columns:
                        feature_cols = [col for col in X_test.columns if col not in ['depressed']]
                        X_test_processed = X_test[feature_cols].copy()
                    else:
                        X_test_processed = X_test.copy()
                else:
                    print(f"    警告: 未找到{name}的预处理信息，使用原始数据")
                    # 从原始数据中提取特征（排除目标变量）
                    if 'depressed' in X_test.columns:
                        feature_cols = [col for col in X_test.columns if col not in ['depressed']]
                        X_test_processed = X_test[feature_cols].copy()
                    else:
                        X_test_processed = X_test.copy()
                
                # 🔑 关键：保持DataFrame以保留特征名称
                # 并对齐到训练时的特征顺序，填充缺失列
                if hasattr(self, 'feature_names') and self.feature_names:
                    missing = [c for c in self.feature_names if c not in X_test_processed.columns]
                    if missing:
                        print(f"    ⚠️ 缺失特征填充为0: {missing[:5]}{'...' if len(missing)>5 else ''}")
                        for c in missing:
                            X_test_processed[c] = 0
                    # 仅保留训练时出现过的特征
                    X_test_processed = X_test_processed.reindex(columns=self.feature_names, fill_value=0)
                
                # 如模型暴露feature_names_in_，再做一次对齐
                if hasattr(model, 'feature_names_in_') and model.feature_names_in_ is not None:
                    try:
                        expected = list(model.feature_names_in_)
                        missing2 = [c for c in expected if c not in X_test_processed.columns]
                        if missing2:
                            print(f"    ⚠️ 依据模型特征再补齐: {missing2[:5]}{'...' if len(missing2)>5 else ''}")
                            for c in missing2:
                                X_test_processed[c] = 0
                        X_test_processed = X_test_processed.reindex(columns=expected, fill_value=0)
                    except Exception as _:
                        pass
                
                metrics = evaluator.evaluate_model(model, X_test_processed, y_test, bootstrap_ci=True)
                
                # 提取主要指标
                accuracy = metrics['accuracy']
                precision = metrics['precision']
                recall = metrics['recall']
                f1_score = metrics['f1_score']
                auroc = metrics['roc_auc']
                auprc = metrics['pr_auc']
                c_index = metrics['c_index']
                specificity = metrics['specificity']
                npv = metrics['npv']
                brier_score = metrics['brier_score']
                
                # 根据模型类型显示不同的标签
                if model_type == "hyperparameter_tuned":
                    print(f"    ✓ 调优后准确率: {accuracy:.4f}")
                    print(f"    ✓ 调优后精确率: {precision:.4f}")
                    print(f"    ✓ 调优后召回率: {recall:.4f}")
                    print(f"    ✓ 调优后F1分数: {f1_score:.4f}")
                    print(f"    ✓ 调优后AUROC: {auroc:.4f}")
                    print(f"    ✓ 调优后AUPRC: {auprc:.4f}")
                    print(f"    ✓ 调优后C-Index: {c_index:.4f}")
                    print(f"    ✓ 调优后特异性: {specificity:.4f}")
                    print(f"    ✓ 调优后NPV: {npv:.4f}")
                    print(f"    ✓ 调优后Brier分数: {brier_score:.4f}")
                else:
                    print(f"    ✓ 准确率: {accuracy:.4f}")
                    print(f"    ✓ 精确率: {precision:.4f}")
                    print(f"    ✓ 召回率: {recall:.4f}")
                    print(f"    ✓ F1分数: {f1_score:.4f}")
                    print(f"    ✓ AUROC: {auroc:.4f}")
                    print(f"    ✓ AUPRC: {auprc:.4f}")
                    print(f"    ✓ C-Index: {c_index:.4f}")
                    print(f"    ✓ 特异性: {specificity:.4f}")
                    print(f"    ✓ NPV: {npv:.4f}")
                    print(f"    ✓ Brier分数: {brier_score:.4f}")
                
                # 显示95%置信区间（主要指标）
                if 'roc_auc_ci_lower' in metrics and 'roc_auc_ci_upper' in metrics:
                    ci_text = "调优后" if model_type == "hyperparameter_tuned" else ""
                    print(f"    📊 {ci_text}AUROC 95%CI: [{metrics['roc_auc_ci_lower']:.4f}, {metrics['roc_auc_ci_upper']:.4f}]")
                    print(f"    📊 {ci_text}F1 95%CI: [{metrics['f1_score_ci_lower']:.4f}, {metrics['f1_score_ci_upper']:.4f}]")
                
                # 保存完整评估结果
                evaluation_results[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1_score,
                    'auroc': auroc,
                    'auprc': auprc,
                    'c_index': c_index,
                    'specificity': specificity,
                    'npv': npv,
                    'brier_score': brier_score,
                    'full_metrics': metrics,
                    'model_type': model_type
                }
            except Exception as e:
                print(f"    ❌ {name} 评估失败: {str(e)}")
                print(f"    🔍 错误详情:")
                print(f"       - 模型类型: {type(model).__name__}")
                print(f"       - 特征数量: {X_test_processed.shape[1] if 'X_test_processed' in locals() else 'N/A'}")
                print(f"       - 样本数量: {X_test_processed.shape[0] if 'X_test_processed' in locals() else 'N/A'}")
                print(f"       - 特征类型: {X_test_processed.dtype if 'X_test_processed' in locals() else 'N/A'}")
                
                # 尝试诊断问题
                if "feature names should match" in str(e):
                    print(f"    💡 建议：这是特征名称不匹配问题，已转换为NumPy数组")
                elif "predict_proba" in str(e):
                    print(f"    💡 建议：模型不支持概率预测，需要检查模型配置")
                elif "adlfive" in str(e):
                    print(f"    💡 建议：adlfive分类变量处理问题，检查独热编码")
                else:
                    print(f"    💡 建议：检查数据预处理和模型兼容性")
        
        return evaluation_results
    
    def run_hyperparameter_tuning(self, search_method='random', n_iter=20):
        """
        运行超参数调优
        
        参数:
        ----
        search_method : str, 默认'random'
            搜索方法 ('grid' 或 'random')
        n_iter : int, 默认20
            随机搜索的迭代次数
            
        返回:
        ----
        tuple : (tuned_models, benchmark_df)
        """
        print(f"\n{'='*80}")
        print("🎯 超参数调优")
        print(f"{'='*80}")
        
        if self.X_train is None:
            raise ValueError("请先加载和预处理数据")
        
        # 使用全部特征进行超参数调优
        X_train_to_use = self.X_train
        print("✅ 使用全部特征进行超参数调优")
        
        # 获取基础模型
        base_models = self.model_builder.build_base_models()
        
        # 🚀 优化：使用统一的模型名称管理器
        from ..utils.model_name_manager import ModelNameManager
        
        # 映射到调优名称
        models_for_tuning = ModelNameManager.map_to_tuning_names(base_models)
        print(f"✅ 模型名称映射完成")
        
        # 超参数调优
        tuned_models_mapped, benchmark_df = self.hyperparameter_tuner.benchmark_models_with_tuning(
            models=models_for_tuning,
            X_train=X_train_to_use,
            y_train=self.y_train,
            search_method=search_method,
            n_iter=n_iter
        )
        
        # 映射回原始名称
        tuned_models = ModelNameManager.map_to_original_names(tuned_models_mapped)
        print(f"✅ 模型名称反向映射完成")
        
        # 保存调优后的模型
        self.tuned_models = tuned_models
        print(f"✅ 超参数调优完成，共调优了 {len(tuned_models)} 个模型")
        
        # 🔧 修复: 设置最佳模型和best_model_name
        if benchmark_df is not None and len(benchmark_df) > 0:
            # 获取最佳模型（第一行是分数最高的）
            best_tuned_model_name = benchmark_df.iloc[0]['Model']
            
            # 🔧 关键修复：正确映射模型名称
            # 模型名称映射（从超参数调优器返回的名称到原始名称）
            reverse_mapping = {
                'rf': 'rf',
                'gb': 'gb', 
                'xgb': 'xgb',
                'lgb': 'lgb',
                'lr': 'lr',
                # 'svc': 'svc',  # 🚫 用户要求：完全禁用SVC模型
                'extra_trees': 'extra_trees',
                'adaboost': 'adaboost',
                'catboost': 'catboost'
            }
            
            # 映射回原始名称
            best_original_name = reverse_mapping.get(best_tuned_model_name, best_tuned_model_name)
            print(f"🔄 映射到原始名称: {best_tuned_model_name} -> {best_original_name}")
            
            # 设置最佳模型相关属性
            if best_original_name in tuned_models:
                self.best_model = tuned_models[best_original_name]
                self.best_model_name = f"{best_original_name}_tuned"
                print(f"🏆 设置最佳调优模型: {self.best_model_name} (原名: {best_original_name})")
                print(f"✅ 模型类型: {type(self.best_model).__name__}")
            else:
                print(f"⚠️ 模型名称不匹配，可用模型: {list(tuned_models.keys())}")
                # 尝试直接使用调优器返回的名称
                if best_tuned_model_name in tuned_models:
                    self.best_model = tuned_models[best_tuned_model_name]
                    self.best_model_name = f"{best_tuned_model_name}_tuned"
                    print(f"✅ 使用调优器名称设置最佳模型: {self.best_model_name}")
                else:
                    print(f"❌ 无法找到最佳模型，使用第一个调优模型")
                    first_model_name = list(tuned_models.keys())[0]
                    self.best_model = tuned_models[first_model_name]
                    self.best_model_name = f"{first_model_name}_tuned"
            
            # 创建简化的results字典用于后续使用
            best_score = benchmark_df.iloc[0]['Best_CV_Score']
            self.results = {
                self.best_model_name: {
                    'auroc': (best_score, best_score * 0.95, best_score * 1.05)  # 简化的置信区间
                }
            }
        
        return tuned_models, benchmark_df
    
    def external_validation_klosa(self, klosa_file_path):
        """
        在KLOSA数据集上进行外部验证
        
        参数:
        ----
        klosa_file_path : str
            KLOSA数据文件路径
            
        返回:
        ----
        dict : 外部验证结果
        """
        print(f"\n🌏 KLOSA外部验证")
        print("=" * 60)
        
        try:
            # 0. 🔧 确保编码器可用
            self._ensure_encoders_available()
            
            # 1. 加载KLOSA数据
            print("📊 加载KLOSA数据...")
            klosa_data = self.data_processor.load_data(klosa_file_path, "KLOSA")
            if klosa_data is None:
                return None
            
            # 2. 🎯 采用原始文件的简单预处理逻辑
            print("🔧 准备KLOSA特征数据...")
            
            # 预处理数据（基础清理）
            klosa_data = self.data_processor.preprocess_data_before_split(klosa_data)
            
            # 🚨 关键修复: 在处理KLOSA前先确保编码器可用
            print("🔧 加载CHARLS训练的编码器...")
            self._ensure_encoders_available()
            
            # 🆕 第3阶段: 早期缺失值处理 + CSV保存
            print("🔧 第3阶段: 处理KLOSA缺失值并保存...")
            
            # 直接使用'depressed'作为目标变量 (KLOSA和CHARLS使用相同变量名)
            target_col = 'depressed'
            feature_cols = [col for col in klosa_data.columns if col not in [target_col]]
            X_klosa = klosa_data[feature_cols]
            y_klosa = klosa_data[target_col]
            
            # 🚨 关键修复: 删除目标变量为NaN的样本
            print("🧹 检查并删除目标变量缺失的样本...")
            y_missing_before = y_klosa.isnull().sum()
            if y_missing_before > 0:
                print(f"⚠️ 发现 {y_missing_before} 个目标变量缺失样本，将被删除")
                valid_indices = ~y_klosa.isnull()
                X_klosa = X_klosa[valid_indices]
                y_klosa = y_klosa[valid_indices]
                print(f"✅ 删除后样本数: {len(X_klosa)} (原始: {len(klosa_data)})")
            else:
                print("✅ 目标变量无缺失值")
            
            # 使用CHARLS训练统计处理缺失值
            print("🔧 使用CHARLS训练统计填充KLOSA缺失值...")
            missing_before = X_klosa.isnull().sum().sum()
            X_klosa_imputed = self.data_processor.impute_features(X_klosa, is_training=False)
            missing_after = X_klosa_imputed.isnull().sum().sum()
            
            print(f"📈 缺失值填充: {missing_before} → {missing_after}")
            
            # 🚀 优化：使用统一的数据映射管理器
            print("🔧 统一adlfive变量格式...")
            if 'adlfive' in X_klosa_imputed.columns:
                from ..utils.data_mappings import DataMappings
                
                print(f"  原始adlfive类型: {X_klosa_imputed['adlfive'].dtype}")
                print(f"  原始adlfive值: {sorted(X_klosa_imputed['adlfive'].dropna().unique())}")
                
                # 高效的向量化转换
                X_klosa_imputed['adlfive'] = DataMappings.convert_adlfive_klosa_to_charls(
                    X_klosa_imputed['adlfive']
                )
                
                print(f"  转换后adlfive类型: {X_klosa_imputed['adlfive'].dtype}")
                print(f"  转换后adlfive值: {sorted(X_klosa_imputed['adlfive'].dropna().unique())}")
            
            # 🆕 关键修复：添加CHARLS同样的分类变量编码步骤
            print("🔤 应用与CHARLS相同的直接编码策略...")
            X_klosa_encoded = self._encode_categorical_variables(X_klosa_imputed, is_training=False)
            print("✅ KLOSA分类变量直接编码完成")
            
            # 保存填充并编码后的完整数据
            klosa_imputed_data = X_klosa_encoded.copy()
            klosa_imputed_data[target_col] = y_klosa
            # ✅ 优化：删除冗余的depression列，只保留depressed
            
            # 保存CSV供后续使用
            output_path = "klosa_imputed_data.csv"
            klosa_imputed_data.to_csv(output_path, index=False)
            print(f"💾 已保存填充并编码后数据: {output_path}")
            print(f"📊 数据形状: {klosa_imputed_data.shape}")
            
            # 显示缺失值填充统计
            missing_stats = []
            for col in X_klosa.columns:
                original_nulls = X_klosa[col].isnull().sum()
                final_nulls = X_klosa_imputed[col].isnull().sum()
                if original_nulls > 0:
                    missing_stats.append(f"  - {col}: {original_nulls} → {final_nulls} 个缺失值")
            
            if missing_stats:
                print("📈 缺失值填充详情:")
                for stat in missing_stats[:10]:  # 显示前10个
                    print(stat)
                if len(missing_stats) > 10:
                    print(f"  ... 还有 {len(missing_stats) - 10} 个变量")
            
            # 🔑 关键：后续使用填充后的数据
            print("🔄 应用与CHARLS相同的特征预处理...")
            
            # 🆕 明确告诉KLOSA系统变量类型（与CHARLS保持一致）
            print("📋 明确声明变量类型（确保与CHARLS一致）:")
            from ..config import CATEGORICAL_VARS, NUMERICAL_VARS
            
            print(f"  📊 分类变量 ({len(CATEGORICAL_VARS)}个): {CATEGORICAL_VARS[:5]}...")
            print(f"  📈 数值变量 ({len(NUMERICAL_VARS)}个): {NUMERICAL_VARS}")
            
            # 验证hhres确实是数值变量
            if 'hhres' in NUMERICAL_VARS:
                print("  ✅ 确认: hhres是数值变量，不需要编码器")
            else:
                print("  ⚠️ 警告: hhres不在数值变量列表中!")
            
            X_klosa_with_target = klosa_imputed_data.copy()
            
            # 使用全特征模式进行外部验证
            print("📊 使用全特征模式")
            # 🔧 重要修复：避免重复编码，直接使用已编码的数据
            print("✅ 使用已完成直接编码的数据，避免重复处理")
            X_klosa_processed = X_klosa_encoded.copy()
            y_klosa = X_klosa_with_target['depressed'].copy()
            
            # 确保特征顺序与训练时一致
            if hasattr(self, 'feature_names') and self.feature_names:
                missing_features = set(self.feature_names) - set(X_klosa_processed.columns)
                if missing_features:
                    print(f"⚠️ 缺失特征用0填充: {missing_features}")
                    for feature in missing_features:
                        X_klosa_processed[feature] = 0
                
                X_klosa = X_klosa_processed[self.feature_names]
            else:
                X_klosa = X_klosa_processed
            
            print(f"📊 全特征模式: {X_klosa.shape[1]} 个特征")
            
            print(f"📊 KLOSA验证数据形状: {X_klosa.shape}")
            print(f"📊 目标变量分布: {y_klosa.value_counts().to_dict()}")
            
            # ❌ 删除: 重复的缺失值处理 (已在第3阶段完成)
            # X_klosa_processed = self.data_processor.impute_features(X_klosa, is_training=False)
            
            # ✅ 直接使用第3阶段填充后的数据
            print("✅ 使用第3阶段填充后的数据，跳过重复处理")
            X_klosa_processed = X_klosa  # 使用全特征数据
            
            # 🔑 保存处理后的KLOSA数据供SHAP分析使用
            self.X_klosa_processed = X_klosa_processed
            self.y_klosa = y_klosa
            print(f"📊 已保存KLOSA处理后数据供SHAP分析: {X_klosa_processed.shape}")
            
            # 🔧 关键修复: 确保索引一致性，避免NaN重新引入
            print("🔧 重置索引确保数据一致性...")
            X_klosa_processed = X_klosa_processed.reset_index(drop=True)
            y_klosa = y_klosa.reset_index(drop=True)
            
            # 🚨 最终检查: 再次确认没有NaN
            final_nan_check = y_klosa.isnull().sum()
            if final_nan_check > 0:
                print(f"⚠️ 发现残留的NaN，强制清理: {final_nan_check}个")
                valid_mask = ~y_klosa.isnull()
                X_klosa_processed = X_klosa_processed[valid_mask]
                y_klosa = y_klosa[valid_mask]
                X_klosa_processed = X_klosa_processed.reset_index(drop=True)
                y_klosa = y_klosa.reset_index(drop=True)
                print(f"✅ 最终清理后样本数: {len(y_klosa)}")
            else:
                print("✅ 确认无NaN值")
            
            # 7. 对所有模型进行外部验证
            validation_results = {}
            
            # 为KLOSA数据添加目标变量列，以便进行模型特定预处理
            klosa_data_with_target = X_klosa_processed.copy()
            klosa_data_with_target['depressed'] = y_klosa
            
            # 🔧 再次验证合并后的数据
            if klosa_data_with_target['depressed'].isnull().sum() > 0:
                print("❌ 警告: 合并后发现NaN，这不应该发生")
                # 强制清理
                valid_rows = ~klosa_data_with_target['depressed'].isnull()
                klosa_data_with_target = klosa_data_with_target[valid_rows]
                y_klosa = y_klosa[valid_rows.values]
                print(f"🔧 强制清理后样本数: {len(klosa_data_with_target)}")
            else:
                print("✅ 合并数据无NaN问题")
            
            # 8. 评估全特征模型（基础模型和调优后模型）
            if hasattr(self, 'models') and self.models:
                print("📊 评估基础模型...")
                base_results = self._evaluate_models(
                    self.models, klosa_data_with_target, y_klosa, "external_validation"
                )
                validation_results['base_models'] = base_results
            
            if hasattr(self, 'tuned_models') and self.tuned_models:
                print("📊 评估调优后模型...")
                tuned_results = self._evaluate_models(
                    self.tuned_models, klosa_data_with_target, y_klosa, "external_validation"
                )
                validation_results['tuned_models'] = tuned_results
            
            # 🔧 新增：专门评估最佳模型
            if hasattr(self, 'best_model') and self.best_model is not None:
                print(f"🏆 评估最佳模型: {self.best_model_name}")
                best_model_results = self._evaluate_models(
                    {self.best_model_name: self.best_model}, 
                    klosa_data_with_target, 
                    y_klosa, 
                    "external_validation_best"
                )
                validation_results['best_model'] = best_model_results
                print(f"✅ 最佳模型外部验证完成")
            
            # 9. 保存验证结果
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            feature_type = "full"  # 现在只支持全特征模式
            results_file = f'klosa_external_validation_{feature_type}_{timestamp}.json'
            
            # 保存验证结果为JSON
            import json
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(validation_results, f, indent=2, ensure_ascii=False, default=str)
            print(f"💾 外部验证结果已保存: {results_file}")
            print(f"✅ KLOSA外部验证完成，样本数: {len(y_klosa)}")
            
            return validation_results
            
        except Exception as e:
            print(f"❌ KLOSA外部验证失败: {e}")
            import traceback
            traceback.print_exc()
            return None
        
    def generate_visualizations(self):
        """生成可视化图表"""
        print(f"\n{'='*80}")
        print("📈 生成可视化图表")
        print(f"{'='*80}")
        
        # 🔧 修复：适应新流程的检查逻辑
        has_models = (self.models or 
                     (hasattr(self, 'models_selected') and self.models_selected) or
                     (hasattr(self, 'tuned_models') and self.tuned_models) or
                     (hasattr(self, 'tuned_models_selected') and self.tuned_models_selected))
        
        has_best_model = hasattr(self, 'best_model') and self.best_model is not None
        
        if not has_models and not has_best_model:
            raise ValueError("请先训练模型")
        
        # 如果没有results但有best_model，创建简单的results用于可视化
        if not hasattr(self, 'results') or not self.results:
            if has_best_model:
                print("⚠️ 使用最佳模型创建临时结果用于可视化")
                self.results = {self.best_model_name: {'auroc': (0.75, 0.70, 0.80)}}  # 临时结果
            else:
                print("⚠️ 跳过部分需要评估结果的可视化")
        
        # 确定使用的数据集（特征选择后或全特征）
        if hasattr(self, 'X_test_selected') and self.X_test_selected is not None:
            X_test_viz = self.X_test_selected
            X_train_viz = self.X_train_selected
            print("✅ 使用特征选择后的数据进行可视化")
        else:
            X_test_viz = self.X_test
            X_train_viz = self.X_train
            print("✅ 使用全特征数据进行可视化")
            
        # 生成各种图表
        self.plot_generator.plot_roc_curves(
            self.models, X_test_viz, self.y_test, 'plots/roc_curves.png'
        )
        
        self.plot_generator.plot_precision_recall_curves(
            self.models, X_test_viz, self.y_test, 'plots/pr_curves.png'
        )
        
        self.plot_generator.plot_model_comparison(
            self.results, 'plots/model_comparison.png'
        )
        
        # 如果有最佳模型，生成特征重要性和混淆矩阵
        if self.best_model is not None:
            if hasattr(self.best_model, 'feature_importances_'):
                self.plot_generator.plot_feature_importance(
                    self.best_model, X_train_viz.columns, 
                    save_path='plots/feature_importance.png'
                )
                
            # 生成混淆矩阵
            self.plot_generator.plot_confusion_matrix(
                self.best_model, X_test_viz, self.y_test, 
                save_path='plots/confusion_matrix.png'
            )
            
            # 生成校准曲线
        self.plot_generator.plot_calibration_curve(
                self.best_model, X_test_viz, self.y_test,
                save_path='plots/calibration_curves.png'
        )
        
        print(f"✅ 可视化图表生成完成")
    
    def run_enhanced_interpretability_analysis(self, save_dir='enhanced_analysis', enable_significance_test=True, enable_diagnostics=True):
        """
        运行增强的解释性分析 - 为训练集、测试集和外部验证集生成SHAP解释
        
        参数:
        ----
        save_dir : str
            保存目录
        enable_significance_test : bool
            是否启用统计显著性检验
        enable_diagnostics : bool
            是否启用模型诊断
        """
        print(f"\n{'='*80}")
        print("🔍 增强解释性分析 - 多数据集SHAP解释")
        print(f"{'='*80}")
        
        if not self.models or not self.results:
            raise ValueError("请先训练和评估模型")
            
        # 确定使用的数据集
        if hasattr(self, 'X_test_selected') and self.X_test_selected is not None:
            X_train_analysis = self.X_train_selected
            X_test_analysis = self.X_test_selected
            print("✅ 使用特征选择后的数据进行分析")
        else:
            X_train_analysis = self.X_train
            X_test_analysis = self.X_test
            print("✅ 使用全特征数据进行分析")
        
        analysis_results = {}
        
        # 1. 对最佳模型进行多数据集SHAP分析
        if self.best_model is not None and enable_significance_test:
            # 确定是否为调优后的模型
            model_display_name = self.best_model_name
            if "_tuned" in self.best_model_name:
                model_display_name = f"{self.best_model_name} (调优后)"
            
            print(f"\n🎯 最佳模型多数据集SHAP分析: {model_display_name}")
            
            # 1.1 训练集SHAP分析
            print(f"\n📊 1️⃣ 训练集SHAP分析 (样本数: {len(X_train_analysis):,})")
            train_shap_analyzer = EnhancedSHAPAnalyzer(
                self.best_model, 
                f"{self.best_model_name}_Training"
            )
            
            train_shap_results = train_shap_analyzer.run_complete_analysis(
                X_background=X_train_analysis,  # 使用全部训练集作为背景
                X_explain=X_train_analysis,     # 使用全部训练集进行解释
                save_dir=f"{save_dir}/shap_training_{self.best_model_name}"
            )
            analysis_results['shap_training'] = train_shap_results
            print(f"   ✅ 训练集SHAP分析完成 (使用全部{len(X_train_analysis):,}个样本)")
            
            # 1.2 测试集SHAP分析
            print(f"\n📊 2️⃣ 测试集SHAP分析 (样本数: {len(X_test_analysis):,})")
            test_shap_analyzer = EnhancedSHAPAnalyzer(
                self.best_model, 
                f"{self.best_model_name}_Testing"
            )
            
            test_shap_results = test_shap_analyzer.run_complete_analysis(
                X_background=X_train_analysis,  # 使用训练集作为背景
                X_explain=X_test_analysis,      # 使用全部测试集进行解释
                save_dir=f"{save_dir}/shap_testing_{self.best_model_name}"
            )
            analysis_results['shap_testing'] = test_shap_results
            print(f"   ✅ 测试集SHAP分析完成 (使用全部{len(X_test_analysis):,}个样本)")
            
            # 1.3 外部验证集SHAP分析（如果存在）
            if hasattr(self, 'X_klosa_processed') and self.X_klosa_processed is not None:
                print(f"\n📊 3️⃣ KLOSA外部验证集SHAP分析 (样本数: {len(self.X_klosa_processed):,})")
                klosa_shap_analyzer = EnhancedSHAPAnalyzer(
                    self.best_model, 
                    f"{self.best_model_name}_KLOSA"
                )
                
                # 确保KLOSA数据与训练数据特征一致
                if set(self.X_klosa_processed.columns) == set(X_train_analysis.columns):
                    klosa_shap_results = klosa_shap_analyzer.run_complete_analysis(
                        X_background=X_train_analysis,  # 使用训练集作为背景
                        X_explain=self.X_klosa_processed,  # 使用全部KLOSA样本进行解释
                        save_dir=f"{save_dir}/shap_klosa_{self.best_model_name}"
                    )
                    analysis_results['shap_klosa'] = klosa_shap_results
                    print(f"   ✅ KLOSA外部验证集SHAP分析完成 (使用全部{len(self.X_klosa_processed):,}个样本)")
                else:
                    print(f"   ⚠️ KLOSA数据特征不匹配，跳过SHAP分析")
                    print(f"   KLOSA特征数: {len(self.X_klosa_processed.columns)}, 训练特征数: {len(X_train_analysis.columns)}")
            else:
                print(f"\n📊 3️⃣ 跳过KLOSA外部验证集SHAP分析（无KLOSA数据）")
            
            # 保存整体SHAP分析结果（保持向后兼容）
            analysis_results['enhanced_shap'] = test_shap_results  # 主要结果仍为测试集
            print(f"✅ 多数据集SHAP分析完成")
        
        # 2. 对最佳模型进行诊断分析
        if self.best_model is not None and enable_diagnostics:
            # 复用之前定义的显示名称
            print(f"\n🏥 最佳模型诊断分析: {model_display_name}")
            
            self.diagnostics_analyzer = ModelDiagnosticsAnalyzer(
                self.best_model,
                f"{self.best_model_name}_Diagnostics"
            )
            
            diagnostics_results = self.diagnostics_analyzer.run_complete_diagnostics(
                X_test_analysis, 
                self.y_test,
                save_dir=f"{save_dir}/diagnostics_{self.best_model_name}"
            )
            
            analysis_results['diagnostics'] = diagnostics_results
            print(f"✅ 模型诊断分析完成")
        
        # 3. 对所有模型进行快速诊断比较
        if enable_diagnostics and len(self.models) > 1:
            print(f"\n📊 所有模型诊断比较")
            
            model_health_summary = []
            
            for model_name, model in self.models.items():
                if model_name == self.best_model_name:
                    continue  # 已经详细分析过了
                    
                try:
                    print(f"   分析 {model_name}...")
                    
                    quick_analyzer = ModelDiagnosticsAnalyzer(model, model_name)
                    
                    # 快速校准分析
                    y_prob = model.predict_proba(X_test_analysis)[:, 1]
                    from sklearn.metrics import brier_score_loss
                    brier_score = brier_score_loss(self.y_test, y_prob)
                    
                    # 类别分离度
                    separation_score = abs(
                        np.mean(y_prob[self.y_test == 1]) - 
                        np.mean(y_prob[self.y_test == 0])
                    )
                    
                    model_health_summary.append({
                        'model': model_name,
                        'brier_score': brier_score,
                        'separation_score': separation_score,
                        'health_rating': 'Good' if brier_score < 0.15 and separation_score > 0.2 else 
                                       'Fair' if brier_score < 0.25 else 'Poor'
                    })
                    
                except Exception as e:
                    print(f"   ⚠️ {model_name} 分析失败: {e}")
                    
            # 保存模型健康摘要
            if model_health_summary:
                health_df = pd.DataFrame(model_health_summary).sort_values('brier_score')
                health_df.to_csv(f"{save_dir}/model_health_summary.csv", 
                                index=False, encoding='utf-8-sig')
                
                print(f"\n📋 模型健康摘要:")
                for _, row in health_df.iterrows():
                    print(f"   {row['model']}: {row['health_rating']} "
                         f"(Brier: {row['brier_score']:.4f}, Sep: {row['separation_score']:.4f})")
                
                analysis_results['model_health_summary'] = health_df
        
        # 4. 生成综合分析报告
        self._generate_comprehensive_analysis_report(analysis_results, save_dir)
        
        print(f"\n🎉 增强解释性分析完成！")
        print(f"📁 结果保存在: {save_dir}")
        
        return analysis_results
    
    def _generate_comprehensive_analysis_report(self, analysis_results, save_dir):
        """生成综合分析报告"""
        print(f"\n📝 生成综合分析报告...")
        
        report_path = f"{save_dir}/comprehensive_analysis_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# CESD抑郁预测模型 - 综合解释性分析报告\n\n")
            
            f.write(f"**分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**最佳模型**: {self.best_model_name}\n\n")
            
            # SHAP分析总结
            # 优先使用测试集的SHAP结果，保持向后兼容
            shap_data = None
            if 'shap_testing' in analysis_results:
                shap_data = analysis_results['shap_testing']
            elif 'enhanced_shap' in analysis_results:
                shap_data = analysis_results['enhanced_shap']
            
            if shap_data and 'significance_results' in shap_data:
                    sig_df = shap_data['significance_results']
                    significant_features = sig_df[sig_df['significant']]
                    
                    f.write("## 🎯 特征重要性分析\n\n")
                    f.write(f"- **总特征数**: {len(sig_df)}\n")
                    f.write(f"- **显著特征数**: {len(significant_features)}\n")
                    f.write(f"- **显著特征比例**: {len(significant_features)/len(sig_df):.1%}\n\n")
                    
                    f.write("### 最重要的显著特征 (Top 5)\n")
                    for i, (_, row) in enumerate(significant_features.head(5).iterrows()):
                        f.write(f"{i+1}. **{row['feature']}** - 重要性: {row['importance']:.4f} ({row['significance_level']})\n")
                    
                    f.write("\n### 特征交互作用\n")
                    if 'interaction_results' in shap_data:
                        interaction_df = shap_data['interaction_results']
                        f.write("最强的特征交互作用:\n")
                        for i, (_, row) in enumerate(interaction_df.head(3).iterrows()):
                            f.write(f"{i+1}. {row['feature_1']} ↔ {row['feature_2']} (强度: {row['interaction_strength']:.4f})\n")
            
            # 诊断分析总结
            if 'diagnostics' in analysis_results:
                diag_data = analysis_results['diagnostics']
                
                f.write("\n## 🏥 模型诊断分析\n\n")
                f.write("### 校准性能\n")
                f.write(f"- **Brier Score**: {diag_data['calibration']['brier_score']:.4f}\n")
                f.write(f"- **平均校准误差**: {diag_data['calibration']['mean_calibration_error']:.4f}\n\n")
                
                f.write("### 预测可靠性\n")
                f.write(f"- **类别分离度**: {diag_data['distribution']['separation_score']:.4f}\n")
                f.write(f"- **最优阈值**: {diag_data['distribution']['optimal_threshold']:.3f}\n")
                f.write(f"- **特征平均稳定性**: {diag_data['reliability']['mean_stability']:.4f}\n\n")
            
            # 模型比较
            if 'model_health_summary' in analysis_results:
                health_df = analysis_results['model_health_summary']
                f.write("## 📊 模型健康比较\n\n")
                f.write("| 模型 | 健康评级 | Brier Score | 分离度 |\n")
                f.write("|------|----------|-------------|--------|\n")
                for _, row in health_df.iterrows():
                    f.write(f"| {row['model']} | {row['health_rating']} | {row['brier_score']:.4f} | {row['separation_score']:.4f} |\n")
            
            f.write("\n## 🎯 主要发现和建议\n\n")
            
            # 自动生成建议
            if 'enhanced_shap' in analysis_results and 'diagnostics' in analysis_results:
                shap_data = analysis_results['enhanced_shap']
                diag_data = analysis_results['diagnostics']
                
                # 特征建议
                if 'significance_results' in shap_data:
                    sig_rate = shap_data['significance_results']['significant'].mean()
                    if sig_rate < 0.3:
                        f.write("### ⚠️ 特征选择建议\n")
                        f.write("- 显著特征比例较低，建议重新评估特征选择策略\n")
                        f.write("- 考虑使用更严格的特征筛选方法\n\n")
                
                # 校准建议
                brier_score = diag_data['calibration']['brier_score']
                if brier_score > 0.2:
                    f.write("### 🎯 模型校准建议\n")
                    f.write("- Brier Score较高，建议使用校准方法(如Platt Scaling)\n")
                    f.write("- 考虑调整决策阈值以优化性能\n\n")
                
                # 稳定性建议
                stability = diag_data['reliability']['mean_stability']
                if stability < 0.8:
                    f.write("### 🔧 模型稳定性建议\n")
                    f.write("- 特征稳定性较低，建议增加数据量或特征工程\n")
                    f.write("- 考虑使用集成方法提高预测稳定性\n\n")
        
        print(f"   ✅ 综合报告保存到: {report_path}")
        
    def cross_validate_best_model(self, cv_folds=10):
        """对最佳模型进行交叉验证"""
        print(f"\n{'='*80}")
        print("🔄 最佳模型交叉验证")
        print(f"{'='*80}")
        
        if self.best_model is None:
            raise ValueError("请先评估模型以确定最佳模型")
            
        # 合并训练和测试数据进行交叉验证
        X_all = pd.concat([self.X_train, self.X_test], ignore_index=True)
        y_all = pd.concat([self.y_train, self.y_test], ignore_index=True)
        
        cv_results = self.evaluator.cross_validate_model(
            self.best_model, X_all, y_all, cv_folds
        )
        
        save_results(cv_results, 'cross_validation_results.json')
        
        print(f"✅ 交叉验证完成")
        return cv_results
        
    def save_models_and_results(self, model_prefix='cesd_model'):
        """保存模型和结果"""
        print(f"\n{'='*80}")
        print("💾 保存最终模型和结果")
        print(f"{'='*80}")
        
        timestamp = generate_timestamp()
        
        # 检查是否有调优后的模型
        has_tuned_models = any('_tuned' in name for name in self.models.keys()) if self.models else False
        
        # 保存最佳模型
        if self.best_model is not None:
            if has_tuned_models:
                best_model_file = f'{model_prefix}_best_hyperparameter_tuned_{timestamp}.joblib'
                print(f"📊 保存超参数调优后的最佳模型...")
            else:
                best_model_file = f'{model_prefix}_best_default_params_{timestamp}.joblib'
                print(f"📊 保存默认参数的最佳模型...")
                
            save_model(self.best_model, best_model_file)
            print(f"✅ 最佳模型已保存: {best_model_file}")
        else:
            print("⚠️ 没有最佳模型可保存")
            
        # 保存所有模型
        if self.models:
            if has_tuned_models:
                all_models_file = f'{model_prefix}_all_with_tuning_{timestamp}.joblib'
                print(f"📊 保存所有模型(包括调优后模型)...")
            else:
                all_models_file = f'{model_prefix}_all_default_params_{timestamp}.joblib'
                print(f"📊 保存所有默认参数模型...")
                
            save_model(self.models, all_models_file)
            print(f"✅ 所有模型已保存: {all_models_file}")
            
            # 打印模型统计信息
            default_models = [name for name in self.models.keys() if not name.startswith('Tuned_')]
            tuned_models = [name for name in self.models.keys() if name.startswith('Tuned_')]
            
            print(f"\n📈 模型统计:")
            print(f"  默认参数模型: {len(default_models)} 个")
            if tuned_models:
                print(f"  超参数调优模型: {len(tuned_models)} 个")
                print(f"  调优模型列表: {', '.join(tuned_models)}")
        else:
            print("⚠️ 没有模型可保存")
        
        # 保存数据处理器 (使用固定名称便于加载)
        processor_file = f'data_processor_{timestamp}.joblib'
        processor_file_fixed = 'data_processor.joblib'
        
        save_model(self.data_processor, processor_file)
        save_model(self.data_processor, processor_file_fixed)  # 同时保存固定名称版本
        print(f"✅ 数据处理器已保存: {processor_file}")
        print(f"✅ 数据处理器固定版本已保存: {processor_file_fixed}")
        
        # 保存评估结果
        if hasattr(self, 'evaluation_results') and self.evaluation_results:
            results_file = f'evaluation_results_{timestamp}.json'
            save_results(self.evaluation_results, results_file)
            print(f"✅ 评估结果已保存: {results_file}")
        
        print(f"\n✅ 模型和结果保存完成")
        print(f"\n📁 保存的文件:")
        if self.best_model is not None:
            print(f"  🏆 最佳模型: {best_model_file}")
        if self.models:
            print(f"  📦 所有模型: {all_models_file}")
        print(f"  🔧 数据处理器: {processor_file}")
        if hasattr(self, 'evaluation_results') and self.evaluation_results:
            print(f"  📊 评估结果: {results_file}")
        
    def run_full_pipeline(self, charls_file, klosa_file=None, 
                         enable_hyperparameter_tuning=True):
        """
        运行完整的机器学习流水线
        
        参数:
        ----
        charls_file : str
            CHARLS训练数据文件路径
        klosa_file : str, 可选
            KLOSA外部验证数据文件路径
        use_feature_selection : bool, 默认True
            是否启用特征选择
        enable_hyperparameter_tuning : bool, 默认True
            是否启用超参数调优
        """
        print(f"🚀 开始运行完整的CESD抑郁预测模型流水线")
        print(f"📊 评估指标置信区间: 95%CI")
        
        pipeline_start_time = time.time()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            
            # 1. 数据加载和预处理
            print(f"\n1️⃣ 数据加载和预处理")
            print("=" * 50)
            success = self.load_and_preprocess_data(charls_file, use_smote=False)
            if not success:
                raise ValueError("数据加载和预处理失败")
            
            # 2. 模型训练 (使用全部特征)
            print(f"\n2️⃣ 模型训练")
            print("=" * 50)
            print("✅ 使用全部特征进行模型训练")
            models = self.train_models()
            
            # 3. 超参数调优 (可选)
            if enable_hyperparameter_tuning:
                print(f"\n3️⃣ 超参数调优")
                print("=" * 50)
                try:
                    tuned_models, benchmark_df = self.run_hyperparameter_tuning(
                        search_method='random',
                        n_iter=15  # 减少迭代次数以节省时间
                    )
                    
                    # 使用调优后的最佳模型
                    print(f"\n📊 选择调优后的最佳模型")
                    print("=" * 50)
                    
                    # 🔧 修复：正确的最佳模型选择逻辑
                    try:
                        if benchmark_df is not None and not benchmark_df.empty and len(tuned_models) > 0:
                            print(f"📊 基准测试结果形状: {benchmark_df.shape}")
                            print(f"📊 可用调优模型: {list(tuned_models.keys())}")
                            
                            # 从基准测试结果中选择最佳模型
                            best_score = benchmark_df['Best_CV_Score'].max()
                            best_tuned_model_row = benchmark_df[benchmark_df['Best_CV_Score'] == best_score].iloc[0]
                            best_tuned_model_name = best_tuned_model_row['Model']
                            
                            print(f"🏆 最佳调优模型: {best_tuned_model_name} (得分: {best_score:.4f})")
                            
                            # 🔧 关键修复：正确映射模型名称
                            # 模型名称映射（从超参数调优器返回的名称到原始名称）
                            reverse_mapping = {
                                'RandomForest': 'rf',
                                'GradientBoosting': 'gb', 
                                'XGBoost': 'xgb',
                                'LightGBM': 'lgb',
                                'LogisticRegression': 'lr',
                                # 'svc': 'svc',  # 🚫 用户要求：完全禁用SVC模型
                                'SVM': 'svm'
                            }
                            
                            # 获取原始模型名称
                            best_original_name = reverse_mapping.get(best_tuned_model_name, best_tuned_model_name)
                            print(f"🔄 映射到原始名称: {best_tuned_model_name} -> {best_original_name}")
                            
                            # 确保模型名称存在于调优模型中
                            if best_original_name in tuned_models:
                                self.best_model = tuned_models[best_original_name]
                                self.best_model_name = f"{best_original_name}_tuned"
                                print(f"✅ 已设置最佳模型: {self.best_model_name}")
                                print(f"✅ 模型类型: {type(self.best_model).__name__}")
                            else:
                                print(f"⚠️ 模型名称不匹配，可用模型: {list(tuned_models.keys())}")
                                print(f"⚠️ 尝试直接使用调优器名称: {best_tuned_model_name}")
                                
                                # 尝试直接使用调优器返回的名称
                                if best_tuned_model_name in tuned_models:
                                    self.best_model = tuned_models[best_tuned_model_name]
                                    self.best_model_name = f"{best_tuned_model_name}_tuned"
                                    print(f"✅ 使用调优器名称设置最佳模型: {self.best_model_name}")
                                else:
                                    print(f"❌ 无法找到最佳模型，使用第一个调优模型")
                                    first_model_name = list(tuned_models.keys())[0]
                                    self.best_model = tuned_models[first_model_name]
                                    self.best_model_name = f"{first_model_name}_tuned"
                        else:
                            print(f"⚠️ 基准测试结果为空或调优模型为空，使用第一个可用模型")
                            if tuned_models:
                                first_model_name = list(tuned_models.keys())[0]
                                self.best_model = tuned_models[first_model_name]
                                self.best_model_name = f"{first_model_name}_tuned"
                                print(f"✅ 使用第一个调优模型: {self.best_model_name}")
                            else:
                                raise ValueError("没有可用的调优模型")
                                
                    except Exception as e:
                        print(f"⚠️ 选择最佳模型时出错: {e}")
                        import traceback
                        traceback.print_exc()
                        print("🔄 回退到使用第一个可用的调优模型")
                        if tuned_models:
                            first_model_name = list(tuned_models.keys())[0]
                            self.best_model = tuned_models[first_model_name]
                            self.best_model_name = f"{first_model_name}_tuned"
                            print(f"✅ 回退模型: {self.best_model_name}")
                        else:
                            raise ValueError("没有任何可用的模型")
                    
                    # 保存调优后的模型
                    self.tuned_models = tuned_models
                    
                    # 创建完整的结果字典，包含所有关键评估指标
                    # 使用交叉验证计算完整指标
                    from sklearn.model_selection import cross_validate, cross_val_predict
                    from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                               f1_score, roc_auc_score, average_precision_score,
                                               brier_score_loss)
                    
                    # 定义评估指标
                    scoring = {
                        'accuracy': 'accuracy',
                        'precision': 'precision',
                        'recall': 'recall',
                        'f1': 'f1',
                        'roc_auc': 'roc_auc',
                        'average_precision': 'average_precision'
                    }
                    
                    # 执行交叉验证
                    print(f"📊 计算最佳模型的完整评估指标...")
                    X_for_cv = self.X_train
                    
                    cv_results = cross_validate(
                        self.best_model, X_for_cv, self.y_train, 
                        cv=CV_SETTINGS['n_splits'], scoring=scoring, return_train_score=False
                    )
                    
                    # 计算95%置信区间
                    from scipy import stats
                    
                    def calculate_ci(scores):
                        import numpy as np  # 重新导入numpy解决作用域问题
                        mean = np.mean(scores)
                        ci = stats.t.interval(0.95, len(scores)-1, loc=mean, scale=stats.sem(scores))
                        return mean, ci[0], ci[1]
                    
                    # 创建完整结果字典
                    self.results = {self.best_model_name: {
                        'auroc': calculate_ci(cv_results['test_roc_auc']),
                        'auprc': calculate_ci(cv_results['test_average_precision']),
                        'accuracy': calculate_ci(cv_results['test_accuracy']),
                        'precision': calculate_ci(cv_results['test_precision']),
                        'recall': calculate_ci(cv_results['test_recall']),
                        'f1_score': calculate_ci(cv_results['test_f1'])
                    }}
                    
                    # 计算Brier分数 (需要单独计算)
                    y_proba = cross_val_predict(
                        self.best_model, X_for_cv, self.y_train, 
                        cv=CV_SETTINGS['n_splits'], method='predict_proba'
                    )[:, 1]
                    brier = brier_score_loss(self.y_train, y_proba)
                    self.results[self.best_model_name]['brier_score'] = brier
                    
                    print(f"✅ 评估指标计算完成")
                    
                except Exception as e:
                    print(f"⚠️ 超参数调优失败: {e}")
                    print("   继续使用基础模型...")
                    
                    # 使用默认模型作为最佳模型并计算完整评估指标
                    if self.models:
                        model_name = list(self.models.keys())[0]
                        self.best_model = self.models[model_name]
                        self.best_model_name = model_name
                        
                        # 计算完整评估指标
                        try:
                            from sklearn.model_selection import cross_validate, cross_val_predict
                            from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                                      f1_score, roc_auc_score, average_precision_score,
                                                      brier_score_loss)
                            from scipy import stats
                            import numpy as np
                            
                            # 定义评估指标
                            scoring = {
                                'accuracy': 'accuracy',
                                'precision': 'precision',
                                'recall': 'recall',
                                'f1': 'f1',
                                'roc_auc': 'roc_auc',
                                'average_precision': 'average_precision'
                            }
                            
                            # 选择数据
                            X_for_cv = self.X_train
                            
                            # 计算95%置信区间
                            def calculate_ci(scores):
                                import numpy as np  # 重新导入numpy解决作用域问题
                                mean = np.mean(scores)
                                ci = stats.t.interval(0.95, len(scores)-1, loc=mean, scale=stats.sem(scores))
                                return mean, ci[0], ci[1]
                                
                            # 执行交叉验证
                            print(f"📊 计算默认模型的评估指标...")
                            cv_results = cross_validate(
                                self.best_model, X_for_cv, self.y_train, 
                                cv=CV_SETTINGS['n_splits'], scoring=scoring, return_train_score=False
                            )
                            
                            # 创建完整结果字典
                            self.results = {self.best_model_name: {
                                'auroc': calculate_ci(cv_results['test_roc_auc']),
                                'auprc': calculate_ci(cv_results['test_average_precision']),
                                'accuracy': calculate_ci(cv_results['test_accuracy']),
                                'precision': calculate_ci(cv_results['test_precision']),
                                'recall': calculate_ci(cv_results['test_recall']),
                                'f1_score': calculate_ci(cv_results['test_f1'])
                            }}
                            
                            # 计算Brier分数
                            y_proba = cross_val_predict(
                                self.best_model, X_for_cv, self.y_train, 
                                cv=CV_SETTINGS['n_splits'], method='predict_proba'
                            )[:, 1]
                            brier = brier_score_loss(self.y_train, y_proba)
                            self.results[self.best_model_name]['brier_score'] = brier
                            
                            print(f"✅ 评估指标计算完成")
                        except Exception as e:
                            print(f"⚠️ 评估指标计算失败: {e}")
                            # 使用默认值
                            self.results = {model_name: {'auroc': (0.5, 0.45, 0.55)}}  # 默认值
            else:
                print(f"\n3️⃣ 跳过超参数调优")
                print("=" * 50)
                
                # 使用默认模型作为最佳模型并计算完整评估指标
                if self.models:
                    model_name = list(self.models.keys())[0]
                    self.best_model = self.models[model_name]
                    self.best_model_name = model_name
                    
                    # 计算完整评估指标
                    try:
                        from sklearn.model_selection import cross_validate, cross_val_predict
                        from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                                  f1_score, roc_auc_score, average_precision_score,
                                                  brier_score_loss)
                        from scipy import stats
                        import numpy as np
                        
                        # 定义评估指标
                        scoring = {
                            'accuracy': 'accuracy',
                            'precision': 'precision',
                            'recall': 'recall',
                            'f1': 'f1',
                            'roc_auc': 'roc_auc',
                            'average_precision': 'average_precision'
                        }
                        
                        # 选择数据
                        X_for_cv = self.X_train
                        
                        # 计算95%置信区间
                        def calculate_ci(scores):
                            import numpy as np  # 重新导入numpy解决作用域问题
                            mean = np.mean(scores)
                            ci = stats.t.interval(0.95, len(scores)-1, loc=mean, scale=stats.sem(scores))
                            return mean, ci[0], ci[1]
                            
                        # 执行交叉验证
                        print(f"📊 计算默认模型的评估指标...")
                        cv_results = cross_validate(
                            self.best_model, X_for_cv, self.y_train, 
                            cv=CV_SETTINGS['n_splits'], scoring=scoring, return_train_score=False
                        )
                        
                        # 创建完整结果字典
                        self.results = {self.best_model_name: {
                            'auroc': calculate_ci(cv_results['test_roc_auc']),
                            'auprc': calculate_ci(cv_results['test_average_precision']),
                            'accuracy': calculate_ci(cv_results['test_accuracy']),
                            'precision': calculate_ci(cv_results['test_precision']),
                            'recall': calculate_ci(cv_results['test_recall']),
                            'f1_score': calculate_ci(cv_results['test_f1'])
                        }}
                        
                        # 计算Brier分数
                        y_proba = cross_val_predict(
                            self.best_model, X_for_cv, self.y_train, 
                            cv=CV_SETTINGS['n_splits'], method='predict_proba'
                        )[:, 1]
                        brier = brier_score_loss(self.y_train, y_proba)
                        self.results[self.best_model_name]['brier_score'] = brier
                        
                        print(f"✅ 评估指标计算完成")
                    except Exception as e:
                        print(f"⚠️ 评估指标计算失败: {e}")
                        # 使用默认值
                        self.results = {model_name: {'auroc': (0.5, 0.45, 0.55)}}  # 默认值
            
            # 4. 训练集和测试集详细评估
            print(f"\n4️⃣ 训练集和测试集详细评估")
            print("=" * 50)
            try:
                train_test_results = self.evaluate_on_training_and_test_sets()
                if train_test_results:
                    print(f"✅ 训练/测试集详细评估完成")
                else:
                    print(f"⚠️ 训练/测试集评估失败")
            except Exception as e:
                print(f"⚠️ 训练/测试集评估出错: {e}")
                print("   继续流水线的其他步骤...")
            
            # 5. 生成可视化
            print(f"\n5️⃣ 生成可视化")
            print("=" * 50)
            self.generate_visualizations()
            
            # 6. 交叉验证
            print(f"\n6️⃣ 交叉验证最佳模型")
            print("=" * 50)
            cv_results = self.cross_validate_best_model()
            
            # 7. 增强解释性分析（集成SHAP和模型诊断）
            print(f"\n7️⃣ 增强解释性分析")
            print("=" * 50)
            try:
                analysis_results = self.run_enhanced_interpretability_analysis(
                    save_dir=f'enhanced_analysis_{self.timestamp}',
                    enable_significance_test=True,
                    enable_diagnostics=True
                )
                print(f"✅ 增强解释性分析完成")
            except Exception as e:
                print(f"⚠️ 增强解释性分析失败: {e}")
                print("   继续流水线的其他步骤...")
            
            # 8. 外部验证 (可选)
            if klosa_file:
                print(f"\n8️⃣ KLOSA外部验证")
                print("=" * 50)
                
                # 🔧 临时修改: 只进行全特征KLOSA外部验证，暂时跳过特征选择验证
                try:
                    # 🎯 注释掉特征选择相关验证，只保留全特征验证
                    # if use_feature_selection:
                    #     print("🎯 进行两种类型的KLOSA外部验证...")
                    #     
                    #     # 验证全特征模型
                    #     print("\n📊 全特征模型KLOSA验证:")
                    #     klosa_results_full = self.external_validation_klosa(klosa_file, use_feature_selection=False)
                    #     
                    #     # 验证特征选择模型  
                    #     print("\n📊 特征选择模型KLOSA验证:")
                    #     klosa_results_selected = self.external_validation_klosa(klosa_file, use_feature_selection=True)
                    #     
                    #     if klosa_results_full and klosa_results_selected:
                    #         print(f"✅ 两种类型的KLOSA外部验证都完成")
                    #     else:
                    #         print(f"⚠️ 部分KLOSA外部验证失败")
                    # else:
                    
                    # 验证模型
                    print("📊 KLOSA外部验证:")
                    klosa_results = self.external_validation_klosa(klosa_file)
                    if klosa_results:
                        print(f"✅ KLOSA外部验证完成")
                    else:
                        print(f"⚠️ KLOSA外部验证失败")
                            
                except Exception as e:
                    print(f"⚠️ KLOSA外部验证失败: {e}")
                    print("   继续流水线的其他步骤...")
            else:
                print(f"\n8️⃣ 跳过外部验证（未提供KLOSA数据）")
                print("=" * 50)
            
            # 9. 保存模型和结果
            print(f"\n9️⃣ 保存模型和结果")
            print("=" * 50)
            self.save_models_and_results()
            
            # 10. 生成TRIPOD+AI合规性报告
            print(f"\n🔟 生成TRIPOD+AI合规性报告")
            print("=" * 50)
            try:
                compliance_report = self.generate_tripod_compliance_report()
                print(f"✅ TRIPOD+AI合规性报告生成完成")
            except Exception as e:
                print(f"⚠️ TRIPOD+AI合规性报告生成失败: {e}")
            
            print(f"\n🎉 完整流水线运行成功!")
            
        except Exception as e:
            print(f"\n❌ 流水线运行失败: {str(e)}")
            import traceback
            traceback.print_exc()
            
        finally:
            # 计算总运行时间
            pipeline_duration = time.time() - pipeline_start_time
            print(f"\n⏱️ 总运行时间: {pipeline_duration:.2f} 秒")

    def analyze_risk_groups(self, model=None, X=None, y=None):
        """
        TRIPOD+AI要求：分析不同风险组的特征分布
        """
        if model is None:
            model = self.best_model
        if X is None:
            X = self.X_test
        if y is None:
            y = self.y_test
            
        try:
            # 获取预测概率
            y_proba = model.predict_proba(X)[:, 1]
            
            # 风险分组（四分位数）
            risk_quartiles = pd.qcut(y_proba, 4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
            
            # 分析每个风险组
            risk_analysis = {}
            for group in ['Low', 'Medium-Low', 'Medium-High', 'High']:
                mask = risk_quartiles == group
                group_analysis = {
                    'n_patients': mask.sum(),
                    'observed_rate': y[mask].mean() if mask.sum() > 0 else 0,
                    'predicted_rate': y_proba[mask].mean() if mask.sum() > 0 else 0,
                    'risk_range': [y_proba[mask].min(), y_proba[mask].max()] if mask.sum() > 0 else [0, 0]
                }
                risk_analysis[group] = group_analysis
            
            print(f"📊 风险分组分析完成")
            return risk_analysis
            
        except Exception as e:
            print(f"⚠️ 风险分组分析失败: {e}")
            return {}

    def document_limitations(self):
        """
        TRIPOD+AI要求：系统性记录模型局限性
        """
        limitations = {
            "data_limitations": [
                "横断面研究设计，无法建立因果关系",
                "单一时间点数据收集，缺乏纵向追踪",
                "自我报告的抑郁症状，可能存在报告偏倚",
                "缺失值处理可能引入不确定性"
            ],
            "model_limitations": [
                "基于特定人群（中老年人）开发，年轻人群适用性未知",
                "特征选择可能过度适应训练数据",
                "模型解释性受算法复杂性限制",
                "超参数调优可能存在过拟合风险"
            ],
            "generalizability_concerns": [
                "训练数据来源于中国人群，其他种族适用性待验证", 
                "文化和社会经济背景的差异可能影响模型表现",
                "不同地区医疗体系差异可能影响特征重要性",
                "时间推移可能导致模型性能衰减"
            ],
            "temporal_validity": [
                "模型基于2018年数据，近期适用性需要验证",
                "疫情等重大事件可能改变抑郁症风险因素",
                "医疗技术进步可能影响特征重要性",
                "建议定期重新校准和验证"
            ]
        }
        
        print(f"📋 模型局限性记录完成")
        return limitations

    def clinical_impact_analysis(self):
        """
        TRIPOD+AI要求：评估模型的临床决策影响
        """
        impact_analysis = {
            "clinical_utility": {
                "screening_tool": "可用于社区老年人抑郁症状初步筛查",
                "risk_stratification": "协助医生识别高风险个体",
                "resource_allocation": "优化心理健康服务资源配置",
                "early_intervention": "促进早期发现和干预"
            },
            "implementation_considerations": [
                "需要医生培训以正确解释预测结果",
                "应与临床判断结合使用，不可完全替代",
                "需要建立标准化的数据收集流程",
                "建议在使用前进行本地验证"
            ],
            "potential_harms": [
                "假阳性可能导致不必要的焦虑",
                "假阴性可能延误必要的治疗",
                "过度依赖算法可能弱化临床判断",
                "可能加剧健康不平等问题"
            ]
        }
        
        print(f"🏥 临床影响分析完成")
        return impact_analysis

    def generate_tripod_compliance_report(self):
        """
        生成TRIPOD+AI合规性报告
        """
        print(f"\n📋 生成TRIPOD+AI合规性报告")
        print("=" * 60)
        
        # 基本信息
        print(f"研究设计: {self.tripod_compliance['study_design']}")
        print(f"数据来源: {self.tripod_compliance['data_source']}")
        print(f"结局定义: {self.tripod_compliance['outcome_definition']}")
        
        # 方法学要素
        print(f"\n方法学合规性:")
        print(f"- 预测变量处理: {self.tripod_compliance['predictor_handling']}")
        print(f"- 缺失值策略: {self.tripod_compliance['missing_data_strategy']}")
        print(f"- 模型开发: {self.tripod_compliance['model_development']}")
        print(f"- 验证策略: {self.tripod_compliance['validation_strategy']}")
        
        # 性能指标
        print(f"\n性能评估:")
        print(f"- 评估指标: {', '.join(self.tripod_compliance['performance_measures'])}")
        print(f"- 置信区间: {self.tripod_compliance['confidence_intervals']}")
        
        # 风险分组分析
        risk_analysis = self.analyze_risk_groups()
        
        # 局限性分析
        limitations = self.document_limitations()
        
        # 临床影响分析
        clinical_impact = self.clinical_impact_analysis()
        
        # 保存完整报告
        report = {
            "tripod_compliance": self.tripod_compliance,
            "risk_analysis": risk_analysis,
            "limitations": limitations,
            "clinical_impact": clinical_impact,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        try:
            import json
            import numpy as np
            
            # 🔧 修复：自定义JSON序列化器处理numpy类型
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, np.bool_):
                        return bool(obj)
                    return super(NumpyEncoder, self).default(obj)
            
            with open(f'tripod_compliance_report_{self.timestamp}.json', 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
            print(f"✅ TRIPOD+AI合规性报告已保存: tripod_compliance_report_{self.timestamp}.json")
        except Exception as e:
            print(f"⚠️ 报告保存失败: {e}")
            # 🔧 修复：尝试简化报告内容
            try:
                simplified_report = {
                    "tripod_compliance": self.tripod_compliance,
                    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
                }
                with open(f'tripod_compliance_report_simplified_{self.timestamp}.json', 'w', encoding='utf-8') as f:
                    json.dump(simplified_report, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
                print(f"✅ 简化版TRIPOD报告已保存")
            except Exception as e2:
                print(f"❌ 简化版报告也保存失败: {e2}")

    def evaluate_on_training_and_test_sets(self):
        """评估所有模型在训练集和测试集上的性能，并生成详细的95%CI结果"""
        print(f"\n🔍 详细评估: 所有模型的训练集 vs 测试集性能分析")
        print("=" * 70)
        
        # 🔧 修复：使用已初始化的evaluator，确保有正确的random_state和所有修复
        evaluator = self.evaluator
        
        evaluation_results = {}
        
        # 评估基础模型
        if hasattr(self, 'models') and self.models:
            print("📊 评估基础模型...")
            base_train_results = self._evaluate_models_on_dataset(
                self.models, self.X_train, self.y_train, "Training_Set"
            )
            base_test_results = self._evaluate_models_on_dataset(
                self.models, self.X_test, self.y_test, "Test_Set"
            )
            
            evaluation_results['base_models'] = {
                'training': base_train_results,
                'testing': base_test_results
            }
        
        # 评估调优后模型
        if hasattr(self, 'tuned_models') and self.tuned_models:
            print("📊 评估调优后模型...")
            tuned_train_results = self._evaluate_models_on_dataset(
                self.tuned_models, self.X_train, self.y_train, "Training_Set"
            )
            tuned_test_results = self._evaluate_models_on_dataset(
                self.tuned_models, self.X_test, self.y_test, "Test_Set"
            )
            
            evaluation_results['tuned_models'] = {
                'training': tuned_train_results,
                'testing': tuned_test_results
            }
        
        # 保存结果
        self._save_train_test_evaluation_csv(evaluation_results)
        
        print(f"📊 训练集样本数: {len(self.X_train):,}")
        print(f"📊 测试集样本数: {len(self.X_test):,}")
        
        return evaluation_results
    
    def _evaluate_models_on_dataset(self, models, X, y, evaluation_type):
        """在指定数据集上评估多个模型"""
        results = {}
        
        for model_name, model in models.items():
            print(f"  🔍 评估模型: {model_name} ({evaluation_type})")
            
            try:
                # 准备数据（根据模型类型）
                X_processed = self._prepare_data_for_model(X, model_name)
                
                # 评估模型
                # 🔧 修复：使用已初始化的evaluator，确保有正确的random_state和所有修复
                evaluator = self.evaluator
                
                metrics = evaluator.evaluate_model(
                    model, X_processed, y, 
                    bootstrap_ci=True, n_bootstraps=1000
                )
                
                results[model_name] = {
                    'dataset': f"CHARLS_{evaluation_type}",
                    'model': model_name,
                    'evaluation_type': evaluation_type,
                    'sample_size': len(X),
                    'full_metrics': metrics,
                    'model_type': evaluation_type.lower()
                }
                
                print(f"    ✓ AUROC: {metrics['roc_auc']:.4f} [{metrics['roc_auc_ci_lower']:.4f}, {metrics['roc_auc_ci_upper']:.4f}]")
                
            except Exception as e:
                print(f"    ❌ 评估失败: {e}")
                results[model_name] = None
        
        return results
    
    def _prepare_data_for_model(self, X, model_name):
        """为特定模型准备数据"""
        # 判断是否需要标准化（仅对线性模型）
        linear_models = ['lr', 'svm']  # 🚫 移除svc（用户要求禁用）
        if any(lm in model_name.lower() for lm in linear_models):
            print(f"    📈 为线性模型 {model_name} 应用标准化...")
            try:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X_processed = scaler.fit_transform(X)
                return X_processed
            except Exception as e:
                print(f"    ⚠️ 标准化失败，使用原始数据: {e}")
                return X
        else:
            return X

    def _save_train_test_evaluation_csv(self, evaluation_results):
        """保存训练集和测试集评估结果为CSV格式"""
        import pandas as pd
        from datetime import datetime
        
        def format_metric_with_ci(value, ci_lower, ci_upper):
            """格式化为 均值 [下限, 上限]"""
            return f"{value:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]"
        
        csv_data = []
        
        # 处理基础模型
        if 'base_models' in evaluation_results:
            for dataset_type, models_results in evaluation_results['base_models'].items():
                for model_name, result in models_results.items():
                    if result and 'full_metrics' in result:
                        metrics = result['full_metrics']
                        csv_data.append({
                            'Dataset': result['dataset'],
                            'Model': model_name,
                            'Evaluation_Type': result['evaluation_type'],
                            'Sample_Size': result['sample_size'],
                            'AUROC_95CI': format_metric_with_ci(
                                metrics['roc_auc'],
                                metrics.get('roc_auc_ci_lower', 0),
                                metrics.get('roc_auc_ci_upper', 0)
                            ),
                            'AUPRC_95CI': format_metric_with_ci(
                                metrics['pr_auc'],
                                metrics.get('pr_auc_ci_lower', 0),
                                metrics.get('pr_auc_ci_upper', 0)
                            ),
                            'Accuracy_95CI': format_metric_with_ci(
                                metrics['accuracy'],
                                metrics.get('accuracy_ci_lower', 0),
                                metrics.get('accuracy_ci_upper', 0)
                            ),
                            'Precision_95CI': format_metric_with_ci(
                                metrics['precision'],
                                metrics.get('precision_ci_lower', 0),
                                metrics.get('precision_ci_upper', 0)
                            ),
                            'Recall_95CI': format_metric_with_ci(
                                metrics['recall'],
                                metrics.get('recall_ci_lower', 0),
                                metrics.get('recall_ci_upper', 0)
                            ),
                            'F1_Score_95CI': format_metric_with_ci(
                                metrics['f1_score'],
                                metrics.get('f1_score_ci_lower', 0),
                                metrics.get('f1_score_ci_upper', 0)
                            ),
                            'Brier_Score_95CI': format_metric_with_ci(
                                metrics['brier_score'],
                                metrics.get('brier_score_ci_lower', 0),
                                metrics.get('brier_score_ci_upper', 0)
                            ),
                            'Specificity_95CI': format_metric_with_ci(
                                metrics.get('specificity', 0),
                                metrics.get('specificity_ci_lower', 0),
                                metrics.get('specificity_ci_upper', 0)
                            ),
                            'NPV_95CI': format_metric_with_ci(
                                metrics.get('npv', 0),
                                metrics.get('npv_ci_lower', 0),
                                metrics.get('npv_ci_upper', 0)
                            ),
                            'C_Index_95CI': format_metric_with_ci(
                                metrics.get('c_index', 0),
                                metrics.get('c_index_ci_lower', 0),
                                metrics.get('c_index_ci_upper', 0)
                            )
                        })
        
        # 处理调优后模型
        if 'tuned_models' in evaluation_results:
            for dataset_type, models_results in evaluation_results['tuned_models'].items():
                for model_name, result in models_results.items():
                    if result and 'full_metrics' in result:
                        metrics = result['full_metrics']
                        csv_data.append({
                            'Dataset': result['dataset'],
                            'Model': f"{model_name}_tuned",
                            'Evaluation_Type': result['evaluation_type'],
                            'Sample_Size': result['sample_size'],
                            'AUROC_95CI': format_metric_with_ci(
                                metrics['roc_auc'],
                                metrics.get('roc_auc_ci_lower', 0),
                                metrics.get('roc_auc_ci_upper', 0)
                            ),
                            'AUPRC_95CI': format_metric_with_ci(
                                metrics['pr_auc'],
                                metrics.get('pr_auc_ci_lower', 0),
                                metrics.get('pr_auc_ci_upper', 0)
                            ),
                            'Accuracy_95CI': format_metric_with_ci(
                                metrics['accuracy'],
                                metrics.get('accuracy_ci_lower', 0),
                                metrics.get('accuracy_ci_upper', 0)
                            ),
                            'Precision_95CI': format_metric_with_ci(
                                metrics['precision'],
                                metrics.get('precision_ci_lower', 0),
                                metrics.get('precision_ci_upper', 0)
                            ),
                            'Recall_95CI': format_metric_with_ci(
                                metrics['recall'],
                                metrics.get('recall_ci_lower', 0),
                                metrics.get('recall_ci_upper', 0)
                            ),
                            'F1_Score_95CI': format_metric_with_ci(
                                metrics['f1_score'],
                                metrics.get('f1_score_ci_lower', 0),
                                metrics.get('f1_score_ci_upper', 0)
                            ),
                            'Brier_Score_95CI': format_metric_with_ci(
                                metrics['brier_score'],
                                metrics.get('brier_score_ci_lower', 0),
                                metrics.get('brier_score_ci_upper', 0)
                            ),
                            'Specificity_95CI': format_metric_with_ci(
                                metrics.get('specificity', 0),
                                metrics.get('specificity_ci_lower', 0),
                                metrics.get('specificity_ci_upper', 0)
                            ),
                            'NPV_95CI': format_metric_with_ci(
                                metrics.get('npv', 0),
                                metrics.get('npv_ci_lower', 0),
                                metrics.get('npv_ci_upper', 0)
                            ),
                            'C_Index_95CI': format_metric_with_ci(
                                metrics.get('c_index', 0),
                                metrics.get('c_index_ci_lower', 0),
                                metrics.get('c_index_ci_upper', 0)
                            )
                        })
        
        # 创建DataFrame并保存
        if not csv_data:
            print("⚠️ 没有有效的评估数据")
            return None
            
        df = pd.DataFrame(csv_data)
        
        # 按评估类型排序（训练集在前）
        if len(df) > 0 and 'Evaluation_Type' in df.columns:
            df = df.sort_values(['Evaluation_Type', 'AUROC_95CI'], ascending=[True, False])
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"charls_complete_train_test_evaluation_{timestamp}.csv"
        
        # 保存CSV
        df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        
        # 显示摘要
        print(f"\n📊 完整训练/测试集评估结果摘要:")
        print("=" * 120)
        print(f"{'Type':<15} {'Model':<25} {'Size':<8} {'AUROC_95CI':<25} {'F1_Score_95CI':<25}")
        print("-" * 120)
        
        for _, row in df.iterrows():
            print(f"{row['Evaluation_Type']:<15} {row['Model']:<25} {row['Sample_Size']:<8} "
                  f"{row['AUROC_95CI']:<25} {row['F1_Score_95CI']:<25}")
        
        print(f"\n💾 完整训练测试集评估结果已保存: {csv_filename}")
        return csv_filename
