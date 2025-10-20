"""
Model builder module for CESD Depression Prediction Model
"""

import numpy as np
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                            VotingClassifier, StackingClassifier, 
                            ExtraTreesClassifier, AdaBoostClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import xgboost as xgb
import lightgbm as lgb
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️ CatBoost未安装，将跳过CatBoost模型")
from ..config import MODEL_PARAMS, CV_SETTINGS, N_JOBS

class ModelBuilder:
    """模型构建类"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.best_params = {}
        self.data_processor = None
        
    def set_data_processor(self, data_processor):
        """设置数据处理器"""
        self.data_processor = data_processor
        
    def build_base_models_with_preprocessing(self, train_data, use_pre_encoded_data=False):
        """
        构建基础模型（支持模型特定预处理）
        
        参数:
        ----
        train_data : DataFrame
            包含目标变量的训练数据
        use_pre_encoded_data : bool
            数据是否已经编码
        """
        print(f"\n🔧 构建基础模型（支持模型特定预处理）...")
        
        # 分离特征和目标变量
        target_col = 'depressed'
        if target_col not in train_data.columns:
            raise ValueError(f"目标变量 '{target_col}' 不在数据中")
        
        feature_cols = [col for col in train_data.columns if col != target_col]
        X = train_data[feature_cols].copy()
        y = train_data[target_col].copy()
        
        print(f"📊 特征数量: {len(feature_cols)}")
        print(f"📊 样本数量: {len(X)}")
        
        models = {}
        self.model_preprocessed_data = {}  # 存储模型特定的预处理信息
        
        if not use_pre_encoded_data:
            # 原有的预处理逻辑
            return self.build_base_models(), {}
        else:
            # 使用已编码的数据，但需要处理自适应SVM
            print(f"✅ 使用已编码数据构建模型...")
            
            # 1. 训练树模型（不需要额外预处理）
            tree_models = self._get_tree_models()
            
            print(f"\n🌳 训练树模型...")
            for name, model in tree_models.items():
                try:
                    print(f"   训练 {name}...")
                    model.fit(X, y)
                    models[name] = model
                    
                    # 记录预处理信息
                    self.model_preprocessed_data[name] = {
                        'model_type': 'tree',
                        'preprocessing': 'none'
                    }
                    
                    print(f"   ✓ {name} 训练完成")
                except Exception as e:
                    print(f"   ✗ {name} 训练失败: {e}")
            
            # 2. 训练线性模型（需要额外预处理）
            linear_models = self._get_linear_models()
            
            print(f"\n📈 训练线性模型 (需要标准化)...")
            try:
                # 对于线性模型，只进行标准化，不重新编码
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                for name, model_config in linear_models.items():
                    try:
                        print(f"   训练 {name}...")
                        
                        # 处理自适应SVM
                        if name == 'svc' and model_config == 'adaptive_svc':
                            model = self._create_adaptive_svm(X_scaled, y)
                        else:
                            model = model_config
                        
                        model.fit(X_scaled, y)
                        models[name] = model
                        
                        # 记录预处理信息
                        self.model_preprocessed_data[name] = {
                            'model_type': 'linear',
                            'preprocessing': 'standardization',
                            'scaler': scaler
                        }
                        
                        print(f"   ✓ {name} 训练完成")
                    except Exception as e:
                        print(f"   ✗ {name} 训练失败: {e}")
            except Exception as e:
                print(f"   ⚠️ 线性模型标准化失败: {e}")
            
            # 3. 跳过集成模型训练（用户要求禁用）
            print(f"\n🚫 跳过集成模型训练（用户要求禁用所有集成模型）")
            print(f"   ✅ 基础模型训练完成，共 {len(models)} 个模型")
            print(f"   📈 基础模型已提供优秀性能，无需集成模型")
            
        # 返回模型和预处理信息
        return models, self.model_preprocessed_data
    
    def _get_tree_models(self):
        """获取树模型字典"""
        from ..config import N_JOBS
        
        tree_models = {
            'rf': RandomForestClassifier(random_state=self.random_state, n_jobs=N_JOBS),
            'gb': GradientBoostingClassifier(random_state=self.random_state),
            'xgb': xgb.XGBClassifier(random_state=self.random_state, n_jobs=N_JOBS, eval_metric='logloss'),
            'lgb': lgb.LGBMClassifier(random_state=self.random_state, n_jobs=N_JOBS, verbose=-1),
            'extra_trees': ExtraTreesClassifier(random_state=self.random_state, n_jobs=N_JOBS),
            'adaboost': AdaBoostClassifier(random_state=self.random_state)
        }
        
        # 检查CatBoost是否可用
        try:
            from catboost import CatBoostClassifier
            tree_models['catboost'] = CatBoostClassifier(
                random_state=self.random_state,
                verbose=False,
                allow_writing_files=False
            )
        except ImportError:
            print("⚠️ CatBoost不可用，跳过")
        
        return tree_models
    
    def _get_linear_models(self):
        """获取线性模型字典"""
        from ..config import N_JOBS
        
        return {
            # 🔧 优化逻辑回归：增加迭代次数，修改求解器，调整容忍度
            'lr': LogisticRegression(
                random_state=self.random_state, 
                max_iter=5000,  # 增加迭代次数
                solver='liblinear',  # 更换求解器，通常收敛更快
                tol=1e-3,  # 放宽容忍度
                n_jobs=N_JOBS
            ),
            # 🚫 用户要求：完全禁用SVC模型（避免长时间训练）
            # 'svc': CalibratedClassifierCV(
            #     LinearSVC(random_state=self.random_state, max_iter=5000, dual=False),
            #     method='sigmoid',  # sigmoid比isotonic更快
            #     cv=3,              # 3折足够，比5折快
            #     n_jobs=1           # 避免嵌套并行
            # )
        }
    
    def _get_optimal_svm(self):
        """
        智能选择最优的SVM实现
        
        根据数据规模和计算资源选择：
        - 小数据集 (< 3000样本): 使用SVC with RBF核
        - 中等数据集 (3000-8000样本): 使用SVC with 线性核
        - 大数据集 (> 8000样本): 使用SVC with 线性核（但增加C参数限制）
        """
        # 这里我们返回一个函数，在实际使用时根据数据规模决定
        return 'adaptive_svc'  # 标记为自适应SVC
    
    def _create_adaptive_svm(self, X_train, y_train):
        """
        根据数据规模创建合适的SVC模型
        """
        sample_count = len(X_train)
        
        if sample_count < 3000:
            # 小数据集：使用RBF核SVC，支持概率预测
            print(f"   数据集较小({sample_count}样本)，使用RBF核SVC")
            return SVC(
                kernel='rbf', 
                probability=True, 
                random_state=self.random_state,
                gamma='scale',
                C=1.0
            )
        elif sample_count < 8000:
            # 中等数据集：使用线性核SVC，支持概率预测
            print(f"   数据集中等({sample_count}样本)，使用线性核SVC")
            return SVC(
                kernel='linear', 
                probability=True, 
                random_state=self.random_state,
                C=1.0
            )
        else:
            # 大数据集：使用线性核SVC，限制C参数以提高速度
            print(f"   数据集较大({sample_count}样本)，使用线性核SVC（优化参数）")
            return SVC(
                kernel='linear', 
                probability=True, 
                random_state=self.random_state,
                C=0.1  # 较小的C值以提高训练速度
            )
        
    def build_base_models(self):
        """构建基础模型（向后兼容）"""
        base_models = {
            'rf': RandomForestClassifier(random_state=self.random_state, n_jobs=N_JOBS),
            'gb': GradientBoostingClassifier(random_state=self.random_state),
            'xgb': xgb.XGBClassifier(random_state=self.random_state, eval_metric='logloss', n_jobs=N_JOBS),
            'lgb': lgb.LGBMClassifier(random_state=self.random_state, verbosity=-1, n_jobs=N_JOBS),
            'lr': LogisticRegression(
                random_state=self.random_state, 
                max_iter=5000, 
                solver='liblinear', 
                tol=1e-3, 
                n_jobs=N_JOBS
            ),
            # 🚫 用户要求：完全禁用SVC模型（避免长时间训练）
            # 'svc': SVC(kernel='linear', probability=True, random_state=self.random_state),
            # 🆕 新增模型
            'extra_trees': ExtraTreesClassifier(random_state=self.random_state, n_jobs=N_JOBS),
            'adaboost': AdaBoostClassifier(random_state=self.random_state),
        }
        
        # 🆕 添加CatBoost (如果可用)
        if CATBOOST_AVAILABLE:
            base_models['catboost'] = CatBoostClassifier(
                random_state=self.random_state, 
                verbose=False,
                iterations=100
            )
        
        return base_models
        
    def build_ensemble_models(self, base_models, X=None, y=None):
        """
        构建集成模型（仅保留软投票和加权投票分类器）
        
        参数:
        ----
        base_models : dict
            基础模型字典，键为模型名称，值为模型对象
        X : array-like, 可选
            训练数据，用于某些需要训练的集成模型
        y : array-like, 可选
            目标变量，用于某些需要训练的集成模型
            
        返回:
        ----
        dict : 集成模型字典
        """
        print("🔧 构建精选集成模型：仅保留加权投票分类器")
        print("   跳过软投票、硬投票和堆叠分类器")
        
        from sklearn.ensemble import VotingClassifier
        ensemble_models = {}
        
        # 跳过软投票分类器 - 用户要求禁用
        print("\n🚫 跳过软投票分类器（用户要求禁用）")
        
        try:
            # 1. 加权投票分类器 (Weighted Voting) - 唯一保留的集成模型
            print("\n⚖️ 创建加权投票分类器...")
            
            # 筛选支持概率预测的模型
            proba_models = {}
            for name, model in base_models.items():
                if hasattr(model, 'predict_proba'):
                    proba_models[name] = model
            
            if len(proba_models) >= 2:
                estimators = [(name, model) for name, model in proba_models.items()]
                
                # 根据模型类型设置权重 (基于您数据的性能表现)
                model_weights = {
                    'adaboost': 1.2,    # AdaBoost表现最好 (AUROC 0.7529)
                    'gb': 1.1,          # GradientBoosting第二 (AUROC 0.7520)
                    'lgb': 1.1,         # LightGBM第三 (AUROC 0.7494)
                    'catboost': 1.1,    # CatBoost第四 (AUROC 0.7474)
                    'lr': 1.0,          # 逻辑回归第五 (AUROC 0.7473)
                    'xgb': 1.0,         # XGBoost
                    'rf': 1.0,          # RandomForest
                    'extra_trees': 0.9, # ExtraTrees
                    'svc': 0.9          # SVC
                }
                
                # 构建权重列表
                weights = []
                for name, _ in estimators:
                    weight = model_weights.get(name, 1.0)
                    weights.append(weight)
                
                weighted_voting = VotingClassifier(
                    estimators=estimators, 
                    voting='soft',
                    weights=weights
                )
                
                # 如果提供了训练数据，则训练分类器
                if X is not None and y is not None:
                    weighted_voting.fit(X, y)
                    
                ensemble_models['weighted_voting'] = weighted_voting
                
                print(f"✅ 加权投票分类器创建成功 (使用{len(proba_models)}个模型)")
                print("   权重分配:")
                for (name, _), weight in zip(estimators, weights):
                    print(f"     • {name}: {weight}")
            else:
                print("⚠️ 没有足够支持概率预测的模型，跳过加权投票分类器")
                
        except Exception as e:
            print(f"❌ 创建加权投票分类器失败: {str(e)}")
        
        print(f"\n📈 精选集成模型构建完成，共创建 {len(ensemble_models)} 个集成模型")
        print("   预期训练时间: 约20-25分钟 (vs 原版149小时)")
        print("   预期性能提升: 1-3% AUROC提升")
        
        return ensemble_models
        
        # 以下是被跳过的耗时代码（硬投票和堆叠分类器）
        try:
            
            # 确保有足够的基础模型用于集成
            if len(base_models) < 3:
                print("⚠️ 需要至少3个基础模型进行集成")
                return {}
            
            ensemble_models = {}
            
            # 1. 硬投票分类器
            try:
                # 🔧 修复：只使用支持概率预测的模型
                proba_models = {}
                for name, model in base_models.items():
                    if hasattr(model, 'predict_proba'):
                        proba_models[name] = model
                
                if len(proba_models) >= 2:
                    estimators = [(name, model) for name, model in proba_models.items()]
                    voting_hard = VotingClassifier(estimators=estimators, voting='hard')
                    
                    # 如果提供了训练数据，则训练投票分类器
                    if X is not None and y is not None:
                        voting_hard.fit(X, y)
                        
                    ensemble_models['voting_hard'] = voting_hard
                    print("✅ 硬投票分类器创建成功")
                else:
                    print("⚠️ 没有足够支持概率预测的模型，跳过硬投票分类器")
            except Exception as e:
                print(f"⚠️ 创建硬投票分类器失败: {str(e)}")
            
            # 2. 软投票分类器
            try:
                # 🔧 修复：只使用支持概率预测的模型
                proba_models = {}
                for name, model in base_models.items():
                    if hasattr(model, 'predict_proba'):
                        proba_models[name] = model
                
                if len(proba_models) >= 2:
                    estimators = [(name, model) for name, model in proba_models.items()]
                    voting_soft = VotingClassifier(estimators=estimators, voting='soft')
                    
                    # 如果提供了训练数据，则训练投票分类器
                    if X is not None and y is not None:
                        voting_soft.fit(X, y)
                        
                    ensemble_models['voting_soft'] = voting_soft
                    print("✅ 软投票分类器创建成功")
                else:
                    print("⚠️ 没有足够支持概率预测的模型，跳过软投票分类器")
            except Exception as e:
                print(f"⚠️ 创建软投票分类器失败: {str(e)}")
            
            # 3. 堆叠分类器 (stacking)
            try:
                # 使用逻辑回归作为最终分类器
                from sklearn.linear_model import LogisticRegression
                
                # 🔧 修复：只使用支持概率预测的模型
                proba_models = {}
                for name, model in base_models.items():
                    if hasattr(model, 'predict_proba'):
                        proba_models[name] = model
                
                if len(proba_models) >= 2:
                    estimators = [(name, model) for name, model in proba_models.items()]
                    stacking_clf = StackingClassifier(
                        estimators=estimators,
                        final_estimator=LogisticRegression(
                    random_state=self.random_state, 
                    max_iter=5000, 
                    solver='liblinear', 
                    tol=1e-3, 
                    n_jobs=N_JOBS
                ),
                        cv=CV_SETTINGS['inner_splits']
                    )
                    
                    # 如果提供了训练数据，则训练堆叠分类器
                    if X is not None and y is not None:
                        stacking_clf.fit(X, y)
                        
                    ensemble_models['stacking'] = stacking_clf
                    print("✅ 堆叠分类器创建成功")
                else:
                    print("⚠️ 没有足够支持概率预测的模型，跳过堆叠分类器")
            except Exception as e:
                print(f"⚠️ 创建堆叠分类器失败: {str(e)}")
            
            # 4. 🆕 加权投票分类器
            try:
                # 选择表现最好的几个模型进行加权投票
                best_models = ['lgb', 'xgb', 'catboost', 'rf'] if 'catboost' in base_models else ['lgb', 'xgb', 'rf', 'gb']
                available_models = {k: v for k, v in base_models.items() if k in best_models and hasattr(v, 'predict_proba')}
                
                if len(available_models) >= 3:
                    estimators = [(name, model) for name, model in available_models.items()]
                    weighted_voting = VotingClassifier(
                        estimators=estimators, 
                        voting='soft',
                        weights=[1.2, 1.1, 1.0, 0.9][:len(estimators)]  # 给更好的模型更高权重
                    )
                    
                    if X is not None and y is not None:
                        weighted_voting.fit(X, y)
                        
                    ensemble_models['weighted_voting'] = weighted_voting
                    print("✅ 加权投票分类器创建成功")
                else:
                    print("⚠️ 没有足够支持概率预测的模型，跳过加权投票分类器")
            except Exception as e:
                print(f"⚠️ 创建加权投票分类器失败: {str(e)}")
            
            print(f"🎯 成功创建 {len(ensemble_models)} 个集成模型")
            return ensemble_models
            
        except Exception as e:
            print(f"❌ 构建集成模型失败: {str(e)}")
            return {}
        
    def get_models(self):
        """获取已训练的模型"""
        return self.models
        
    def tune_hyperparameters(self, X_train, y_train, model_type='random_forest', search_method='random'):
        """超参数调优"""
        print(f"\n{'-'*40}")
        print(f"超参数调优: {model_type}")
        print(f"{'-'*40}")
        
        # 获取模型和参数网格
        model = self.build_base_models()[model_type]
        param_grid = MODEL_PARAMS.get(model_type, {})
        
        if not param_grid:
            print(f"✗ 未找到 {model_type} 的参数网格")
            return model
            
        # 选择搜索方法
        if search_method == 'grid':
            search = GridSearchCV(
                model,
                param_grid,
                cv=CV_SETTINGS['inner_splits'],
                scoring='roc_auc',
                n_jobs=N_JOBS,
                verbose=1
            )
        else:
            search = RandomizedSearchCV(
                model,
                param_grid,
                n_iter=20,
                cv=CV_SETTINGS['inner_splits'],
                scoring='roc_auc',
                random_state=self.random_state,
                n_jobs=N_JOBS,
                verbose=1
            )
            
        # 执行搜索
        try:
            search.fit(X_train, y_train)
            self.best_params[model_type] = search.best_params_
            
            print(f"\n最佳参数:")
            for param, value in search.best_params_.items():
                print(f"  {param}: {value}")
            print(f"最佳得分: {search.best_score_:.4f}")
            
            return search.best_estimator_
            
        except Exception as e:
            print(f"✗ 超参数调优失败: {e}")
            return model
            
    def save_hyperparameter_results(self, search_results, model_type):
        """保存超参数调优结果"""
        self.best_params[model_type] = {
            'best_params': search_results.best_params_,
            'best_score': search_results.best_score_,
            'cv_results': search_results.cv_results_
        } 
 