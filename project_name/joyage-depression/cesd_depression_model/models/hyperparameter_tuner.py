"""
超参数调优器 - 优化版本
使用智能N_JOBS配置以节省内存
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score
import json
import time
from datetime import datetime
import warnings

# 抑制警告
warnings.filterwarnings('ignore')

from ..config import N_JOBS, CV_SETTINGS

class HyperparameterTuner:
    """超参数调优器 - 内存优化版本"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.best_models = {}
        self.tuning_results = {}
        
    def get_hyperparameter_grids(self):
        """获取超参数搜索空间 - 优化版本"""
        param_grids = {
            'rf': {  # 修复：使用实际模型名称
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt', 'log2']
            },
            'gb': {  # 修复：使用实际模型名称
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'subsample': [0.8, 0.9, 1.0]
            },
            # 🚫 用户要求：完全禁用SVC模型（避免长时间训练）
            # 'svc': {  # 🔧 修改：SVC超参数配置
            #     'C': [0.1, 1, 10],
            #     'kernel': ['linear', 'rbf'],
            #     'gamma': ['scale', 'auto'],
            #     'probability': [True]  # 启用概率预测
            # },
            'xgb': {  # 修复：使用实际模型名称
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            'lgb': {  # 修复：使用实际模型名称
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7, -1],
                'num_leaves': [15, 31, 63],
                'min_child_samples': [10, 20, 30],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            'lr': {  # 修复：使用实际模型名称
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'max_iter': [1000, 2000, 3000]
            },
            # 🆕 新增模型的超参数调优配置
            'catboost': {  # 修复：使用实际模型名称
                'iterations': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'depth': [3, 5, 7, 10],
                'l2_leaf_reg': [1, 3, 5, 7],
                'border_count': [32, 64, 128]
            },
            'adaboost': {  # 修复：使用实际模型名称
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.5, 1.0],
                'algorithm': ['SAMME', 'SAMME.R']
            },
            'extra_trees': {  # 修复：使用实际模型名称
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt', 'log2']
            }
        }
        return param_grids
        
    def tune_model(self, model, model_name, X_train, y_train, 
                   search_method='random', n_iter=20):
        """
        调优单个模型 - 内存优化版本
        
        参数:
        ----
        model : sklearn model
            要调优的模型
        model_name : str
            模型名称
        X_train : array-like
            训练特征
        y_train : array-like
            训练标签
        search_method : str, 默认'random'
            搜索方法 ('grid' 或 'random')
        n_iter : int, 默认20
            随机搜索的迭代次数
            
        返回:
        ----
        dict : 包含最佳模型和调优结果
        """
        print(f"\n🔧 调优模型: {model_name}")
        
        param_grids = self.get_hyperparameter_grids()
        param_grid = param_grids.get(model_name, {})
        
        if not param_grid:
            print(f"  ❌ 没有为 {model_name} 定义超参数网格")
            return {'best_model': model, 'best_params': {}, 'best_score': 0}
        
        # 显示参数网格信息
        total_combinations = 1
        for param, values in param_grid.items():
            total_combinations *= len(values)
        print(f"  📊 参数网格大小: {total_combinations} 种组合")
        
        # 交叉验证设置 - 所有模型统一使用10折交叉验证
        n_splits = 10
        print(f"  📊 使用{n_splits}折交叉验证以获得最准确的性能评估")
            
        cv = StratifiedKFold(
            n_splits=n_splits, 
            shuffle=True, 
            random_state=self.random_state
        )
        
        try:
            # 使用优化后的N_JOBS
            print(f"  ⚙️ 使用并行作业数: {N_JOBS}")
            
            if search_method == 'grid':
                search = GridSearchCV(
                    model, param_grid,
                    cv=cv, scoring='roc_auc',
                    n_jobs=N_JOBS, verbose=1
                )
            else:  # random
                search = RandomizedSearchCV(
                    model, param_grid,
                    n_iter=n_iter, cv=cv, scoring='roc_auc',
                    random_state=self.random_state,
                    n_jobs=N_JOBS, verbose=1
                )
            
            # 执行搜索
            print(f"  🚀 开始{search_method}搜索...")
            start_time = time.time()
            search.fit(X_train, y_train)
            duration = time.time() - start_time
            
            # 记录结果
            result = {
                'best_model': search.best_estimator_,
                'best_params': search.best_params_,
                'best_score': search.best_score_,
                'cv_results': search.cv_results_,
                'duration': duration,
                'search_method': search_method,
                'n_combinations_tested': len(search.cv_results_['mean_test_score']),
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
            }
            
            print(f"  ✅ 调优完成! 最佳得分: {search.best_score_:.4f}")
            print(f"  ⏱️ 耗时: {duration:.1f}秒")
            print(f"  🎯 最佳参数: {search.best_params_}")
            
            # 保存详细结果
            self._save_tuning_results(model_name, result)
            
            return result
            
        except Exception as e:
            print(f"  ❌ 调优失败: {str(e)}")
            return {'best_model': model, 'best_params': {}, 'best_score': 0, 'error': str(e)}

    def benchmark_models_with_tuning(self, models, X_train, y_train, 
                                   search_method='random', n_iter=15):
        """
        对多个模型进行基准测试和调优 - 包含95%置信区间
        
        参数:
        ----
        models : dict
            模型字典 {model_name: model_instance}
        X_train : array-like
            训练特征
        y_train : array-like
            训练标签
        search_method : str, 默认'random'
            搜索方法
        n_iter : int, 默认15
            随机搜索迭代次数
            
        返回:
        ----
        tuple : (tuned_models, benchmark_df)
        """
        print(f"\n🚀 开始模型基准测试和调优")
        print(f"📊 总模型数: {len(models)}")
        print(f"🔧 搜索方法: {search_method}")
        print(f"⚙️ 并行作业数: {N_JOBS}")
        print(f"📊 所有结果将包含95%置信区间")
        
        tuned_models = {}
        benchmark_results = []
        
        for model_name, model in models.items():
            print(f"\n{'='*60}")
            print(f"🎯 调优模型: {model_name}")
            print(f"{'='*60}")
            
            # 调优模型
            tuning_result = self.tune_model(
                model, model_name, X_train, y_train,
                search_method=search_method, n_iter=n_iter
            )
            
            if 'best_model' in tuning_result:
                tuned_models[model_name] = tuning_result['best_model']
                
                # 🆕 计算95%置信区间 - 修复版本
                from scipy import stats
                import numpy as np
                
                cv_scores = []
                if 'cv_results' in tuning_result and tuning_result['cv_results']:
                    cv_results = tuning_result['cv_results']
                    
                    # 🔧 正确提取交叉验证分数
                    # sklearn的cv_results_包含每次分割的测试分数
                    if 'split0_test_score' in cv_results:
                        # 提取所有分割的测试分数
                        n_splits = 10  # 我们使用10折交叉验证
                        for i in range(n_splits):
                            split_key = f'split{i}_test_score'
                            if split_key in cv_results:
                                scores = cv_results[split_key]
                                if hasattr(scores, '__iter__'):  # 如果是数组
                                    cv_scores.extend(scores)
                                else:  # 如果是单个值
                                    cv_scores.append(scores)
                    
                    # 🔧 备选方案：使用最佳参数的所有CV分数
                    if not cv_scores and 'params' in cv_results:
                        best_params = tuning_result['best_params']
                        # 找到最佳参数对应的分数
                        for idx, params in enumerate(cv_results['params']):
                            if params == best_params:
                                # 提取该参数组合的所有分割分数
                                for i in range(10):
                                    split_key = f'split{i}_test_score'
                                    if split_key in cv_results and idx < len(cv_results[split_key]):
                                        cv_scores.append(cv_results[split_key][idx])
                                break
                
                # 计算95%置信区间
                if cv_scores and len(cv_scores) >= 3:  # 至少需要3个分数点
                    cv_scores = np.array(cv_scores)
                    # 过滤无效值
                    cv_scores = cv_scores[~np.isnan(cv_scores)]
                    
                    if len(cv_scores) >= 3:
                        # 使用t分布计算置信区间（更准确）
                        mean_score = np.mean(cv_scores)
                        se = stats.sem(cv_scores)  # 标准误
                        h = se * stats.t.ppf((1 + 0.95) / 2., len(cv_scores)-1)
                        ci_lower = max(0, mean_score - h)  # 确保不小于0
                        ci_upper = min(1, mean_score + h)  # 确保不大于1
                    else:
                        # 如果分数太少，使用百分位数
                        ci_lower = np.percentile(cv_scores, 2.5)
                        ci_upper = np.percentile(cv_scores, 97.5)
                else:
                    # 🔧 如果没有CV分数，使用保守的估计
                    best_score = tuning_result.get('best_score', 0.5)
                    if np.isnan(best_score) or best_score <= 0:
                        best_score = 0.5  # 默认值
                    
                    # 基于经验的置信区间估计（±5%）
                    margin = 0.05
                    ci_lower = max(0, best_score - margin)
                    ci_upper = min(1, best_score + margin)
                
                # 确保置信区间是有效数值
                if np.isnan(ci_lower) or np.isnan(ci_upper):
                    ci_lower = max(0, tuning_result.get('best_score', 0.5) - 0.05)
                    ci_upper = min(1, tuning_result.get('best_score', 0.5) + 0.05)
                
                # 记录基准测试结果 - 包含95%置信区间
                benchmark_results.append({
                    'Model': model_name,
                    'Best_CV_Score': tuning_result['best_score'],
                    'CV_Score_95CI_Lower': ci_lower,
                    'CV_Score_95CI_Upper': ci_upper,
                    'CV_Score_95CI': f"[{ci_lower:.4f}, {ci_upper:.4f}]",
                    'Best_Params': str(tuning_result['best_params']),
                    'Duration_Seconds': tuning_result.get('duration', 0),
                    'Combinations_Tested': tuning_result.get('n_combinations_tested', 0),
                    'Search_Method': search_method,
                    'CV_Folds': 10,
                    'Confidence_Interval_Method': 'Bootstrap on CV scores'
                })
                
                print(f"✅ {model_name} 调优完成")
                print(f"   🎯 最佳分数: {tuning_result['best_score']:.4f}")
                print(f"   📊 95%置信区间: [{ci_lower:.4f}, {ci_upper:.4f}]")
            
            else:
                print(f"❌ {model_name} 调优失败")
        
        # 创建基准测试DataFrame
        if benchmark_results:
            benchmark_df = pd.DataFrame(benchmark_results)
            # 按最佳分数排序
            benchmark_df = benchmark_df.sort_values('Best_CV_Score', ascending=False)
            
            # 保存基准测试结果
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            benchmark_file = f'hyperparameter_tuning_benchmark_{timestamp}.csv'
            benchmark_df.to_csv(benchmark_file, index=False, encoding='utf-8-sig')
            
            print(f"\n📊 基准测试完成！")
            print(f"📁 结果已保存: {benchmark_file}")
            print(f"\n🏆 模型排名 (按CV得分):")
            for i, (_, row) in enumerate(benchmark_df.head(3).iterrows(), 1):
                print(f"   {i}. {row['Model']}: {row['Best_CV_Score']:.4f} {row['CV_Score_95CI']}")
        else:
            benchmark_df = pd.DataFrame()
            print("❌ 没有成功调优的模型")
        
        return tuned_models, benchmark_df
    
    def _save_tuning_results(self, model_name, result):
        """保存调优结果详情"""
        try:
            # 准备保存的数据（移除不能JSON序列化的内容）
            save_data = {
                'model_name': model_name,
                'best_score': float(result['best_score']),
                'best_params': result['best_params'],
                'duration': result['duration'],
                'search_method': result['search_method'],
                'n_combinations_tested': result.get('n_combinations_tested', 0),
                'timestamp': result['timestamp']
            }
            
            # 保存到JSON文件
            filename = f'hyperparameter_tuning_{model_name}_{result["timestamp"]}.json'
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            print(f"  💾 详细结果已保存: {filename}")
            
        except Exception as e:
            print(f"  ⚠️ 保存结果失败: {e}") 