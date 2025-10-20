"""
Feature selection module for CESD Depression Prediction Model
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    SelectKBest, f_classif, chi2, mutual_info_classif,
    RFE, RFECV, SelectFromModel, VarianceThreshold
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
import warnings
warnings.filterwarnings('ignore')

class FeatureSelector:
    """特征选择器"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.selected_features = {}
        self.feature_scores = {}
        self.selection_methods = {}
        
    def fit(self, X, y, methods=['variance', 'univariate', 'rfe', 'model_based'], 
            k_best=None, variance_threshold=0.01):
        """
        拟合特征选择器
        
        Parameters:
        -----------
        X : DataFrame
            特征数据
        y : Series
            目标变量
        methods : list
            特征选择方法列表
        k_best : int, 可选
            选择的特征数量。如果为None，默认使用特征总数的30%
        variance_threshold : float
            方差阈值
        """
        print(f"\n{'='*60}")
        print("🔍 开始特征选择")
        print(f"{'='*60}")
        
        print(f"原始特征数量: {X.shape[1]}")
        print(f"样本数量: {X.shape[0]}")
        print(f"选择方法: {', '.join(methods)}")
        
        # 如果k_best未指定，默认使用特征总数的60%
        if k_best is None:
            k_best = max(5, int(X.shape[1] * 0.6))
            print(f"未指定特征数量，使用默认值: {k_best} (特征总数的60%)")
        
        self.all_features = X.columns.tolist()
        
        # 1. 方差过滤
        if 'variance' in methods:
            self._variance_filter(X, y, variance_threshold)
        
        # 2. 单变量统计检验
        if 'univariate' in methods or 'statistical' in methods:
            self._univariate_selection(X, y, k_best)
        
        # 3. 递归特征消除
        if 'rfe' in methods:
            self._recursive_feature_elimination(X, y, k_best)
        
        # 4. 基于模型的特征选择
        if 'model_based' in methods or 'model' in methods:
            self._model_based_selection(X, y, k_best)
        
        # 5. 综合选择
        self._ensemble_selection(k_best)
        
        print(f"\n✅ 特征选择完成")
        return self
        
    def transform(self, X):
        """应用特征选择"""
        if not hasattr(self, 'final_features'):
            raise ValueError("请先调用fit方法")
            
        transformed_data = X[self.final_features]
        return {
            'transformed_data': transformed_data,
            'selected_features': self.final_features,
            'feature_scores': getattr(self, 'feature_scores', {})
        }
    
    def fit_transform(self, X, y, **kwargs):
        """拟合并转换"""
        return self.fit(X, y, **kwargs).transform(X)
    
    def _variance_filter(self, X, y, threshold):
        """方差过滤"""
        print(f"\n📊 方差过滤 (阈值: {threshold})")
        
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X)
        
        selected_features = X.columns[selector.get_support()].tolist()
        removed_count = len(X.columns) - len(selected_features)
        
        self.selected_features['variance'] = selected_features
        self.feature_scores['variance'] = selector.variances_
        
        print(f"  移除低方差特征: {removed_count} 个")
        print(f"  保留特征: {len(selected_features)} 个")
        
    def _univariate_selection(self, X, y, k):
        """单变量统计检验"""
        print(f"\n📈 单变量统计检验 (选择前 {k} 个)")
        
        # 分别处理数值型和分类型特征
        numeric_features = X.select_dtypes(include=[np.number]).columns
        categorical_features = X.select_dtypes(exclude=[np.number]).columns
        
        selected_features = []
        scores_dict = {}
        
        # 数值型特征使用F检验
        if len(numeric_features) > 0:
            selector_f = SelectKBest(score_func=f_classif, k='all')
            selector_f.fit(X[numeric_features], y)
            
            f_scores = selector_f.scores_
            f_features = numeric_features[np.argsort(f_scores)[-min(k//2, len(numeric_features)):]]
            selected_features.extend(f_features.tolist())
            
            for i, feature in enumerate(numeric_features):
                scores_dict[feature] = f_scores[i]
            
            print(f"  F检验选择数值型特征: {len(f_features)} 个")
        
        # 分类型特征使用互信息
        if len(categorical_features) > 0:
            selector_mi = SelectKBest(score_func=mutual_info_classif, k='all')
            selector_mi.fit(X[categorical_features], y)
            
            mi_scores = selector_mi.scores_
            mi_features = categorical_features[np.argsort(mi_scores)[-min(k//2, len(categorical_features)):]]
            selected_features.extend(mi_features.tolist())
            
            for i, feature in enumerate(categorical_features):
                scores_dict[feature] = mi_scores[i]
            
            print(f"  互信息选择分类型特征: {len(mi_features)} 个")
        
        # 如果总数超过k，按分数排序选择前k个
        if len(selected_features) > k:
            feature_scores = [(f, scores_dict[f]) for f in selected_features]
            feature_scores.sort(key=lambda x: x[1], reverse=True)
            selected_features = [f[0] for f in feature_scores[:k]]
        
        self.selected_features['univariate'] = selected_features
        self.feature_scores['univariate'] = scores_dict
        
        print(f"  最终选择: {len(selected_features)} 个特征")
        
    def _recursive_feature_elimination(self, X, y, k):
        """递归特征消除"""
        print(f"\n🔄 递归特征消除 (目标: {k} 个特征)")
        
        # 使用逻辑回归作为基础估计器
        estimator = LogisticRegression(random_state=self.random_state, max_iter=2000, solver='liblinear')
        
        # 使用交叉验证的RFE
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        selector = RFECV(
            estimator, 
            step=1, 
            cv=cv, 
            scoring='roc_auc',
            min_features_to_select=min(k, X.shape[1]//2)
        )
        
        try:
            selector.fit(X, y)
            selected_features = X.columns[selector.support_].tolist()
            
            # 如果选择的特征太多，选择排名最高的k个
            if len(selected_features) > k:
                rankings = selector.ranking_
                top_indices = np.where(selector.support_)[0]
                top_rankings = rankings[top_indices]
                sorted_indices = np.argsort(top_rankings)[:k]
                selected_features = [selected_features[i] for i in sorted_indices]
            
            self.selected_features['rfe'] = selected_features
            self.feature_scores['rfe'] = selector.ranking_
            
            print(f"  选择特征: {len(selected_features)} 个")
            print(f"  最佳特征数量: {selector.n_features_}")
            
        except Exception as e:
            print(f"  ⚠️ RFE失败: {e}")
            # 备选方案：使用简单RFE
            simple_rfe = RFE(estimator, n_features_to_select=k)
            simple_rfe.fit(X, y)
            selected_features = X.columns[simple_rfe.support_].tolist()
            
            self.selected_features['rfe'] = selected_features
            self.feature_scores['rfe'] = simple_rfe.ranking_
            print(f"  使用简单RFE，选择特征: {len(selected_features)} 个")
    
    def _model_based_selection(self, X, y, k):
        """基于模型的特征选择"""
        print(f"\n🌳 基于模型的特征选择 (目标: {k} 个特征)")
        
        # 使用随机森林
        rf = RandomForestClassifier(
            n_estimators=100, 
            random_state=self.random_state,
            n_jobs=-1
        )
        rf.fit(X, y)
        
        # 基于特征重要性选择
        selector = SelectFromModel(rf, max_features=k, prefit=True)
        selected_features = X.columns[selector.get_support()].tolist()
        
        # 如果选择的特征不足k个，补充重要性最高的特征
        if len(selected_features) < k:
            importances = rf.feature_importances_
            remaining_features = [f for f in X.columns if f not in selected_features]
            remaining_importances = [importances[X.columns.get_loc(f)] for f in remaining_features]
            
            # 按重要性排序
            sorted_remaining = sorted(zip(remaining_features, remaining_importances), 
                                    key=lambda x: x[1], reverse=True)
            
            # 补充特征直到达到k个
            need_count = k - len(selected_features)
            additional_features = [f[0] for f in sorted_remaining[:need_count]]
            selected_features.extend(additional_features)
        
        self.selected_features['model_based'] = selected_features
        self.feature_scores['model_based'] = rf.feature_importances_
        
        print(f"  选择特征: {len(selected_features)} 个")
        print(f"  平均特征重要性: {np.mean(rf.feature_importances_):.4f}")
    
    def _ensemble_selection(self, k):
        """集成特征选择"""
        print(f"\n🎯 集成特征选择")
        
        # 统计每个特征被选中的次数
        feature_votes = {}
        for feature in self.all_features:
            votes = 0
            for method, features in self.selected_features.items():
                if feature in features:
                    votes += 1
            feature_votes[feature] = votes
        
        # 按投票数排序
        sorted_features = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)
        
        # 选择获得最多投票的前k个特征
        self.final_features = [f[0] for f in sorted_features[:k]]
        self.feature_votes = feature_votes
        
        print(f"  最终选择特征: {len(self.final_features)} 个")
        print(f"  平均投票数: {np.mean([v for v in feature_votes.values()]):.2f}")
        
        # 显示投票结果
        print(f"\n📊 特征投票结果 (前10个):")
        for i, (feature, votes) in enumerate(sorted_features[:10]):
            status = "✓" if feature in self.final_features else "✗"
            print(f"  {i+1:2d}. {status} {feature}: {votes} 票")
    
    def get_feature_importance_summary(self):
        """获取特征重要性总结"""
        if not hasattr(self, 'final_features'):
            return None
            
        summary = []
        for feature in self.final_features:
            votes = self.feature_votes.get(feature, 0)
            
            # 收集各方法的分数
            scores = {}
            for method, score_dict in self.feature_scores.items():
                if isinstance(score_dict, dict):
                    scores[method] = score_dict.get(feature, 0)
                elif hasattr(score_dict, '__getitem__'):
                    try:
                        idx = self.all_features.index(feature)
                        scores[method] = score_dict[idx]
                    except:
                        scores[method] = 0
            
            summary.append({
                'feature': feature,
                'votes': votes,
                **scores
            })
        
        return pd.DataFrame(summary)
    
    def save_results(self, filepath='feature_selection_results.csv'):
        """保存特征选择结果"""
        summary = self.get_feature_importance_summary()
        if summary is not None:
            summary.to_csv(filepath, index=False, encoding='utf-8-sig')
            print(f"✅ 特征选择结果已保存: {filepath}")
        
        # 保存详细结果
        detailed_results = {
            'final_features': self.final_features,
            'feature_votes': self.feature_votes,
            'selected_by_method': self.selected_features
        }
        
        import json
        with open(filepath.replace('.csv', '_detailed.json'), 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
        return summary 

    def optimize_feature_count(self, X, y, max_features=None, step_size=1, cv_folds=10):
        """
        通过交叉验证自动优化特征数量，测试从1个特征到全部特征的各种组合。
        
        参数:
            X (DataFrame): 特征矩阵
            y (Series): 目标变量
            max_features (int, optional): 最大特征数量。默认为特征总数
            step_size (int, optional): 特征数量的递增步长。默认为1
            cv_folds (int, optional): 交叉验证折数。默认为5
            
        返回:
            dict: 包含最优特征数量和详细优化结果的字典
        """
        if max_features is None:
            max_features = X.shape[1]  # 使用全部特征
        
        # 使用RandomForestClassifier作为评估基准模型
        base_model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # 获取特征重要性排序
        print("\n🔍 获取特征重要性排序...")
        temp_model = RandomForestClassifier(n_estimators=100, random_state=42)
        temp_model.fit(X, y)
        
        # 按重要性排序特征
        feature_importances = temp_model.feature_importances_
        feature_names = X.columns.tolist()
        sorted_indices = feature_importances.argsort()[::-1]  # 从高到低排序
        ranked_features = [feature_names[i] for i in sorted_indices]
        
        # 准备存储不同特征数量的性能
        performance = {}
        feature_counts = list(range(1, max_features + 1, step_size))
        
        print(f"\n📊 测试从1到{max_features}个特征(步长={step_size})的性能...")
        
        # 测试不同特征数量的性能
        for k in feature_counts:
            selected_features = ranked_features[:k]
            X_selected = X[selected_features]
            
            # 使用分层交叉验证评估当前特征数量的性能
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_scores = cross_val_score(base_model, X_selected, y, cv=cv, scoring='roc_auc')
            
            performance[k] = {
                'mean_auc': cv_scores.mean(),
                'std_auc': cv_scores.std(),
                'features': selected_features
            }
            
            print(f"  特征数量: {k:2d} → AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # 找出AUC最高的特征数量
        optimal_count = max(performance.keys(), key=lambda k: performance[k]['mean_auc'])
        optimal_auc = performance[optimal_count]['mean_auc']
        optimal_std = performance[optimal_count]['std_auc']
        
        print(f"\n✨ 最佳特征数量: {optimal_count} (AUC: {optimal_auc:.4f} ± {optimal_std:.4f})")
        print(f"✅ 最佳特征集: {performance[optimal_count]['features']}")
        
        # 绘制性能曲线并在最佳点标星
        self._plot_feature_count_performance(performance, optimal_count)
        
        # 保存详细优化结果
        optimization_results = {
            'optimal_count': optimal_count,
            'optimal_auc': float(optimal_auc),
            'optimal_std': float(optimal_std),
            'optimal_features': performance[optimal_count]['features'],
            'all_results': {k: {'mean_auc': float(v['mean_auc']), 'std_auc': float(v['std_auc'])} 
                           for k, v in performance.items()}
        }
        
        # 将最佳特征保存到selected_features_
        self.selected_features_ = {'model': performance[optimal_count]['features']}
        
        return optimization_results

    def _plot_feature_count_performance(self, performance, optimal_count):
        """
        绘制特征数量与性能关系图，并在最佳特征点标记星号。
        
        参数:
            performance (dict): 包含不同特征数量性能的字典
            optimal_count (int): 最佳特征数量
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            from datetime import datetime
            
            # 提取数据用于绘图
            feature_counts = sorted(performance.keys())
            mean_aucs = [performance[k]['mean_auc'] for k in feature_counts]
            std_aucs = [performance[k]['std_auc'] for k in feature_counts]
            
            # 创建图表
            plt.figure(figsize=(10, 6))
            
            # 绘制AUC曲线
            plt.plot(feature_counts, mean_aucs, 'b-', marker='o', markersize=4, label='平均AUC')
            
            # 添加误差带
            plt.fill_between(
                feature_counts, 
                [m-s for m,s in zip(mean_aucs, std_aucs)],
                [m+s for m,s in zip(mean_aucs, std_aucs)],
                color='b', alpha=0.1, label='标准差'
            )
            
            # 标记最佳点(星星)
            plt.plot(optimal_count, performance[optimal_count]['mean_auc'], 'r*', markersize=15, 
                    label=f'最佳特征数: {optimal_count}')
            
            # 添加标题和标签
            plt.title('特征数量优化: AUC性能曲线', fontsize=14)
            plt.xlabel('特征数量', fontsize=12)
            plt.ylabel('平均AUC (5折交叉验证)', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(loc='lower right')
            
            # 设置y轴从0.5开始
            y_min = min(0.5, min(m-s for m,s in zip(mean_aucs, std_aucs))-0.02)
            y_max = max(m+s for m,s in zip(mean_aucs, std_aucs))+0.02
            plt.ylim([y_min, y_max])
            
            # 保存图表
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"feature_optimization_{timestamp}.png"
            plt.tight_layout()
            plt.savefig(save_path)
            print(f"\n📈 性能曲线已保存为: {save_path}")
            
            # 关闭图表避免内存泄漏
            plt.close()
            
        except Exception as e:
            print(f"\n⚠️ 无法生成性能曲线图: {str(e)}") 