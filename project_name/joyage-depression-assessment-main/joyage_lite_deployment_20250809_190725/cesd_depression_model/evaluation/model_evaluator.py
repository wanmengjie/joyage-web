"""
Model evaluation module for CESD Depression Prediction Model
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    brier_score_loss, roc_curve, precision_recall_curve
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """模型评估器 - 包含完整的评估指标，固定95%置信区间"""
    
    def __init__(self, random_state=42):
        """
        初始化模型评估器
        
        参数:
        ----
        random_state : int, 默认42
            随机种子
        """
        self.random_state = random_state
        
    def evaluate_model(self, model, X_test, y_test, bootstrap_ci=True, n_bootstraps=1000):
        """
        全面评估模型性能
        
        参数:
        ----
        model : sklearn model
            训练好的模型
        X_test : DataFrame
            测试特征
        y_test : Series
            测试目标
        bootstrap_ci : bool, 默认为 True
            是否计算Bootstrap置信区间
        n_bootstraps : int, 默认为 1000
            Bootstrap采样次数
            
        返回:
        ----
        dict : 包含所有评估指标的字典
        """
        try:
            # 预测
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # 计算混淆矩阵
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            
            # 基本指标
            metrics = {
                # 1. 准确率
                'accuracy': accuracy_score(y_test, y_pred),
                
                # 2. 精确率
                'precision': precision_score(y_test, y_pred, zero_division=0),
                
                # 3. 召回率（敏感性）
                'recall': recall_score(y_test, y_pred, zero_division=0),
                
                # 4. F1分数
                'f1_score': f1_score(y_test, y_pred, zero_division=0),
                
                # 5. AUROC
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                
                # 6. AUPRC
                'pr_auc': average_precision_score(y_test, y_pred_proba),
                
                # 7. C_Index (对于二分类问题，等同于AUROC)
                'c_index': roc_auc_score(y_test, y_pred_proba),
                
                # 8. 特异性
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                
                # 9. 负预测值 (NPV)
                'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,
                
                # 10. Brier分数
                'brier_score': brier_score_loss(y_test, y_pred_proba),
                
                # 额外有用指标
                'positive_predictive_value': precision_score(y_test, y_pred, zero_division=0),  # 与precision相同
                'sensitivity': recall_score(y_test, y_pred, zero_division=0),  # 与recall相同
                'true_positive': int(tp),
                'true_negative': int(tn),
                'false_positive': int(fp),
                'false_negative': int(fn),
                'sample_size': len(y_test)
            }
            
            # 计算置信区间
            if bootstrap_ci:
                ci_metrics = self._calculate_bootstrap_confidence_intervals(
                    model, X_test, y_test, n_bootstraps
                )
                # 合并置信区间结果
                for key, ci in ci_metrics.items():
                    metrics[f'{key}_ci_lower'] = ci[0]
                    metrics[f'{key}_ci_upper'] = ci[1]
            
            return metrics
            
        except Exception as e:
            print(f"❌ 模型评估失败: {str(e)}")
            return {}
    
    def _calculate_bootstrap_confidence_intervals(self, model, X_test, y_test, n_bootstraps=1000):
        """
        计算Bootstrap 95%置信区间
        
        参数:
        ----
        model : sklearn model
            训练好的模型
        X_test : DataFrame
            测试特征
        y_test : Series
            测试目标
        n_bootstraps : int
            Bootstrap采样次数
            
        返回:
        ----
        dict : 各指标的置信区间
        """
        np.random.seed(self.random_state)
        
        bootstrap_metrics = {
            'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [],
            'roc_auc': [], 'pr_auc': [], 'c_index': [], 'specificity': [], 
            'npv': [], 'brier_score': []
        }
        
        n_samples = len(y_test)
        
        for i in range(n_bootstraps):
            # Bootstrap采样
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X_test.iloc[indices]
            y_boot = y_test.iloc[indices]
            
            try:
                # 预测
                y_pred_boot = model.predict(X_boot)
                y_pred_proba_boot = model.predict_proba(X_boot)[:, 1]
                
                # 计算混淆矩阵
                tn, fp, fn, tp = confusion_matrix(y_boot, y_pred_boot).ravel()
                
                # 计算指标
                bootstrap_metrics['accuracy'].append(accuracy_score(y_boot, y_pred_boot))
                bootstrap_metrics['precision'].append(precision_score(y_boot, y_pred_boot, zero_division=0))
                bootstrap_metrics['recall'].append(recall_score(y_boot, y_pred_boot, zero_division=0))
                bootstrap_metrics['f1_score'].append(f1_score(y_boot, y_pred_boot, zero_division=0))
                bootstrap_metrics['roc_auc'].append(roc_auc_score(y_boot, y_pred_proba_boot))
                bootstrap_metrics['pr_auc'].append(average_precision_score(y_boot, y_pred_proba_boot))
                bootstrap_metrics['c_index'].append(roc_auc_score(y_boot, y_pred_proba_boot))
                bootstrap_metrics['specificity'].append(tn / (tn + fp) if (tn + fp) > 0 else 0)
                bootstrap_metrics['npv'].append(tn / (tn + fn) if (tn + fn) > 0 else 0)
                bootstrap_metrics['brier_score'].append(brier_score_loss(y_boot, y_pred_proba_boot))
                
            except Exception:
                # 如果某次采样出现问题，跳过
                continue
        
        # 计算95%置信区间
        confidence_intervals = {}
        for metric, values in bootstrap_metrics.items():
            if values:  # 确保有值
                ci_lower = np.percentile(values, 2.5)
                ci_upper = np.percentile(values, 97.5)
                confidence_intervals[metric] = (ci_lower, ci_upper)
            else:
                confidence_intervals[metric] = (0, 0)
        
        return confidence_intervals
    
    def evaluate_all_models(self, models, X_test, y_test, bootstrap_ci=True):
        """
        评估所有模型
        
        参数:
        ----
        models : dict
            模型字典
        X_test : DataFrame
            测试特征
        y_test : Series
            测试目标
        bootstrap_ci : bool
            是否计算置信区间
            
        返回:
        ----
        dict : 所有模型的评估结果
        """
        all_results = {}
        
        for model_name, model in models.items():
            try:
                print(f"  📊 评估 {model_name}...")
                results = self.evaluate_model(model, X_test, y_test, bootstrap_ci=bootstrap_ci)
                all_results[model_name] = results
                
                # 打印主要指标
                print(f"    ✓ AUROC: {results['roc_auc']:.4f}")
                print(f"    ✓ AUPRC: {results['pr_auc']:.4f}")
                print(f"    ✓ F1: {results['f1_score']:.4f}")
                print(f"    ✓ 准确率: {results['accuracy']:.4f}")
                
                if bootstrap_ci and 'roc_auc_ci_lower' in results:
                    print(f"    ✓ AUROC 95%CI: [{results['roc_auc_ci_lower']:.4f}, {results['roc_auc_ci_upper']:.4f}]")
                
            except Exception as e:
                print(f"    ❌ {model_name} 评估失败: {str(e)}")
                all_results[model_name] = {}
        
        return all_results
    
    def find_best_model(self, results, metric='roc_auc'):
        """
        找到最佳模型
        
        参数:
        ----
        results : dict
            评估结果字典
        metric : str
            用于比较的指标
            
        返回:
        ----
        tuple : (最佳模型名称, 最佳分数)
        """
        if not results:
            return None, None
            
        best_model = None
        best_score = -1
        
        for model_name, metrics in results.items():
            if metric in metrics and metrics[metric] > best_score:
                best_score = metrics[metric]
                best_model = model_name
        
        return best_model, best_score
    
    def generate_comparison_table(self, results, save_path=None):
        """
        生成模型比较表
        
        参数:
        ----
        results : dict
            评估结果字典
        save_path : str, 可选
            保存路径
            
        返回:
        ----
        DataFrame : 比较表
        """
        if not results:
            return pd.DataFrame()
        
        # 提取所有指标
        comparison_data = []
        
        for model_name, metrics in results.items():
            if metrics:  # 确保有评估结果
                row = {'Model': model_name}
                
                # 主要指标
                main_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 
                              'roc_auc', 'pr_auc', 'c_index', 'specificity', 
                              'npv', 'brier_score']
                
                for metric in main_metrics:
                    if metric in metrics:
                        row[metric.upper()] = round(metrics[metric], 4)
                    
                    # 添加置信区间（如果存在）
                    ci_lower_key = f'{metric}_ci_lower'
                    ci_upper_key = f'{metric}_ci_upper'
                    if ci_lower_key in metrics and ci_upper_key in metrics:
                        row[f'{metric.upper()}_95CI'] = f"[{metrics[ci_lower_key]:.4f}, {metrics[ci_upper_key]:.4f}]"
                
                # 混淆矩阵指标
                if 'true_positive' in metrics:
                    row['TP'] = metrics['true_positive']
                    row['TN'] = metrics['true_negative']
                    row['FP'] = metrics['false_positive']
                    row['FN'] = metrics['false_negative']
                
                comparison_data.append(row)
        
        # 创建DataFrame并按AUROC排序
        df = pd.DataFrame(comparison_data)
        if not df.empty and 'ROC_AUC' in df.columns:
            df = df.sort_values('ROC_AUC', ascending=False)
        
        # 保存结果
        if save_path:
            df.to_csv(save_path, index=False)
            print(f"✅ 模型比较表已保存: {save_path}")
        
        return df
    
    def cross_validate_model(self, model, X, y, cv_folds=10, scoring_metrics=None):
        """
        交叉验证评估模型 - 使用Bootstrap计算95%置信区间
        
        参数:
        ----
        model : sklearn model
            要评估的模型
        X : DataFrame
            特征数据
        y : Series
            目标变量
        cv_folds : int
            交叉验证折数
        scoring_metrics : list, 可选
            评估指标列表
            
        返回:
        ----
        dict : 交叉验证结果，包含Bootstrap 95%置信区间
        """
        if scoring_metrics is None:
            scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'average_precision']
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        cv_results = {}
        
        print(f"🔄 执行 {cv_folds} 折交叉验证...")
        
        for metric in scoring_metrics:
            try:
                # 执行交叉验证
                scores = cross_val_score(model, X, y, cv=cv, scoring=metric, n_jobs=1)
                
                # 基本统计
                cv_results[f'{metric}_scores'] = scores.tolist()
                cv_results[f'{metric}_mean'] = scores.mean()
                cv_results[f'{metric}_std'] = scores.std()
                
                # 使用Bootstrap重采样计算真正的95%置信区间
                # 对交叉验证得分进行Bootstrap重采样
                n_bootstrap = 1000
                np.random.seed(self.random_state)
                bootstrap_means = []
                
                for _ in range(n_bootstrap):
                    # 对CV分数进行有放回抽样
                    bootstrap_sample = np.random.choice(scores, size=len(scores), replace=True)
                    bootstrap_means.append(bootstrap_sample.mean())
                
                # 计算95%置信区间
                ci_lower = np.percentile(bootstrap_means, 2.5)
                ci_upper = np.percentile(bootstrap_means, 97.5)
                
                cv_results[f'{metric}_ci_lower'] = ci_lower
                cv_results[f'{metric}_ci_upper'] = ci_upper
                
                print(f"  ✓ {metric}: {scores.mean():.4f} ± {scores.std():.4f} [95%CI: {ci_lower:.4f}, {ci_upper:.4f}]")
                
            except Exception as e:
                print(f"⚠️ 交叉验证指标 {metric} 计算失败: {str(e)}")
        
        # 添加交叉验证的元信息
        cv_results['cv_folds'] = cv_folds
        cv_results['total_samples'] = len(y)
        cv_results['samples_per_fold'] = len(y) // cv_folds
        cv_results['confidence_interval_method'] = 'Bootstrap on CV scores'
        
        return cv_results
    
    def print_detailed_metrics(self, results, model_name):
        """
        打印详细的评估指标
        
        参数:
        ----
        results : dict
            评估结果
        model_name : str
            模型名称
        """
        if not results:
            print(f"❌ {model_name} 没有评估结果")
            return
        
        print(f"\n📊 {model_name} 详细评估指标:")
        print("=" * 60)
        
        # 主要性能指标
        print("🎯 主要性能指标:")
        main_metrics = [
            ('accuracy', '准确率'), ('precision', '精确率'), ('recall', '召回率'),
            ('f1_score', 'F1分数'), ('roc_auc', 'AUROC'), ('pr_auc', 'AUPRC'),
            ('c_index', 'C-Index'), ('specificity', '特异性'), ('npv', '负预测值'),
            ('brier_score', 'Brier分数')
        ]
        
        for metric_key, metric_name in main_metrics:
            if metric_key in results:
                value = results[metric_key]
                print(f"  {metric_name}: {value:.4f}")
                
                # 打印置信区间（如果存在）
                ci_lower_key = f'{metric_key}_ci_lower'
                ci_upper_key = f'{metric_key}_ci_upper'
                if ci_lower_key in results and ci_upper_key in results:
                    print(f"    95%CI: [{results[ci_lower_key]:.4f}, {results[ci_upper_key]:.4f}]")
        
        # 混淆矩阵
        if all(key in results for key in ['true_positive', 'true_negative', 'false_positive', 'false_negative']):
            print(f"\n🔢 混淆矩阵:")
            print(f"  真阳性 (TP): {results['true_positive']}")
            print(f"  真阴性 (TN): {results['true_negative']}")
            print(f"  假阳性 (FP): {results['false_positive']}")
            print(f"  假阴性 (FN): {results['false_negative']}")
            print(f"  样本总数: {results['sample_size']}") 