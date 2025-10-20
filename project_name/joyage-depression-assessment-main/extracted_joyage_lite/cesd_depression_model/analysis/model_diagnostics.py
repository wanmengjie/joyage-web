#!/usr/bin/env python3
"""
模型诊断分析器 - 提供全面的模型诊断功能
"""

import numpy as np
import pandas as pd

# 设置matplotlib后端为非交互式，避免tkinter错误
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    precision_recall_curve, roc_curve,
    confusion_matrix, classification_report
)

# 处理不同sklearn版本的brier_score_loss导入
try:
    from sklearn.metrics import brier_score_loss
except ImportError:
    # 为更老版本提供brier_score_loss的简单实现
    def brier_score_loss(y_true, y_prob):
        return np.mean((y_prob - y_true) ** 2)

# 处理不同sklearn版本的calibration_curve导入
try:
    from sklearn.calibration import calibration_curve
except ImportError:
    try:
        from sklearn.metrics import calibration_curve
    except ImportError:
        # 为更老的sklearn版本提供简单的替代实现
        def calibration_curve(y_true, y_prob_pos, n_bins=5, strategy='uniform'):
            bin_boundaries = np.linspace(0., 1. + 1e-8, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            bin_centers = []
            fraction_of_positives = []
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (y_prob_pos > bin_lower) & (y_prob_pos <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    fraction_of_positives_in_bin = y_true[in_bin].mean()
                    bin_centers.append((bin_lower + bin_upper) / 2)
                    fraction_of_positives.append(fraction_of_positives_in_bin)
            
            return np.array(fraction_of_positives), np.array(bin_centers)
import warnings
import os
warnings.filterwarnings('ignore')

class ModelDiagnosticsAnalyzer:
    """模型诊断分析器"""
    
    def __init__(self, model, model_name="Model"):
        """
        初始化模型诊断分析器
        
        参数:
        ----
        model : sklearn model
            训练好的模型
        model_name : str
            模型名称
        """
        self.model = model
        self.model_name = model_name
        
        # 设置绘图样式
        plt.style.use('default')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 12
        
    def calibration_analysis(self, X_test, y_test, n_bins=10, save_dir='model_diagnostics'):
        """
        模型校准分析
        
        参数:
        ----
        X_test : DataFrame
            测试特征
        y_test : Series
            测试标签
        n_bins : int
            校准曲线的分箱数
        save_dir : str
            保存目录
            
        返回:
        ----
        dict: 校准分析结果
        """
        print(f"\n📊 {self.model_name} - 模型校准分析")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 获取预测概率
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        # 计算校准曲线
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, y_prob, n_bins=n_bins, strategy='uniform'
        )
        
        # 计算Brier分数
        brier_score = brier_score_loss(y_test, y_prob)
        
        # 绘制校准曲线
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 子图1：校准曲线
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        ax1.plot(mean_predicted_value, fraction_of_positives, 'o-', 
                label=f'{self.model_name} (Brier Score: {brier_score:.4f})')
        ax1.set_xlabel('Mean Predicted Probability')
        ax1.set_ylabel('Fraction of Positives')
        ax1.set_title('Calibration Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 子图2：预测概率分布
        ax2.hist(y_prob[y_test == 0], bins=20, alpha=0.7, label='Negative Class', density=True)
        ax2.hist(y_prob[y_test == 1], bins=20, alpha=0.7, label='Positive Class', density=True)
        ax2.set_xlabel('Predicted Probability')
        ax2.set_ylabel('Density')
        ax2.set_title('Prediction Probability Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'calibration_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 计算校准指标
        calibration_metrics = {
            'brier_score': brier_score,
            'mean_calibration_error': np.mean(np.abs(fraction_of_positives - mean_predicted_value)),
            'max_calibration_error': np.max(np.abs(fraction_of_positives - mean_predicted_value)),
            'calibration_bins': n_bins,
            'fraction_of_positives': fraction_of_positives,
            'mean_predicted_value': mean_predicted_value
        }
        
        print(f"   ✅ Brier Score: {brier_score:.4f}")
        print(f"   ✅ Mean Calibration Error: {calibration_metrics['mean_calibration_error']:.4f}")
        print(f"   ✅ Max Calibration Error: {calibration_metrics['max_calibration_error']:.4f}")
        
        return calibration_metrics
    
    def residual_analysis(self, X_test, y_test, save_dir='model_diagnostics'):
        """
        残差分析（适用于概率预测）
        
        参数:
        ----
        X_test : DataFrame
            测试特征
        y_test : Series
            测试标签
        save_dir : str
            保存目录
            
        返回:
        ----
        dict: 残差分析结果
        """
        print(f"\n🔍 {self.model_name} - 残差分析")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 获取预测
        y_prob = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        
        # 计算残差（对于分类问题，使用Pearson残差）
        residuals = y_test - y_prob
        pearson_residuals = residuals / np.sqrt(y_prob * (1 - y_prob) + 1e-8)
        
        # 创建残差图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 子图1：残差 vs 预测值
        axes[0, 0].scatter(y_prob, residuals, alpha=0.6)
        axes[0, 0].axhline(y=0, color='red', linestyle='--')
        axes[0, 0].set_xlabel('Predicted Probability')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 子图2：Pearson残差 vs 预测值
        axes[0, 1].scatter(y_prob, pearson_residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='red', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Probability')
        axes[0, 1].set_ylabel('Pearson Residuals')
        axes[0, 1].set_title('Pearson Residuals vs Predicted')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 子图3：残差Q-Q图
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot of Residuals')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 子图4：残差直方图
        axes[1, 1].hist(residuals, bins=30, density=True, alpha=0.7)
        axes[1, 1].set_xlabel('Residuals')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Distribution of Residuals')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'residual_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 计算残差统计量
        residual_stats = {
            'mean_residual': np.mean(residuals),
            'std_residual': np.std(residuals),
            'mean_abs_residual': np.mean(np.abs(residuals)),
            'mean_pearson_residual': np.mean(pearson_residuals),
            'std_pearson_residual': np.std(pearson_residuals),
            'shapiro_test_pvalue': stats.shapiro(residuals)[1],
            'ljung_box_test': self._ljung_box_test(residuals)
        }
        
        print(f"   ✅ Mean Residual: {residual_stats['mean_residual']:.4f}")
        print(f"   ✅ Mean Absolute Residual: {residual_stats['mean_abs_residual']:.4f}")
        print(f"   ✅ Shapiro-Wilk Test p-value: {residual_stats['shapiro_test_pvalue']:.4f}")
        
        return residual_stats
    
    def prediction_distribution_analysis(self, X_test, y_test, save_dir='model_diagnostics'):
        """
        预测分布分析
        
        参数:
        ----
        X_test : DataFrame
            测试特征
        y_test : Series
            测试标签
        save_dir : str
            保存目录
            
        返回:
        ----
        dict: 预测分布分析结果
        """
        print(f"\n📈 {self.model_name} - 预测分布分析")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 获取预测
        y_prob = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        
        # 创建分布分析图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 子图1：预测概率密度图
        axes[0, 0].hist(y_prob[y_test == 0], bins=30, alpha=0.7, label='True Negative', density=True)
        axes[0, 0].hist(y_prob[y_test == 1], bins=30, alpha=0.7, label='True Positive', density=True)
        axes[0, 0].set_xlabel('Predicted Probability')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Prediction Probability Distribution by True Class')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 子图2：预测概率箱线图
        prob_data = [y_prob[y_test == 0], y_prob[y_test == 1]]
        axes[0, 1].boxplot(prob_data, labels=['Negative', 'Positive'])
        axes[0, 1].set_ylabel('Predicted Probability')
        axes[0, 1].set_title('Prediction Probability Box Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 子图3：混淆矩阵热图
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        axes[1, 0].set_title('Confusion Matrix')
        
        # 子图4：阈值分析
        thresholds = np.linspace(0, 1, 101)
        precisions, recalls, f1_scores = [], [], []
        
        for threshold in thresholds:
            y_pred_thresh = (y_prob >= threshold).astype(int)
            if len(np.unique(y_pred_thresh)) > 1:
                precision = np.sum((y_pred_thresh == 1) & (y_test == 1)) / np.sum(y_pred_thresh == 1)
                recall = np.sum((y_pred_thresh == 1) & (y_test == 1)) / np.sum(y_test == 1)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            else:
                precision, recall, f1 = 0, 0, 0
            
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
        
        axes[1, 1].plot(thresholds, precisions, label='Precision')
        axes[1, 1].plot(thresholds, recalls, label='Recall')
        axes[1, 1].plot(thresholds, f1_scores, label='F1-Score')
        axes[1, 1].axvline(x=0.5, color='red', linestyle='--', label='Default Threshold')
        axes[1, 1].set_xlabel('Threshold')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Performance vs Threshold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'prediction_distribution_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 计算分布统计量
        distribution_stats = {
            'prob_mean_negative': np.mean(y_prob[y_test == 0]),
            'prob_std_negative': np.std(y_prob[y_test == 0]),
            'prob_mean_positive': np.mean(y_prob[y_test == 1]),
            'prob_std_positive': np.std(y_prob[y_test == 1]),
            'separation_score': np.abs(np.mean(y_prob[y_test == 1]) - np.mean(y_prob[y_test == 0])),
            'optimal_threshold': thresholds[np.argmax(f1_scores)] if f1_scores else 0.5,
            'max_f1_score': np.max(f1_scores) if f1_scores else 0
        }
        
        print(f"   ✅ Class Separation Score: {distribution_stats['separation_score']:.4f}")
        print(f"   ✅ Optimal Threshold: {distribution_stats['optimal_threshold']:.3f}")
        print(f"   ✅ Max F1 Score: {distribution_stats['max_f1_score']:.4f}")
        
        return distribution_stats
    
    def feature_reliability_analysis(self, X_test, y_test, feature_names=None, save_dir='model_diagnostics'):
        """
        特征可靠性分析
        
        参数:
        ----
        X_test : DataFrame
            测试特征
        y_test : Series
            测试标签
        feature_names : list, 可选
            特征名称列表
        save_dir : str
            保存目录
            
        返回:
        ----
        dict: 特征可靠性分析结果
        """
        print(f"\n🔧 {self.model_name} - 特征可靠性分析")
        
        os.makedirs(save_dir, exist_ok=True)
        
        if feature_names is None:
            feature_names = X_test.columns if hasattr(X_test, 'columns') else [f'Feature_{i}' for i in range(X_test.shape[1])]
        
        # 特征稳定性分析（通过预测一致性）
        feature_stability = []
        
        for i, feature_name in enumerate(feature_names):
            # 创建该特征的扰动版本
            X_perturbed = X_test.copy()
            feature_std = np.std(X_test.iloc[:, i])
            X_perturbed.iloc[:, i] += np.random.normal(0, feature_std * 0.1, len(X_test))
            
            # 比较原始预测和扰动后预测
            y_prob_original = self.model.predict_proba(X_test)[:, 1]
            y_prob_perturbed = self.model.predict_proba(X_perturbed)[:, 1]
            
            # 计算预测一致性
            consistency = 1 - np.mean(np.abs(y_prob_original - y_prob_perturbed))
            feature_stability.append(consistency)
        
        # 创建特征可靠性图
        plt.figure(figsize=(12, 8))
        feature_stability_df = pd.DataFrame({
            'feature': feature_names,
            'stability': feature_stability
        }).sort_values('stability', ascending=True)
        
        plt.barh(range(len(feature_stability_df)), feature_stability_df['stability'])
        plt.yticks(range(len(feature_stability_df)), feature_stability_df['feature'])
        plt.xlabel('Prediction Stability (1 - Mean Absolute Change)')
        plt.title(f'Feature Reliability Analysis - {self.model_name}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'feature_reliability.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        reliability_stats = {
            'feature_stability': dict(zip(feature_names, feature_stability)),
            'mean_stability': np.mean(feature_stability),
            'min_stability': np.min(feature_stability),
            'most_stable_feature': feature_names[np.argmax(feature_stability)],
            'least_stable_feature': feature_names[np.argmin(feature_stability)]
        }
        
        print(f"   ✅ Mean Feature Stability: {reliability_stats['mean_stability']:.4f}")
        print(f"   ✅ Most Stable Feature: {reliability_stats['most_stable_feature']}")
        print(f"   ✅ Least Stable Feature: {reliability_stats['least_stable_feature']}")
        
        return reliability_stats
    
    def _ljung_box_test(self, residuals, lags=10):
        """
        Ljung-Box检验（检验残差的自相关性）
        
        参数:
        ----
        residuals : array
            残差
        lags : int
            滞后阶数
            
        返回:
        ----
        dict: 检验结果
        """
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            result = acorr_ljungbox(residuals, lags=lags, return_df=True)
            return {
                'statistic': float(result['lb_stat'].iloc[-1]),
                'p_value': float(result['lb_pvalue'].iloc[-1]),
                'significant_autocorr': float(result['lb_pvalue'].iloc[-1]) < 0.05
            }
        except ImportError:
            return {
                'statistic': None,
                'p_value': None,
                'significant_autocorr': None,
                'note': 'statsmodels not available'
            }
    
    def generate_diagnostic_report(self, calibration_results, residual_results, 
                                 distribution_results, reliability_results, 
                                 save_dir='model_diagnostics'):
        """
        生成诊断报告
        
        参数:
        ----
        calibration_results : dict
            校准分析结果
        residual_results : dict
            残差分析结果
        distribution_results : dict
            分布分析结果
        reliability_results : dict
            可靠性分析结果
        save_dir : str
            保存目录
        """
        print(f"\n📝 生成{self.model_name}诊断报告...")
        
        report_path = os.path.join(save_dir, f'{self.model_name}_diagnostic_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# {self.model_name} - 模型诊断报告\n\n")
            
            f.write("## 1. 模型校准分析\n\n")
            f.write("### 校准指标\n")
            f.write(f"- **Brier Score**: {calibration_results['brier_score']:.4f}\n")
            f.write(f"- **平均校准误差**: {calibration_results['mean_calibration_error']:.4f}\n")
            f.write(f"- **最大校准误差**: {calibration_results['max_calibration_error']:.4f}\n\n")
            
            f.write("### 解释\n")
            if calibration_results['brier_score'] < 0.1:
                f.write("- ✅ Brier分数较低，模型校准良好\n")
            elif calibration_results['brier_score'] < 0.2:
                f.write("- ⚠️ Brier分数中等，模型校准一般\n")
            else:
                f.write("- ❌ Brier分数较高，模型校准较差\n")
            
            f.write("\n## 2. 残差分析\n\n")
            f.write("### 残差统计\n")
            f.write(f"- **平均残差**: {residual_results['mean_residual']:.4f}\n")
            f.write(f"- **残差标准差**: {residual_results['std_residual']:.4f}\n")
            f.write(f"- **平均绝对残差**: {residual_results['mean_abs_residual']:.4f}\n")
            f.write(f"- **Shapiro-Wilk正态性检验p值**: {residual_results['shapiro_test_pvalue']:.4f}\n\n")
            
            f.write("### 解释\n")
            if abs(residual_results['mean_residual']) < 0.01:
                f.write("- ✅ 平均残差接近0，无系统性偏差\n")
            else:
                f.write("- ⚠️ 平均残差偏离0，可能存在系统性偏差\n")
                
            if residual_results['shapiro_test_pvalue'] > 0.05:
                f.write("- ✅ 残差近似正态分布\n")
            else:
                f.write("- ⚠️ 残差偏离正态分布\n")
            
            f.write("\n## 3. 预测分布分析\n\n")
            f.write("### 分布特征\n")
            f.write(f"- **负类预测概率均值**: {distribution_results['prob_mean_negative']:.4f}\n")
            f.write(f"- **正类预测概率均值**: {distribution_results['prob_mean_positive']:.4f}\n")
            f.write(f"- **类别分离度**: {distribution_results['separation_score']:.4f}\n")
            f.write(f"- **最优阈值**: {distribution_results['optimal_threshold']:.3f}\n")
            f.write(f"- **最大F1分数**: {distribution_results['max_f1_score']:.4f}\n\n")
            
            f.write("### 解释\n")
            if distribution_results['separation_score'] > 0.3:
                f.write("- ✅ 类别分离度良好，模型区分能力强\n")
            elif distribution_results['separation_score'] > 0.1:
                f.write("- ⚠️ 类别分离度中等，模型区分能力一般\n")
            else:
                f.write("- ❌ 类别分离度较低，模型区分能力较弱\n")
            
            f.write("\n## 4. 特征可靠性分析\n\n")
            f.write("### 可靠性指标\n")
            f.write(f"- **平均特征稳定性**: {reliability_results['mean_stability']:.4f}\n")
            f.write(f"- **最稳定特征**: {reliability_results['most_stable_feature']}\n")
            f.write(f"- **最不稳定特征**: {reliability_results['least_stable_feature']}\n\n")
            
            f.write("### Top 5 最稳定特征\n")
            stability_sorted = sorted(reliability_results['feature_stability'].items(), 
                                    key=lambda x: x[1], reverse=True)
            for i, (feature, stability) in enumerate(stability_sorted[:5]):
                f.write(f"{i+1}. {feature}: {stability:.4f}\n")
            
            f.write("\n## 5. 总体诊断结论\n\n")
            f.write("### 模型健康状态\n")
            
            # 综合评分
            health_score = 0
            if calibration_results['brier_score'] < 0.15:
                health_score += 25
            if abs(residual_results['mean_residual']) < 0.01:
                health_score += 25
            if distribution_results['separation_score'] > 0.2:
                health_score += 25
            if reliability_results['mean_stability'] > 0.8:
                health_score += 25
                
            f.write(f"**综合健康评分**: {health_score}/100\n\n")
            
            if health_score >= 80:
                f.write("- ✅ 模型整体健康状态良好\n")
            elif health_score >= 60:
                f.write("- ⚠️ 模型健康状态一般，建议优化\n")
            else:
                f.write("- ❌ 模型健康状态较差，需要重新设计\n")
        
        print(f"   ✅ 诊断报告保存到: {report_path}")
    
    def run_complete_diagnostics(self, X_test, y_test, save_dir='model_diagnostics'):
        """
        运行完整的模型诊断分析
        
        参数:
        ----
        X_test : DataFrame
            测试特征
        y_test : Series
            测试标签
        save_dir : str
            保存目录
            
        返回:
        ----
        dict: 完整诊断结果
        """
        print(f"\n🚀 开始{self.model_name}的完整诊断分析")
        print("=" * 60)
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. 校准分析
        calibration_results = self.calibration_analysis(X_test, y_test, save_dir=save_dir)
        
        # 2. 残差分析
        residual_results = self.residual_analysis(X_test, y_test, save_dir=save_dir)
        
        # 3. 预测分布分析
        distribution_results = self.prediction_distribution_analysis(X_test, y_test, save_dir=save_dir)
        
        # 4. 特征可靠性分析
        reliability_results = self.feature_reliability_analysis(X_test, y_test, save_dir=save_dir)
        
        # 5. 生成综合报告
        self.generate_diagnostic_report(
            calibration_results, residual_results, 
            distribution_results, reliability_results, 
            save_dir=save_dir
        )
        
        print(f"\n🎉 {self.model_name}完整诊断分析完成！")
        print(f"📁 结果保存在: {save_dir}")
        
        return {
            'calibration': calibration_results,
            'residual': residual_results,
            'distribution': distribution_results,
            'reliability': reliability_results,
            'model_name': self.model_name
        } 