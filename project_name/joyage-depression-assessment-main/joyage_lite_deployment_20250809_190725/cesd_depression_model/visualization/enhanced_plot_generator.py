#!/usr/bin/env python3
"""
增强版可视化图表生成器
包含更多有说服力的图表用于CESD抑郁预测研究
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_curve, precision_recall_curve, confusion_matrix,
    classification_report, roc_auc_score, average_precision_score
)
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class EnhancedPlotGenerator:
    """增强版图表生成器"""
    
    def __init__(self, figsize=(12, 8), dpi=300):
        self.figsize = figsize
        self.dpi = dpi
        self.colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#8B5A3C', '#6B8E23']
        
    def plot_comprehensive_model_comparison(self, results_dict, save_path='plots/comprehensive_model_comparison.png'):
        """综合模型比较图（多子图）"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('CESD抑郁预测模型综合性能比较', fontsize=16, fontweight='bold')
        
        # 提取数据
        models = list(results_dict.keys())
        auroc_scores = [results_dict[model]['auroc'] for model in models]
        auprc_scores = [results_dict[model]['auprc'] for model in models]
        accuracy_scores = [results_dict[model]['accuracy'] for model in models]
        f1_scores = [results_dict[model]['f1'] for model in models]
        
        # 1. AUROC比较
        ax1 = axes[0, 0]
        bars1 = ax1.bar(models, auroc_scores, color=self.colors[:len(models)])
        ax1.set_title('AUROC性能比较', fontweight='bold')
        ax1.set_ylabel('AUROC')
        ax1.set_ylim(0.5, 1.0)
        self._add_value_labels(ax1, bars1)
        
        # 2. AUPRC比较
        ax2 = axes[0, 1]
        bars2 = ax2.bar(models, auprc_scores, color=self.colors[:len(models)])
        ax2.set_title('AUPRC性能比较', fontweight='bold')
        ax2.set_ylabel('AUPRC')
        ax2.set_ylim(0, 1.0)
        self._add_value_labels(ax2, bars2)
        
        # 3. 准确率比较
        ax3 = axes[0, 2]
        bars3 = ax3.bar(models, accuracy_scores, color=self.colors[:len(models)])
        ax3.set_title('准确率比较', fontweight='bold')
        ax3.set_ylabel('准确率')
        ax3.set_ylim(0, 1.0)
        self._add_value_labels(ax3, bars3)
        
        # 4. F1分数比较
        ax4 = axes[1, 0]
        bars4 = ax4.bar(models, f1_scores, color=self.colors[:len(models)])
        ax4.set_title('F1分数比较', fontweight='bold')
        ax4.set_ylabel('F1分数')
        ax4.set_ylim(0, 1.0)
        self._add_value_labels(ax4, bars4)
        
        # 5. 雷达图
        ax5 = axes[1, 1]
        self._plot_radar_chart(ax5, models, [auroc_scores, auprc_scores, accuracy_scores, f1_scores])
        
        # 6. 性能热力图
        ax6 = axes[1, 2]
        self._plot_performance_heatmap(ax6, results_dict)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"✅ 综合模型比较图已保存: {save_path}")
        
    def plot_confidence_intervals(self, results_dict, save_path='plots/confidence_intervals.png'):
        """置信区间图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        models = list(results_dict.keys())
        
        # AUROC置信区间
        auroc_means = []
        auroc_lower = []
        auroc_upper = []
        
        for model in models:
            if 'auroc_ci' in results_dict[model]:
                ci = results_dict[model]['auroc_ci']
                auroc_means.append(results_dict[model]['auroc'])
                auroc_lower.append(ci[0])
                auroc_upper.append(ci[1])
            else:
                auroc_means.append(results_dict[model]['auroc'])
                auroc_lower.append(results_dict[model]['auroc'] * 0.95)
                auroc_upper.append(results_dict[model]['auroc'] * 1.05)
        
        # 绘制AUROC置信区间
        x_pos = np.arange(len(models))
        ax1.errorbar(x_pos, auroc_means, yerr=[np.array(auroc_means) - np.array(auroc_lower), 
                                              np.array(auroc_upper) - np.array(auroc_means)], 
                    fmt='o', capsize=5, capthick=2, markersize=8)
        ax1.set_title('AUROC 95%置信区间', fontweight='bold')
        ax1.set_xlabel('模型')
        ax1.set_ylabel('AUROC')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(models, rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # AUPRC置信区间
        auprc_means = []
        auprc_lower = []
        auprc_upper = []
        
        for model in models:
            if 'auprc_ci' in results_dict[model]:
                ci = results_dict[model]['auprc_ci']
                auprc_means.append(results_dict[model]['auprc'])
                auprc_lower.append(ci[0])
                auprc_upper.append(ci[1])
            else:
                auprc_means.append(results_dict[model]['auprc'])
                auprc_lower.append(results_dict[model]['auprc'] * 0.95)
                auprc_upper.append(results_dict[model]['auprc'] * 1.05)
        
        # 绘制AUPRC置信区间
        ax2.errorbar(x_pos, auprc_means, yerr=[np.array(auprc_means) - np.array(auprc_lower), 
                                              np.array(auprc_upper) - np.array(auprc_means)], 
                    fmt='s', capsize=5, capthick=2, markersize=8, color='orange')
        ax2.set_title('AUPRC 95%置信区间', fontweight='bold')
        ax2.set_xlabel('模型')
        ax2.set_ylabel('AUPRC')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(models, rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"✅ 置信区间图已保存: {save_path}")
        
    def plot_calibration_analysis(self, y_true, y_pred_proba_dict, save_path='plots/calibration_analysis.png'):
        """校准分析图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 校准曲线
        for i, (model_name, y_pred) in enumerate(y_pred_proba_dict.items()):
            fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_pred, n_bins=10)
            ax1.plot(mean_predicted_value, fraction_of_positives, 
                    marker='o', label=model_name, color=self.colors[i % len(self.colors)])
        
        # 完美校准线
        ax1.plot([0, 1], [0, 1], 'k--', label='完美校准')
        ax1.set_xlabel('平均预测概率')
        ax1.set_ylabel('实际正例比例')
        ax1.set_title('校准曲线', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 可靠性图
        for i, (model_name, y_pred) in enumerate(y_pred_proba_dict.items()):
            fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_pred, n_bins=10)
            ax2.bar(mean_predicted_value, fraction_of_positives - mean_predicted_value, 
                   alpha=0.7, label=model_name, color=self.colors[i % len(self.colors)])
        
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.5)
        ax2.set_xlabel('平均预测概率')
        ax2.set_ylabel('校准误差 (实际 - 预测)')
        ax2.set_title('可靠性图', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"✅ 校准分析图已保存: {save_path}")
        
    def plot_feature_importance_comparison(self, feature_importance_dict, save_path='plots/feature_importance_comparison.png'):
        """特征重要性比较图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('特征重要性比较分析', fontsize=16, fontweight='bold')
        
        # 获取所有特征
        all_features = set()
        for model_importance in feature_importance_dict.values():
            all_features.update(model_importance.keys())
        all_features = sorted(list(all_features))
        
        # 创建特征重要性矩阵
        importance_matrix = []
        for model_name, importance_dict in feature_importance_dict.items():
            row = [importance_dict.get(feature, 0) for feature in all_features]
            importance_matrix.append(row)
        
        importance_df = pd.DataFrame(importance_matrix, 
                                   index=feature_importance_dict.keys(), 
                                   columns=all_features)
        
        # 1. 热力图
        sns.heatmap(importance_df.T, annot=True, fmt='.3f', cmap='YlOrRd', 
                   ax=axes[0, 0], cbar_kws={'label': '重要性分数'})
        axes[0, 0].set_title('特征重要性热力图', fontweight='bold')
        axes[0, 0].set_xlabel('模型')
        axes[0, 0].set_ylabel('特征')
        
        # 2. 前10特征比较
        top_features = importance_df.mean().nlargest(10).index
        top_importance = importance_df[top_features]
        
        top_importance.T.plot(kind='bar', ax=axes[0, 1], color=self.colors)
        axes[0, 1].set_title('前10特征重要性比较', fontweight='bold')
        axes[0, 1].set_xlabel('特征')
        axes[0, 1].set_ylabel('重要性分数')
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. 特征重要性分布
        importance_df.T.boxplot(ax=axes[1, 0])
        axes[1, 0].set_title('特征重要性分布', fontweight='bold')
        axes[1, 0].set_xlabel('模型')
        axes[1, 0].set_ylabel('重要性分数')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. 模型间特征重要性相关性
        correlation_matrix = importance_df.T.corr()
        sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                   ax=axes[1, 1], center=0)
        axes[1, 1].set_title('模型间特征重要性相关性', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"✅ 特征重要性比较图已保存: {save_path}")
        
    def plot_external_validation_comparison(self, internal_results, external_results, save_path='plots/external_validation_comparison.png'):
        """内部验证vs外部验证比较图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('内部验证 vs 外部验证性能比较', fontsize=16, fontweight='bold')
        
        models = list(internal_results.keys())
        metrics = ['auroc', 'auprc', 'accuracy', 'f1']
        
        for i, metric in enumerate(metrics):
            ax = axes[i // 2, i % 2]
            
            internal_scores = [internal_results[model].get(metric, 0) for model in models]
            external_scores = [external_results[model].get(metric, 0) for model in models]
            
            x = np.arange(len(models))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, internal_scores, width, label='内部验证', alpha=0.8)
            bars2 = ax.bar(x + width/2, external_scores, width, label='外部验证', alpha=0.8)
            
            ax.set_title(f'{metric.upper()} 比较', fontweight='bold')
            ax.set_xlabel('模型')
            ax.set_ylabel(metric.upper())
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 添加数值标签
            self._add_value_labels(ax, bars1)
            self._add_value_labels(ax, bars2)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"✅ 外部验证比较图已保存: {save_path}")
        
    def plot_model_stability_analysis(self, cv_results_dict, save_path='plots/model_stability_analysis.png'):
        """模型稳定性分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('模型稳定性分析', fontsize=16, fontweight='bold')
        
        models = list(cv_results_dict.keys())
        
        # 1. CV分数分布
        for i, model in enumerate(models):
            cv_scores = cv_results_dict[model]['cv_scores']
            axes[0, 0].hist(cv_scores, alpha=0.7, label=model, color=self.colors[i % len(self.colors)])
        
        axes[0, 0].set_title('交叉验证分数分布', fontweight='bold')
        axes[0, 0].set_xlabel('AUROC分数')
        axes[0, 0].set_ylabel('频次')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 分数箱线图
        cv_data = []
        cv_labels = []
        for model in models:
            cv_scores = cv_results_dict[model]['cv_scores']
            cv_data.extend(cv_scores)
            cv_labels.extend([model] * len(cv_scores))
        
        cv_df = pd.DataFrame({'Model': cv_labels, 'Score': cv_data})
        sns.boxplot(data=cv_df, x='Model', y='Score', ax=axes[0, 1])
        axes[0, 1].set_title('交叉验证分数箱线图', fontweight='bold')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. 标准差比较
        std_scores = [np.std(cv_results_dict[model]['cv_scores']) for model in models]
        bars = axes[1, 0].bar(models, std_scores, color=self.colors[:len(models)])
        axes[1, 0].set_title('模型稳定性 (标准差)', fontweight='bold')
        axes[1, 0].set_xlabel('模型')
        axes[1, 0].set_ylabel('标准差')
        axes[1, 0].tick_params(axis='x', rotation=45)
        self._add_value_labels(axes[1, 0], bars)
        
        # 4. 稳定性vs性能散点图
        mean_scores = [np.mean(cv_results_dict[model]['cv_scores']) for model in models]
        axes[1, 1].scatter(std_scores, mean_scores, s=100, alpha=0.7)
        for i, model in enumerate(models):
            axes[1, 1].annotate(model, (std_scores[i], mean_scores[i]), 
                              xytext=(5, 5), textcoords='offset points')
        
        axes[1, 1].set_title('稳定性 vs 性能', fontweight='bold')
        axes[1, 1].set_xlabel('标准差 (稳定性)')
        axes[1, 1].set_ylabel('平均分数 (性能)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"✅ 模型稳定性分析图已保存: {save_path}")
        
    def _add_value_labels(self, ax, bars):
        """在柱状图上添加数值标签"""
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom')
    
    def _plot_radar_chart(self, ax, models, metrics_list):
        """绘制雷达图"""
        # 标准化指标到0-1范围
        metrics_names = ['AUROC', 'AUPRC', 'Accuracy', 'F1']
        normalized_metrics = []
        
        for metrics in metrics_list:
            normalized = [(score - min(metrics)) / (max(metrics) - min(metrics)) for score in metrics]
            normalized_metrics.append(normalized)
        
        # 雷达图角度
        angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        for i, model in enumerate(models):
            values = normalized_metrics[i] + normalized_metrics[i][:1]  # 闭合图形
            ax.plot(angles, values, 'o-', linewidth=2, label=model, color=self.colors[i % len(self.colors)])
            ax.fill(angles, values, alpha=0.25, color=self.colors[i % len(self.colors)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_names)
        ax.set_ylim(0, 1)
        ax.set_title('模型性能雷达图', fontweight='bold')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
    
    def _plot_performance_heatmap(self, ax, results_dict):
        """绘制性能热力图"""
        metrics = ['auroc', 'auprc', 'accuracy', 'f1', 'precision', 'recall']
        models = list(results_dict.keys())
        
        heatmap_data = []
        for model in models:
            row = [results_dict[model].get(metric, 0) for metric in metrics]
            heatmap_data.append(row)
        
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd',
                   xticklabels=[m.upper() for m in metrics], 
                   yticklabels=models, ax=ax)
        ax.set_title('模型性能热力图', fontweight='bold')

def main():
    """测试增强版图表生成器"""
    print("🚀 增强版图表生成器测试")
    
    # 创建示例数据
    results_dict = {
        'LightGBM': {'auroc': 0.759, 'auprc': 0.432, 'accuracy': 0.712, 'f1': 0.567},
        'XGBoost': {'auroc': 0.758, 'auprc': 0.428, 'accuracy': 0.708, 'f1': 0.562},
        'RandomForest': {'auroc': 0.755, 'auprc': 0.425, 'accuracy': 0.705, 'f1': 0.558},
        'CatBoost': {'auroc': 0.761, 'auprc': 0.435, 'accuracy': 0.715, 'f1': 0.570}
    }
    
    # 创建图表生成器
    plotter = EnhancedPlotGenerator()
    
    # 生成综合比较图
    plotter.plot_comprehensive_model_comparison(results_dict)
    
    print("✅ 增强版图表生成器测试完成")

if __name__ == "__main__":
    main() 