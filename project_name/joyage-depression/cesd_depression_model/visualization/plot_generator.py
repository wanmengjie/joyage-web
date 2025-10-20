"""
Visualization module for CESD Depression Prediction Model
"""

import numpy as np
import pandas as pd

# 设置matplotlib后端为非交互式，避免tkinter错误
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, roc_auc_score, average_precision_score, auc
import matplotlib.font_manager as fm

class PlotGenerator:
    """图表生成类"""
    
    def __init__(self):
        self._setup_style()
        
    def _setup_style(self):
        """设置图表样式"""
        # 设置字体
        try:
            available_fonts = [f.name for f in fm.fontManager.ttflist]
            if 'Times New Roman' in available_fonts:
                plt.rcParams['font.family'] = 'Times New Roman'
            else:
                plt.rcParams['font.family'] = 'serif'
        except:
            plt.rcParams['font.family'] = 'serif'
            
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.dpi'] = 300
        
    def plot_roc_curves(self, models, X_test, y_test, save_path='plots/roc_curves.png'):
        """绘制ROC曲线"""
        plt.figure(figsize=(10, 8))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        
        for (model_name, model), color in zip(models.items(), colors):
            try:
                # 🔧 修复：检查特征数匹配
                if hasattr(model, 'n_features_in_') and model.n_features_in_ != X_test.shape[1]:
                    print(f"⚠️ {model_name} 特征数不匹配 (模型: {model.n_features_in_}, 数据: {X_test.shape[1]})，跳过")
                    continue
                    
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)[:, 1]
                else:
                    y_proba = model.decision_function(X_test)
                
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, color=color, lw=2, alpha=0.8,
                        label=f'{model_name} (AUC = {roc_auc:.3f})')
            except Exception as e:
                print(f"⚠️ {model_name} ROC曲线绘制失败: {e}")
                continue
                
        plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_precision_recall_curves(self, models, X_test, y_test, save_path='plots/pr_curves.png'):
        """绘制PR曲线"""
        plt.figure(figsize=(10, 8))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        
        for (model_name, model), color in zip(models.items(), colors):
            try:
                # 🔧 修复：检查特征数匹配
                if hasattr(model, 'n_features_in_') and model.n_features_in_ != X_test.shape[1]:
                    print(f"⚠️ {model_name} 特征数不匹配 (模型: {model.n_features_in_}, 数据: {X_test.shape[1]})，跳过")
                    continue
                    
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)[:, 1]
                else:
                    y_proba = model.decision_function(X_test)
                
                precision, recall, _ = precision_recall_curve(y_test, y_proba)
                avg_precision = average_precision_score(y_test, y_proba)
                
                plt.plot(recall, precision, color=color, lw=2, alpha=0.8,
                        label=f'{model_name} (AP = {avg_precision:.3f})')
            except Exception as e:
                print(f"⚠️ {model_name} PR曲线绘制失败: {e}")
                continue
                
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_feature_importance(self, model, feature_names, top_n=20, save_path='feature_importance.png'):
        """绘制特征重要性"""
        if not hasattr(model, 'feature_importances_'):
            print("⚠️ 模型不支持特征重要性")
            return
            
        plt.figure(figsize=(12, 8))
        
        # 自动对齐长度，防止报错
        importances = model.feature_importances_
        if len(feature_names) != len(importances):
            print(f"⚠️ 特征名数量({len(feature_names)})与重要性数量({len(importances)})不一致，自动对齐！")
            min_len = min(len(feature_names), len(importances))
            feature_names = list(feature_names)[:min_len]
            importances = importances[:min_len]
        
        # 获取特征重要性
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        # 绘制条形图
        plt.barh(range(len(importance_df)), importance_df['importance'], alpha=0.8)
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Feature Importance', fontsize=12)
        plt.title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 特征重要性图已保存: {save_path}")
        
    def plot_confusion_matrix(self, model, X_test, y_test, save_path='plots/confusion_matrix.png'):
        """绘制混淆矩阵"""
        try:
            # 🔧 修复：检查特征数匹配
            if hasattr(model, 'n_features_in_') and model.n_features_in_ != X_test.shape[1]:
                print(f"⚠️ 模型特征数不匹配 (模型: {model.n_features_in_}, 数据: {X_test.shape[1]})，跳过混淆矩阵绘制")
                return
            
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Non-Depressed', 'Depressed'],
                       yticklabels=['Non-Depressed', 'Depressed'])
            plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
            plt.ylabel('True Label', fontsize=12)
            plt.xlabel('Predicted Label', fontsize=12)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"⚠️ 混淆矩阵绘制失败: {e}")
        
    def plot_model_comparison(self, results, save_path='model_comparison.png'):
        """绘制模型性能比较"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 提取数据
        models = list(results.keys())
        metrics = ['AUROC', 'AUPRC', 'F1_Score', 'Accuracy']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            values = []
            errors = []
            
            for model in models:
                if metric in results[model]:
                    metric_data = results[model][metric]
                    if isinstance(metric_data, dict):
                        values.append(metric_data['value'])
                        # 计算误差条
                        error = max(
                            metric_data['value'] - metric_data['ci_lower'],
                            metric_data['ci_upper'] - metric_data['value']
                        )
                        errors.append(error)
                    else:
                        values.append(metric_data)
                        errors.append(0)
                else:
                    values.append(0)
                    errors.append(0)
                    
            # 绘制条形图
            bars = ax.bar(models, values, yerr=errors, capsize=5, alpha=0.8)
            ax.set_ylabel(metric, fontsize=12)
            ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=10)
                
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 模型比较图已保存: {save_path}")
        
    def plot_training_validation_comparison(self, train_results, test_results, save_path='train_test_comparison.png'):
        """绘制训练集vs验证集性能对比"""
        plt.figure(figsize=(12, 8))
        
        models = list(train_results.keys())
        metrics = ['AUROC', 'F1_Score']
        
        x = np.arange(len(models))
        width = 0.35
        
        for idx, metric in enumerate(metrics):
            plt.subplot(1, 2, idx + 1)
            
            train_values = [train_results[model][metric] for model in models]
            test_values = [test_results[model][metric] for model in models]
            
            plt.bar(x - width/2, train_values, width, label='Train', alpha=0.8)
            plt.bar(x + width/2, test_values, width, label='Test', alpha=0.8)
            
            plt.xlabel('Models', fontsize=12)
            plt.ylabel(metric, fontsize=12)
            plt.title(f'{metric}: Train vs Test', fontsize=12, fontweight='bold')
            plt.xticks(x, models, rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3, axis='y')
            
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 训练验证对比图已保存: {save_path}")
        
    def plot_calibration_curve(self, model, X_test, y_test, save_path='plots/calibration_curves.png'):
        """绘制校准曲线"""
        try:
            # 🔧 修复：检查特征数匹配
            if hasattr(model, 'n_features_in_') and model.n_features_in_ != X_test.shape[1]:
                print(f"⚠️ 模型特征数不匹配 (模型: {model.n_features_in_}, 数据: {X_test.shape[1]})，跳过校准曲线绘制")
                return
            
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # 计算校准数据
            n_bins = 10
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            bin_centers = []
            bin_accs = []
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = y_test[in_bin].mean()
                    bin_centers.append((bin_lower + bin_upper) / 2)
                    bin_accs.append(accuracy_in_bin)
                    
            plt.figure(figsize=(10, 8))
            plt.plot(bin_centers, bin_accs, 'o-', label='Model', linewidth=2, markersize=6)
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1, label='Perfect Calibration')
            plt.xlabel('Mean Predicted Probability', fontsize=12)
            plt.ylabel('Fraction of Positives', fontsize=12)
            plt.title('Calibration Curve', fontsize=14, fontweight='bold')
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"⚠️ 校准曲线绘制失败: {e}") 