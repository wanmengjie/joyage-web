"""
SHAP Analysis module for CESD Depression Prediction Model
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 导入SHAP库
try:
    import shap
    print("✅ SHAP库导入成功")
except ImportError:
    print("❌ SHAP库未安装，请运行: pip install shap")
    shap = None

from ..config import CATEGORICAL_VARS, EXCLUDED_VARS

class SHAPAnalyzer:
    """SHAP分析器"""
    
    def __init__(self, model, label_encoders=None):
        self.model = model
        self.label_encoders = label_encoders or {}
        self._setup_matplotlib()
        
    def _setup_matplotlib(self):
        """设置matplotlib样式"""
        try:
            import matplotlib.font_manager as fm
            available_fonts = [f.name for f in fm.fontManager.ttflist]
            if 'Times New Roman' in available_fonts:
                plt.rcParams['font.family'] = 'Times New Roman'
            else:
                plt.rcParams['font.family'] = 'serif'
        except:
            plt.rcParams['font.family'] = 'serif'
        plt.rcParams['axes.unicode_minus'] = False
        
    def generate_shap_explanations(self, X, save_dir='shap_plots'):
        """
        生成SHAP解释图表
        """
        print(f"\n{'='*60}")
        print("生成SHAP解释")
        print(f"{'='*60}")
        
        print(f"使用模型: {type(self.model).__name__}")
        
        # 检查SHAP库是否可用
        if shap is None:
            print("❌ SHAP库未导入，无法生成SHAP解释。")
            return None
            
        os.makedirs(save_dir, exist_ok=True)
        
        # 检查模型类型
        if not hasattr(self.model, 'feature_importances_'):
            print("❌ 模型不支持SHAP分析（需要树模型）")
            return None
        
        # 准备数据
        X_for_shap = self._prepare_shap_data(X)
        
        # 初始化SHAP解释器
        print("初始化SHAP解释器...")
        explainer = shap.TreeExplainer(self.model, X_for_shap)
        shap_values = explainer.shap_values(X_for_shap)
        
        if isinstance(shap_values, list):
            # 对于多分类，取正类的SHAP值
            shap_values = shap_values[1]
        
        print(f"SHAP值计算完成，形状: {shap_values.shape}")
        
        # 生成各种图表
        self._plot_summary_plots(shap_values, X_for_shap, save_dir)
        self._plot_dependence_plots(shap_values, X_for_shap, save_dir)
        self._plot_individual_explanations(shap_values, X_for_shap, explainer, save_dir)
        self._plot_interaction_heatmap(explainer, X_for_shap, save_dir)
        
        print(f"✅ SHAP图表已保存到: {save_dir}")
        return shap_values
        
    def _prepare_shap_data(self, X):
        """为SHAP分析准备数据"""
        X_prepared = X.copy()
        
        # 确保分类变量是整数
        for col in CATEGORICAL_VARS:
            if col in X_prepared.columns:
                if not pd.api.types.is_integer_dtype(X_prepared[col]):
                    X_prepared[col] = X_prepared[col].round().astype(int)
                    
        return X_prepared
        
    def _plot_summary_plots(self, shap_values, X, save_dir):
        """生成SHAP摘要图"""
        print("生成SHAP摘要图...")
        
        # 特征重要性条形图
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        plt.title("SHAP Feature Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'shap_summary_bar.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 特征影响散点图
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X, show=False)
        plt.title("SHAP Feature Impact")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'shap_summary_dot.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_dependence_plots(self, shap_values, X, save_dir):
        """生成依赖性图"""
        print("生成SHAP依赖性图...")
        
        # 选择前10个最重要的特征
        feature_importance = np.abs(shap_values).mean(0)
        top_features_idx = np.argsort(feature_importance)[-10:]
        top_features = X.columns[top_features_idx]
        
        for feature in top_features:
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(feature, shap_values, X, show=False)
            
            # 为分类变量添加标签
            if feature in CATEGORICAL_VARS:
                self._add_categorical_labels(feature, X)
                
            plt.title(f"SHAP Dependence Plot - {feature}")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'shap_dependence_{feature}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
    def _plot_individual_explanations(self, shap_values, X, explainer, save_dir, n_samples=5):
        """生成个体解释图"""
        print("生成个体解释图...")
        
        # 随机选择样本
        sample_indices = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
        
        for i, idx in enumerate(sample_indices):
            # 力图
            plt.figure(figsize=(16, 8))
            shap.force_plot(
                explainer.expected_value,
                shap_values[idx],
                X.iloc[idx],
                matplotlib=True,
                show=False
            )
            plt.title(f"Individual Prediction Explanation - Sample {i+1}")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'shap_force_sample_{i+1}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # 瀑布图
            plt.figure(figsize=(12, 8))
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[idx],
                    base_values=explainer.expected_value,
                    data=X.iloc[idx],
                    feature_names=X.columns.tolist()
                ),
                show=False
            )
            plt.title(f"Individual Prediction Explanation (Waterfall) - Sample {i+1}")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'shap_waterfall_sample_{i+1}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
    def _plot_interaction_heatmap(self, explainer, X, save_dir):
        """生成交互效应热图"""
        print("生成SHAP交互效应热图...")
        
        try:
            # 计算交互值（可能需要较长时间）
            shap_interaction_values = explainer.shap_interaction_values(X.iloc[:1000])  # 限制样本数
            interaction_matrix = np.abs(shap_interaction_values).mean(0)
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(interaction_matrix, 
                       xticklabels=X.columns, 
                       yticklabels=X.columns, 
                       cmap='Reds',
                       annot=False)
            plt.title("SHAP Interaction Effect Heatmap")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'shap_interaction_heatmap.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"⚠️ 交互效应热图生成失败: {e}")
            
    def _add_categorical_labels(self, feature, X):
        """为分类变量添加标签"""
        if feature in self.label_encoders:
            unique_values = sorted(X[feature].unique())
            try:
                labels = []
                for val in unique_values:
                    try:
                        original_label = self.label_encoders[feature].classes_[int(val)]
                        labels.append(f"{val}={original_label}")
                    except (IndexError, ValueError):
                        labels.append(str(int(val)))
                
                plt.xticks(unique_values, [str(int(x)) for x in unique_values])
                
                # 添加标签说明
                label_text = " | ".join(labels)
                plt.figtext(0.5, 0.02, f"Labels: {label_text}", 
                           ha='center', fontsize=9,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
            except Exception as e:
                print(f"⚠️ {feature} 标签添加失败: {e}")
        else:
            # 如果没有编码器，只显示数值
            unique_values = sorted(X[feature].unique())
            plt.xticks(unique_values, [str(int(x)) for x in unique_values]) 