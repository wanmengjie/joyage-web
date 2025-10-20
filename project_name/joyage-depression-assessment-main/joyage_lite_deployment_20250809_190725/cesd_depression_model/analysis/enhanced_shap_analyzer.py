#!/usr/bin/env python3
"""
增强版SHAP分析器 - 集成统计显著性检验和全面解释
"""

import numpy as np
import pandas as pd

# 设置matplotlib后端为非交互式，避免tkinter错误
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import permutation_test
import os
import warnings
warnings.filterwarnings('ignore')

# SHAP导入
try:
    import shap
    print("✅ SHAP库导入成功")
except ImportError:
    shap = None
    print("❌ SHAP库导入失败")

class EnhancedSHAPAnalyzer:
    """增强版SHAP分析器 - 包含统计显著性检验"""
    
    def __init__(self, model, model_name="Model"):
        """
        初始化增强版SHAP分析器
        
        参数:
        ----
        model : sklearn model
            训练好的模型
        model_name : str
            模型名称
        """
        if shap is None:
            raise ImportError("SHAP库未安装，无法进行解释性分析")
            
        self.model = model
        self.model_name = model_name
        self.explainer = None
        self.shap_values = None
        self.feature_names = None
        
        # 设置matplotlib中文字体
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        except:
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
    def compute_shap_values(self, X_background, X_explain=None, sample_size=None):
        """
        计算SHAP值
        
        参数:
        ----
        X_background : DataFrame
            背景数据集（用于建立基线）
        X_explain : DataFrame, 可选
            需要解释的数据集，如果为None则使用X_background
        sample_size : int, 可选
            用于解释的样本数量，如果为None则使用全部样本
        """
        print(f"\n🔍 计算{self.model_name}的SHAP值...")
        
        if X_explain is None:
            X_explain = X_background
            
        self.feature_names = list(X_background.columns)
        
        # 使用全部数据，不进行采样
        background_sample = X_background
        
        if sample_size is not None and len(X_explain) > sample_size:
            explain_sample = X_explain.sample(n=sample_size, random_state=42)
            print(f"   使用{sample_size}个样本进行解释")
        else:
            explain_sample = X_explain
            print(f"   使用全部{len(X_explain)}个样本进行解释")
            
        # 初始化SHAP解释器
        try:
            if hasattr(self.model, 'feature_importances_'):
                # 树模型
                self.explainer = shap.TreeExplainer(self.model, background_sample)
                print("   使用TreeExplainer")
            else:
                # 其他模型
                self.explainer = shap.KernelExplainer(self.model.predict_proba, background_sample)
                print("   使用KernelExplainer")
                
            # 计算SHAP值
            self.shap_values = self.explainer.shap_values(explain_sample)
            
            # 处理多分类情况
            if isinstance(self.shap_values, list):
                if len(self.shap_values) == 2:
                    # 二分类情况，取正类
                    self.shap_values = self.shap_values[1]
                else:
                    # 多分类情况，取最后一类
                    self.shap_values = self.shap_values[-1]
            
            # 确保SHAP值是2D数组
            if hasattr(self.shap_values, 'ndim'):
                if self.shap_values.ndim == 3:
                    self.shap_values = self.shap_values[:, :, 1]  # 取正类的SHAP值
                elif self.shap_values.ndim == 1:
                    # 如果是1D，重新reshape
                    self.shap_values = self.shap_values.reshape(1, -1)
            
            # 最终验证
            if not hasattr(self.shap_values, 'shape') or self.shap_values.ndim != 2:
                raise ValueError(f"SHAP值格式异常: {type(self.shap_values)}, shape: {getattr(self.shap_values, 'shape', 'unknown')}")
                
            print(f"   ✅ SHAP值计算完成: {self.shap_values.shape}")
            return explain_sample
            
        except Exception as e:
            print(f"   ❌ SHAP值计算失败: {e}")
            return None
    
    def statistical_significance_test(self, n_permutations=1000):
        """
        特征重要性统计显著性检验
        
        参数:
        ----
        n_permutations : int
            置换检验次数
            
        返回:
        ----
        DataFrame: 包含p值和显著性的特征重要性
        """
        print(f"\n📊 进行特征重要性统计显著性检验...")
        
        if self.shap_values is None:
            raise ValueError("请先计算SHAP值")
            
        # 计算每个特征的平均绝对SHAP值
        feature_importance = np.abs(self.shap_values).mean(axis=0)
        
        # 进行置换检验
        p_values = []
        
        for i, feature in enumerate(self.feature_names):
            print(f"   检验特征 {i+1}/{len(self.feature_names)}: {feature}")
            
            # 原始重要性
            original_importance = feature_importance[i]
            
            # 置换检验
            permuted_importance = []
            for _ in range(n_permutations):
                # 随机置换该特征的SHAP值
                permuted_shap = self.shap_values.copy()
                
                # 确保我们正在操作正确的维度
                if permuted_shap.ndim == 2:
                    np.random.shuffle(permuted_shap[:, i])
                    permuted_imp = np.abs(permuted_shap[:, i]).mean()
                else:
                    # 如果维度不对，跳过这个特征
                    print(f"     ⚠️ 跳过特征 {feature}，SHAP值维度异常: {permuted_shap.shape}")
                    permuted_imp = original_importance
                
                permuted_importance.append(permuted_imp)
            
            # 计算p值（双侧检验）
            permuted_importance = np.array(permuted_importance)
            p_value = np.mean(permuted_importance >= original_importance)
            p_values.append(p_value)
        
        # 多重比较校正（Bonferroni）
        p_values = np.array(p_values)
        p_values_corrected = np.minimum(p_values * len(p_values), 1.0)
        
        # 创建结果DataFrame
        significance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importance,
            'p_value': p_values,
            'p_value_corrected': p_values_corrected,
            'significant': p_values_corrected < 0.05,
            'significance_level': ['***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns' 
                                 for p in p_values_corrected]
        }).sort_values('importance', ascending=False)
        
        print(f"   ✅ 统计检验完成")
        print(f"   📊 显著特征数量: {significance_df['significant'].sum()}/{len(significance_df)}")
        
        return significance_df
    
    def generate_comprehensive_plots(self, X_data, save_dir='enhanced_shap_analysis'):
        """
        生成全面的SHAP解释图表
        
        参数:
        ----
        X_data : DataFrame
            用于解释的数据
        save_dir : str
            保存目录
        """
        if self.shap_values is None:
            raise ValueError("请先计算SHAP值")
            
        os.makedirs(save_dir, exist_ok=True)
        print(f"\n📊 生成综合SHAP解释图表...")
        
        # 1. SHAP Summary Plot (Feature Importance)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(self.shap_values, X_data, plot_type="bar", show=False)
        plt.title(f'SHAP Feature Importance - {self.model_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'shap_importance_bar.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. SHAP Summary Plot (Impact Distribution)
        plt.figure(figsize=(12, 10))
        shap.summary_plot(self.shap_values, X_data, show=False)
        plt.title(f'SHAP Feature Impact Distribution - {self.model_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'shap_impact_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Top Features Dependence Plots
        feature_importance = np.abs(self.shap_values).mean(axis=0)
        top_features_idx = np.argsort(feature_importance)[-6:][::-1]  # Top 6 features
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for i, feat_idx in enumerate(top_features_idx):
            feature_name = self.feature_names[feat_idx]
            shap.dependence_plot(feat_idx, self.shap_values, X_data, show=False, ax=axes[i])
            axes[i].set_title(f'Dependence: {feature_name}', fontsize=12, fontweight='bold')
        
        plt.suptitle(f'SHAP Dependence Plots - Top Features ({self.model_name})', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'shap_dependence_plots.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. SHAP Force Plot (示例预测)
        try:
            # 选择几个代表性样本
            sample_indices = [0, len(X_data)//4, len(X_data)//2, 3*len(X_data)//4, -1]
            
            for i, idx in enumerate(sample_indices):
                if idx >= len(self.shap_values):
                    continue
                    
                shap.force_plot(
                    self.explainer.expected_value, 
                    self.shap_values[idx], 
                    X_data.iloc[idx],
                    matplotlib=True,
                    show=False
                )
                plt.savefig(os.path.join(save_dir, f'shap_force_plot_sample_{i+1}.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            print(f"   ⚠️ Force plot生成失败: {e}")
        
        print(f"   ✅ SHAP图表保存到: {save_dir}")
    
    def feature_interaction_analysis(self, X_data, top_n=10):
        """
        特征交互作用分析
        
        参数:
        ----
        X_data : DataFrame
            特征数据
        top_n : int
            分析前N个重要特征的交互作用
            
        返回:
        ----
        DataFrame: 特征交互作用强度
        """
        print(f"\n🔀 进行特征交互作用分析...")
        
        if self.shap_values is None:
            raise ValueError("请先计算SHAP值")
            
        # 选择top特征
        feature_importance = np.abs(self.shap_values).mean(axis=0)
        top_features_idx = np.argsort(feature_importance)[-top_n:][::-1]
        top_features = [self.feature_names[i] for i in top_features_idx]
        
        # 计算特征间的SHAP值相关性（代表交互作用强度）
        shap_values_top = self.shap_values[:, top_features_idx]
        correlation_matrix = np.corrcoef(shap_values_top.T)
        
        # 创建交互作用DataFrame
        interactions = []
        for i in range(len(top_features)):
            for j in range(i+1, len(top_features)):
                interaction_strength = abs(correlation_matrix[i, j])
                interactions.append({
                    'feature_1': top_features[i],
                    'feature_2': top_features[j],
                    'interaction_strength': interaction_strength,
                    'correlation': correlation_matrix[i, j]
                })
        
        interaction_df = pd.DataFrame(interactions).sort_values(
            'interaction_strength', ascending=False
        )
        
        print(f"   ✅ 交互作用分析完成")
        print(f"   🔀 最强交互作用: {interaction_df.iloc[0]['feature_1']} ↔ {interaction_df.iloc[0]['feature_2']}")
        
        return interaction_df
    
    def generate_statistical_report(self, significance_df, interaction_df, save_dir='enhanced_shap_analysis'):
        """
        生成统计分析报告
        
        参数:
        ----
        significance_df : DataFrame
            特征显著性结果
        interaction_df : DataFrame
            特征交互作用结果
        save_dir : str
            保存目录
        """
        print(f"\n📝 生成统计分析报告...")
        
        report_path = os.path.join(save_dir, 'statistical_analysis_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# {self.model_name} - 统计分析报告\n\n")
            
            f.write("## 1. 特征重要性统计显著性检验\n\n")
            f.write("### 方法说明\n")
            f.write("- 使用置换检验评估特征重要性的统计显著性\n")
            f.write("- 应用Bonferroni校正控制多重比较的假阳性率\n")
            f.write("- 显著性水平：*** p<0.001, ** p<0.01, * p<0.05, ns 不显著\n\n")
            
            f.write("### 显著特征排序\n")
            f.write("| 排名 | 特征名称 | 重要性 | P值 | 校正P值 | 显著性 |\n")
            f.write("|------|----------|--------|-----|---------|--------|\n")
            
            for idx, row in significance_df.head(10).iterrows():
                f.write(f"| {idx+1} | {row['feature']} | {row['importance']:.4f} | "
                       f"{row['p_value']:.4f} | {row['p_value_corrected']:.4f} | {row['significance_level']} |\n")
            
            f.write(f"\n**统计摘要：**\n")
            f.write(f"- 总特征数：{len(significance_df)}\n")
            f.write(f"- 显著特征数：{significance_df['significant'].sum()}\n")
            f.write(f"- 显著特征比例：{significance_df['significant'].mean():.2%}\n\n")
            
            f.write("## 2. 特征交互作用分析\n\n")
            f.write("### 最强交互作用（前10对）\n")
            f.write("| 排名 | 特征1 | 特征2 | 交互强度 | 相关系数 |\n")
            f.write("|------|-------|-------|----------|----------|\n")
            
            for idx, row in interaction_df.head(10).iterrows():
                f.write(f"| {idx+1} | {row['feature_1']} | {row['feature_2']} | "
                       f"{row['interaction_strength']:.4f} | {row['correlation']:.4f} |\n")
            
            f.write("\n## 3. 解释性分析结论\n\n")
            top_significant = significance_df[significance_df['significant']].head(5)
            f.write("### 关键发现\n")
            f.write("1. **最重要的显著特征：**\n")
            for _, row in top_significant.iterrows():
                f.write(f"   - {row['feature']} (重要性: {row['importance']:.4f}, {row['significance_level']})\n")
            
            f.write("\n2. **特征交互作用：**\n")
            top_interactions = interaction_df.head(3)
            for _, row in top_interactions.iterrows():
                f.write(f"   - {row['feature_1']} ↔ {row['feature_2']} (强度: {row['interaction_strength']:.4f})\n")
        
        print(f"   ✅ 统计报告保存到: {report_path}")
    
    def run_complete_analysis(self, X_background, X_explain=None, save_dir='enhanced_shap_analysis'):
        """
        运行完整的增强SHAP分析
        
        参数:
        ----
        X_background : DataFrame
            背景数据集
        X_explain : DataFrame, 可选
            解释数据集
        save_dir : str
            保存目录
            
        返回:
        ----
        dict: 分析结果
        """
        print(f"\n🚀 开始{self.model_name}的完整增强SHAP分析")
        print("=" * 60)
        
        # 1. 计算SHAP值
        X_data = self.compute_shap_values(X_background, X_explain)
        if X_data is None:
            return None
        
        # 2. 统计显著性检验
        significance_df = self.statistical_significance_test()
        
        # 3. 特征交互作用分析
        interaction_df = self.feature_interaction_analysis(X_data)
        
        # 4. 生成图表
        self.generate_comprehensive_plots(X_data, save_dir)
        
        # 5. 生成报告
        self.generate_statistical_report(significance_df, interaction_df, save_dir)
        
        # 6. 保存数据
        significance_df.to_csv(os.path.join(save_dir, 'feature_significance.csv'), 
                              index=False, encoding='utf-8-sig')
        interaction_df.to_csv(os.path.join(save_dir, 'feature_interactions.csv'), 
                             index=False, encoding='utf-8-sig')
        
        print(f"\n🎉 增强SHAP分析完成！")
        print(f"📁 结果保存在: {save_dir}")
        
        return {
            'significance_results': significance_df,
            'interaction_results': interaction_df,
            'shap_values': self.shap_values,
            'feature_names': self.feature_names
        } 