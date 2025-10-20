"""
Performance Comparison Analysis for Feature Selection
特征选择效能对比分析模块
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

class PerformanceComparator:
    """特征选择效能对比分析器"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.comparison_results = {}
        self.timing_results = {}
        self.memory_results = {}
        
    def comprehensive_comparison(self, pipeline, train_path, test_path=None, 
                                feature_selection_configs=None):
        """
        全面的特征选择效能对比分析
        
        Parameters:
        -----------
        pipeline : CESDPredictionPipeline
            主流水线对象
        train_path : str
            训练数据路径
        test_path : str, optional
            测试数据路径
        feature_selection_configs : list
            特征选择配置列表
        """
        print(f"\n{'='*80}")
        print("🔍 特征选择全面效能对比分析")
        print(f"{'='*80}")
        
        # 默认特征选择配置
        if feature_selection_configs is None:
            feature_selection_configs = [
                {
                    'name': 'NoSelection',
                    'use_feature_selection': False,
                    'description': '不使用特征选择（全特征）'
                },
                {
                    'name': 'Conservative_20',
                    'use_feature_selection': True,
                    'k_best': 20,
                    'methods': ['variance', 'univariate', 'rfe', 'model_based'],
                    'description': '保守策略（20个特征）'
                },
                {
                    'name': 'Moderate_15',
                    'use_feature_selection': True,
                    'k_best': 15,
                    'methods': ['univariate', 'rfe', 'model_based'],
                    'description': '中等策略（15个特征）'
                },
                {
                    'name': 'Aggressive_10',
                    'use_feature_selection': True,
                    'k_best': 10,
                    'methods': ['rfe', 'model_based'],
                    'description': '激进策略（10个特征）'
                },
                {
                    'name': 'Statistical_Only',
                    'use_feature_selection': True,
                    'k_best': 18,
                    'methods': ['variance', 'univariate'],
                    'description': '仅统计方法（18个特征）'
                }
            ]
        
        # 对每种配置进行完整分析
        for config in feature_selection_configs:
            print(f"\n{'='*60}")
            print(f"🔧 测试配置: {config['name']}")
            print(f"📝 描述: {config['description']}")
            print(f"{'='*60}")
            
            # 重新初始化流水线
            pipeline.__init__(self.random_state)
            
            # 运行分析并记录性能
            config_results = self._run_single_configuration(
                pipeline, train_path, test_path, config
            )
            
            self.comparison_results[config['name']] = config_results
        
        # 生成综合分析报告
        self._generate_comprehensive_report()
        
        return self.comparison_results
    
    def _run_single_configuration(self, pipeline, train_path, test_path, config):
        """运行单个配置的完整分析"""
        results = {
            'config': config,
            'timing': {},
            'memory': {},
            'performance': {},
            'feature_info': {}
        }
        
        try:
            # 1. 数据加载和预处理
            start_time = time.time()
            pipeline.load_and_preprocess_data(train_path, test_path, use_smote=False)
            preprocessing_time = time.time() - start_time
            
            results['timing']['preprocessing'] = preprocessing_time
            results['feature_info']['original_features'] = pipeline.X_train.shape[1]
            results['feature_info']['training_samples'] = pipeline.X_train.shape[0]
            
            # 2. 特征选择（如果启用）
            if config['use_feature_selection']:
                start_time = time.time()
                pipeline.apply_feature_selection(
                    methods=config.get('methods', ['variance', 'univariate', 'rfe', 'model_based']),
                    k_best=config.get('k_best', 20),
                    variance_threshold=config.get('variance_threshold', 0.01)
                )
                feature_selection_time = time.time() - start_time
                
                results['timing']['feature_selection'] = feature_selection_time
                results['feature_info']['selected_features'] = pipeline.X_train_selected.shape[1]
                results['feature_info']['feature_reduction_ratio'] = (
                    1 - pipeline.X_train_selected.shape[1] / pipeline.X_train.shape[1]
                )
            else:
                results['timing']['feature_selection'] = 0
                results['feature_info']['selected_features'] = pipeline.X_train.shape[1]
                results['feature_info']['feature_reduction_ratio'] = 0
            
            # 3. 模型训练
            start_time = time.time()
            models = pipeline.train_models(use_feature_selection=config['use_feature_selection'])
            training_time = time.time() - start_time
            
            results['timing']['training'] = training_time
            results['feature_info']['models_trained'] = len(models)
            
            # 4. 模型评估
            start_time = time.time()
            evaluation_results = pipeline.evaluate_models(use_feature_selection=config['use_feature_selection'])
            evaluation_time = time.time() - start_time
            
            results['timing']['evaluation'] = evaluation_time
            results['timing']['total'] = sum(results['timing'].values())
            
            # 5. 提取性能指标
            for model_name, metrics in evaluation_results.items():
                results['performance'][model_name] = {
                    'auroc': metrics['auroc']['value'],
                    'auprc': metrics['auprc']['value'],
                    'accuracy': metrics['accuracy']['value'],
                    'f1_score': metrics['f1_score']['value'],
                    'precision': metrics['precision']['value'],
                    'recall': metrics['recall']['value']
                }
            
            # 6. 计算平均性能
            all_auroc = [m['auroc'] for m in results['performance'].values()]
            all_f1 = [m['f1_score'] for m in results['performance'].values()]
            
            results['performance']['average'] = {
                'auroc': np.mean(all_auroc),
                'auroc_std': np.std(all_auroc),
                'f1_score': np.mean(all_f1),
                'f1_std': np.std(all_f1),
                'best_auroc': max(all_auroc),
                'best_f1': max(all_f1)
            }
            
            print(f"✅ {config['name']} 配置完成")
            print(f"   特征数: {results['feature_info']['original_features']} → {results['feature_info']['selected_features']}")
            print(f"   平均AUROC: {results['performance']['average']['auroc']:.3f}")
            print(f"   总用时: {results['timing']['total']:.1f}秒")
            
        except Exception as e:
            print(f"❌ {config['name']} 配置失败: {e}")
            results['error'] = str(e)
        
        return results
    
    def _generate_comprehensive_report(self):
        """生成综合分析报告"""
        print(f"\n{'='*80}")
        print("📊 特征选择效能对比综合报告")
        print(f"{'='*80}")
        
        # 1. 性能对比表
        self._generate_performance_table()
        
        # 2. 时间效率对比
        self._generate_timing_analysis()
        
        # 3. 特征维度对比
        self._generate_feature_analysis()
        
        # 4. 可视化对比
        self._generate_visualizations()
        
        # 5. 保存详细结果
        self._save_detailed_results()
    
    def _generate_performance_table(self):
        """生成性能对比表"""
        print(f"\n📈 性能指标对比")
        print("-" * 80)
        
        performance_data = []
        
        for config_name, results in self.comparison_results.items():
            if 'error' not in results:
                row = {
                    'Configuration': config_name,
                    'Features': results['feature_info']['selected_features'],
                    'Reduction': f"{results['feature_info']['feature_reduction_ratio']*100:.1f}%",
                    'Avg_AUROC': results['performance']['average']['auroc'],
                    'Best_AUROC': results['performance']['average']['best_auroc'],
                    'Avg_F1': results['performance']['average']['f1_score'],
                    'Best_F1': results['performance']['average']['best_f1'],
                    'Total_Time': results['timing']['total']
                }
                performance_data.append(row)
        
        performance_df = pd.DataFrame(performance_data)
        performance_df = performance_df.sort_values('Avg_AUROC', ascending=False)
        
        # 显示表格
        print(performance_df.to_string(index=False, float_format='%.3f'))
        
        # 保存表格
        performance_df.to_csv('feature_selection_performance_comparison.csv', 
                             index=False, encoding='utf-8-sig')
        
        # 计算最佳配置
        best_config = performance_df.iloc[0]
        baseline_config = performance_df[performance_df['Configuration'] == 'NoSelection']
        
        if not baseline_config.empty:
            baseline_auroc = baseline_config['Avg_AUROC'].iloc[0]
            best_auroc = best_config['Avg_AUROC']
            improvement = ((best_auroc - baseline_auroc) / baseline_auroc) * 100
            
            print(f"\n🏆 最佳配置: {best_config['Configuration']}")
            print(f"   AUROC改进: {improvement:+.2f}%")
            print(f"   特征减少: {best_config['Reduction']}")
            print(f"   时间节省: {(1 - best_config['Total_Time'] / baseline_config['Total_Time'].iloc[0]) * 100:.1f}%")
    
    def _generate_timing_analysis(self):
        """生成时间效率分析"""
        print(f"\n⏱️ 时间效率分析")
        print("-" * 50)
        
        timing_data = []
        for config_name, results in self.comparison_results.items():
            if 'error' not in results:
                timing_data.append({
                    'Configuration': config_name,
                    'Features': results['feature_info']['selected_features'],
                    'Preprocessing': results['timing']['preprocessing'],
                    'Feature_Selection': results['timing']['feature_selection'],
                    'Training': results['timing']['training'],
                    'Evaluation': results['timing']['evaluation'],
                    'Total': results['timing']['total']
                })
        
        timing_df = pd.DataFrame(timing_data)
        print(timing_df.to_string(index=False, float_format='%.2f'))
        
        # 计算时间节省
        baseline_time = timing_df[timing_df['Configuration'] == 'NoSelection']['Total'].iloc[0]
        timing_df['Time_Savings'] = ((baseline_time - timing_df['Total']) / baseline_time * 100)
        
        print(f"\n💡 时间效率洞察:")
        for _, row in timing_df.iterrows():
            if row['Configuration'] != 'NoSelection':
                print(f"   {row['Configuration']:15s}: {row['Time_Savings']:+6.1f}% 时间变化")
    
    def _generate_feature_analysis(self):
        """生成特征维度分析"""
        print(f"\n🔍 特征维度分析")
        print("-" * 50)
        
        feature_data = []
        for config_name, results in self.comparison_results.items():
            if 'error' not in results:
                feature_data.append({
                    'Configuration': config_name,
                    'Original': results['feature_info']['original_features'],
                    'Selected': results['feature_info']['selected_features'],
                    'Reduction_Ratio': results['feature_info']['feature_reduction_ratio'],
                    'Avg_AUROC': results['performance']['average']['auroc']
                })
        
        feature_df = pd.DataFrame(feature_data)
        print(feature_df.to_string(index=False, float_format='%.3f'))
        
        # 分析特征数量与性能的关系
        correlation = feature_df['Selected'].corr(feature_df['Avg_AUROC'])
        print(f"\n📊 特征数量与AUROC相关性: {correlation:.3f}")
        
        if correlation > 0.5:
            print("   💭 更多特征通常带来更好性能")
        elif correlation < -0.5:
            print("   💭 更少特征可能带来更好性能（避免过拟合）")
        else:
            print("   💭 特征数量与性能关系不明显，质量比数量更重要")
    
    def _generate_visualizations(self):
        """生成可视化对比图"""
        print(f"\n📊 生成可视化对比图...")
        
        # 准备数据
        configs = []
        aurocs = []
        f1s = []
        features = []
        times = []
        
        for config_name, results in self.comparison_results.items():
            if 'error' not in results:
                configs.append(config_name)
                aurocs.append(results['performance']['average']['auroc'])
                f1s.append(results['performance']['average']['f1_score'])
                features.append(results['feature_info']['selected_features'])
                times.append(results['timing']['total'])
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建2x2子图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. AUROC对比
        bars1 = ax1.bar(configs, aurocs, color='skyblue', alpha=0.7)
        ax1.set_title('AUROC Performance Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('AUROC Score')
        ax1.set_ylim(min(aurocs) - 0.01, max(aurocs) + 0.01)
        ax1.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, value in zip(bars1, aurocs):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 2. 特征数量对比
        bars2 = ax2.bar(configs, features, color='lightgreen', alpha=0.7)
        ax2.set_title('Feature Count Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Number of Features')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars2, features):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value}', ha='center', va='bottom', fontsize=10)
        
        # 3. 时间效率对比
        bars3 = ax3.bar(configs, times, color='orange', alpha=0.7)
        ax3.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Total Time (seconds)')
        ax3.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars3, times):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.01,
                    f'{value:.1f}s', ha='center', va='bottom', fontsize=10)
        
        # 4. 效率vs性能散点图
        scatter = ax4.scatter(features, aurocs, s=[t*10 for t in times], 
                             c=range(len(configs)), cmap='viridis', alpha=0.7)
        ax4.set_title('Efficiency vs Performance', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Number of Features')
        ax4.set_ylabel('AUROC Score')
        
        # 添加配置标签
        for i, config in enumerate(configs):
            ax4.annotate(config, (features[i], aurocs[i]), 
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=9, alpha=0.8)
        
        plt.tight_layout()
        plt.savefig('feature_selection_comprehensive_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ 可视化图表已保存: feature_selection_comprehensive_comparison.png")
    
    def _save_detailed_results(self):
        """保存详细结果"""
        # 保存完整的对比结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON格式详细结果
        detailed_file = f'feature_selection_detailed_comparison_{timestamp}.json'
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(self.comparison_results, f, indent=2, ensure_ascii=False, default=str)
        
        # 生成总结报告
        summary_file = f'feature_selection_summary_report_{timestamp}.md'
        self._generate_summary_report(summary_file)
        
        print(f"\n📁 详细结果已保存:")
        print(f"   - {detailed_file}")
        print(f"   - {summary_file}")
        print(f"   - feature_selection_performance_comparison.csv")
        print(f"   - feature_selection_comprehensive_comparison.png")
    
    def _generate_summary_report(self, filename):
        """生成总结报告"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("# 特征选择效能对比分析报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 📊 主要发现\n\n")
            
            # 找出最佳配置
            best_auroc = 0
            best_config = None
            baseline_auroc = 0
            
            for config_name, results in self.comparison_results.items():
                if 'error' not in results:
                    auroc = results['performance']['average']['auroc']
                    if config_name == 'NoSelection':
                        baseline_auroc = auroc
                    elif auroc > best_auroc:
                        best_auroc = auroc
                        best_config = config_name
            
            if best_config and baseline_auroc > 0:
                improvement = ((best_auroc - baseline_auroc) / baseline_auroc) * 100
                f.write(f"### 🏆 最佳特征选择策略: {best_config}\n")
                f.write(f"- **性能改进**: {improvement:+.2f}%\n")
                f.write(f"- **最佳AUROC**: {best_auroc:.3f}\n")
                f.write(f"- **基线AUROC**: {baseline_auroc:.3f}\n\n")
            
            f.write("## 📈 详细结果\n\n")
            f.write("| 配置 | 特征数 | 平均AUROC | 最佳AUROC | 时间(秒) |\n")
            f.write("|------|--------|-----------|-----------|----------|\n")
            
            for config_name, results in self.comparison_results.items():
                if 'error' not in results:
                    f.write(f"| {config_name} | {results['feature_info']['selected_features']} | "
                           f"{results['performance']['average']['auroc']:.3f} | "
                           f"{results['performance']['average']['best_auroc']:.3f} | "
                           f"{results['timing']['total']:.1f} |\n")
            
            f.write("\n## 💡 建议\n\n")
            
            if best_config:
                best_results = self.comparison_results[best_config]
                f.write(f"1. **推荐使用 {best_config} 策略**\n")
                f.write(f"   - 特征数量: {best_results['feature_info']['selected_features']}\n")
                f.write(f"   - 特征选择方法: {', '.join(best_results['config'].get('methods', []))}\n")
                f.write(f"   - 预期性能提升: {improvement:+.2f}%\n\n")
            
            f.write("2. **根据应用场景选择**\n")
            f.write("   - 追求最高性能: 使用最佳配置\n")
            f.write("   - 追求计算效率: 选择特征数较少的配置\n")
            f.write("   - 追求模型解释性: 选择10-15个特征的配置\n\n")

def create_performance_comparison_example():
    """创建性能对比示例函数"""
    print("🚀 特征选择效能对比分析示例")
    print("=" * 50)
    
    try:
        from ..core.main_pipeline import CESDPredictionPipeline
        
        # 创建流水线和对比分析器
        pipeline = CESDPredictionPipeline(random_state=42)
        comparator = PerformanceComparator(random_state=42)
        
        # 运行全面对比分析
        results = comparator.comprehensive_comparison(
            pipeline=pipeline,
            train_path="charls2018 20250722.csv",
            test_path="klosa2018 20250722.csv"
        )
        
        print("\n✅ 特征选择效能对比分析完成！")
        return results
        
    except Exception as e:
        print(f"❌ 对比分析失败: {e}")
        return None

if __name__ == "__main__":
    create_performance_comparison_example() 