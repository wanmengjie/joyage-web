"""
特征选择效能对比分析运行脚本
"""

import sys
import os

# 添加模块路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cesd_depression_model.core.main_pipeline import CESDPredictionPipeline
from cesd_depression_model.analysis.performance_comparator import PerformanceComparator

def run_feature_selection_comparison():
    """运行特征选择效能对比分析"""
    
    print("="*80)
    print("🔍 特征选择 vs 无特征选择 效能对比分析")
    print("="*80)
    
    print("\n📋 分析说明:")
    print("本次分析将对比以下配置的效能差异：")
    print("1. 🚫 无特征选择 - 使用全部42个特征")
    print("2. 🎯 保守特征选择 - 选择20个最重要特征") 
    print("3. ⚡ 中等特征选择 - 选择15个关键特征")
    print("4. 🔥 激进特征选择 - 仅选择10个核心特征")
    print("5. 📊 统计特征选择 - 基于统计方法选择18个特征")
    
    print("\n📊 对比维度:")
    print("- 🎯 预测性能 (AUROC, F1-Score)")
    print("- ⏱️ 计算时间 (预处理, 训练, 评估)")
    print("- 🔢 特征维度 (原始 vs 选择)")
    print("- 💡 效率收益 (性能提升 vs 时间节省)")
    
    try:
        # 创建流水线和对比分析器
        pipeline = CESDPredictionPipeline(random_state=42)
        comparator = PerformanceComparator(random_state=42)
        
        # 定义特征选择配置
        configs = [
            {
                'name': 'NoSelection',
                'use_feature_selection': False,
                'description': '基线：无特征选择（全部42个特征）'
            },
            {
                'name': 'Conservative_20',
                'use_feature_selection': True,
                'k_best': 20,
                'methods': ['variance', 'univariate', 'rfe', 'model_based'],
                'description': '保守策略：4种方法选择20个特征'
            },
            {
                'name': 'Moderate_15', 
                'use_feature_selection': True,
                'k_best': 15,
                'methods': ['univariate', 'rfe', 'model_based'],
                'description': '中等策略：3种方法选择15个特征'
            },
            {
                'name': 'Aggressive_10',
                'use_feature_selection': True,
                'k_best': 10,
                'methods': ['rfe', 'model_based'],
                'description': '激进策略：2种方法选择10个特征'
            }
        ]
        
        print(f"\n⏳ 开始效能对比分析...")
        print(f"   预计用时: 10-15分钟")
        print(f"   数据集: CHARLS 2018")
        
        # 运行全面对比分析
        results = comparator.comprehensive_comparison(
            pipeline=pipeline,
            train_path="charls2018 20250722.csv",
            test_path=None,  # 使用数据分割
            feature_selection_configs=configs
        )
        
        print("\n" + "="*80)
        print("🎉 特征选择效能对比分析完成！")
        print("="*80)
        
        # 显示主要结论
        print_key_findings(results)
        
        print("\n📁 生成的文件:")
        print("   📊 feature_selection_performance_comparison.csv - 性能对比表")
        print("   📈 feature_selection_comprehensive_comparison.png - 可视化对比")
        print("   📋 feature_selection_summary_report_*.md - 详细分析报告")
        print("   📄 feature_selection_detailed_comparison_*.json - 完整数据")
        
        return results
        
    except Exception as e:
        print(f"\n❌ 效能对比分析失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def print_key_findings(results):
    """打印关键发现"""
    print("\n🔍 关键发现总结:")
    
    if not results:
        print("   ❌ 无有效结果")
        return
    
    # 提取关键数据
    baseline_auroc = 0
    best_auroc = 0
    best_config = None
    
    summary_data = []
    
    for config_name, result in results.items():
        if 'error' not in result:
            auroc = result['performance']['average']['auroc']
            features = result['feature_info']['selected_features']
            time_total = result['timing']['total']
            
            summary_data.append({
                'config': config_name,
                'auroc': auroc,
                'features': features,
                'time': time_total
            })
            
            if config_name == 'NoSelection':
                baseline_auroc = auroc
            elif auroc > best_auroc:
                best_auroc = auroc
                best_config = config_name
    
    # 显示对比结果
    print(f"\n📊 性能对比:")
    for data in sorted(summary_data, key=lambda x: x['auroc'], reverse=True):
        if data['config'] == 'NoSelection':
            status = "📍 基线"
        elif data['config'] == best_config:
            status = "🏆 最佳"
        else:
            status = "   "
        
        improvement = ((data['auroc'] - baseline_auroc) / baseline_auroc * 100) if baseline_auroc > 0 else 0
        
        print(f"   {status} {data['config']:15s}: AUROC {data['auroc']:.3f} "
              f"({improvement:+.1f}%), {data['features']:2d}特征, {data['time']:4.1f}秒")
    
    # 计算效率收益
    if best_config and baseline_auroc > 0:
        best_data = next(d for d in summary_data if d['config'] == best_config)
        baseline_data = next(d for d in summary_data if d['config'] == 'NoSelection')
        
        performance_gain = ((best_data['auroc'] - baseline_data['auroc']) / baseline_data['auroc']) * 100
        feature_reduction = (1 - best_data['features'] / baseline_data['features']) * 100
        time_change = ((best_data['time'] - baseline_data['time']) / baseline_data['time']) * 100
        
        print(f"\n💡 效率收益分析:")
        print(f"   🎯 性能提升: {performance_gain:+.2f}%")
        print(f"   📉 特征减少: {feature_reduction:.1f}%")
        print(f"   ⏱️ 时间变化: {time_change:+.1f}%")
        
        # 给出建议
        if performance_gain > 1.0:
            print(f"\n✅ 建议: 使用 {best_config} 策略")
            print(f"   理由: 显著提升性能 ({performance_gain:+.1f}%)，同时减少特征维度")
        elif feature_reduction > 20 and performance_gain > -2.0:
            print(f"\n💡 建议: 考虑使用特征选择")
            print(f"   理由: 大幅减少特征维度，性能损失可接受")
        else:
            print(f"\n🤔 建议: 保持全特征或调整特征选择参数")
            print(f"   理由: 当前特征选择策略收益不明显")

def quick_comparison():
    """快速对比示例"""
    print("\n🚀 快速对比示例")
    print("-"*50)
    
    try:
        # 创建流水线
        pipeline = CESDPredictionPipeline(random_state=42)
        
        # 加载数据
        pipeline.load_and_preprocess_data("charls2018 20250722.csv", use_smote=False)
        
        print(f"原始特征数: {pipeline.X_train.shape[1]}")
        
        # 1. 训练无特征选择模型
        print("\n🔧 训练基线模型（无特征选择）...")
        import time
        start = time.time()
        pipeline.train_models(use_feature_selection=False)
        results_baseline = pipeline.evaluate_models(use_feature_selection=False)
        time_baseline = time.time() - start
        
        # 2. 应用特征选择并训练
        print("\n🔍 应用特征选择...")
        pipeline.apply_feature_selection(k_best=15)
        
        print("\n🔧 训练特征选择模型...")
        start = time.time()
        pipeline.train_models(use_feature_selection=True)
        results_selected = pipeline.evaluate_models(use_feature_selection=True)
        time_selected = time.time() - start
        
        # 3. 对比结果
        print(f"\n📊 快速对比结果:")
        
        # 计算平均AUROC
        baseline_aurocs = [r['auroc']['value'] for r in results_baseline.values()]
        selected_aurocs = [r['auroc']['value'] for r in results_selected.values()]
        
        avg_baseline = sum(baseline_aurocs) / len(baseline_aurocs)
        avg_selected = sum(selected_aurocs) / len(selected_aurocs)
        
        improvement = ((avg_selected - avg_baseline) / avg_baseline) * 100
        
        print(f"   基线模型 (42特征): AUROC {avg_baseline:.3f}, 用时 {time_baseline:.1f}秒")
        print(f"   特征选择 (15特征): AUROC {avg_selected:.3f}, 用时 {time_selected:.1f}秒")
        print(f"   性能变化: {improvement:+.2f}%")
        print(f"   特征减少: {(1-15/42)*100:.1f}%")
        
        return {
            'baseline': {'auroc': avg_baseline, 'time': time_baseline, 'features': 42},
            'selected': {'auroc': avg_selected, 'time': time_selected, 'features': 15},
            'improvement': improvement
        }
        
    except Exception as e:
        print(f"❌ 快速对比失败: {e}")
        return None

if __name__ == "__main__":
    # 选择运行模式
    print("请选择运行模式:")
    print("1. 🔍 完整效能对比分析 (推荐，约15分钟)")
    print("2. ⚡ 快速对比示例 (约3分钟)")
    
    choice = input("\n请输入选择 (1/2，默认1): ").strip() or "1"
    
    if choice == "2":
        quick_comparison()
    else:
        run_feature_selection_comparison()
    
    print("\n🎉 分析完成！") 