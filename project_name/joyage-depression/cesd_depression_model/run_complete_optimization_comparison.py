"""
特征选择 + 参数优化完整效能对比分析
"""

import sys
import os
import time

# 添加模块路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cesd_depression_model.core.main_pipeline import CESDPredictionPipeline
import pandas as pd
import numpy as np

def comprehensive_optimization_comparison():
    """
    完整的优化策略对比分析
    对比4种策略：
    1. 基线：无特征选择 + 默认参数
    2. 特征选择：特征选择 + 默认参数
    3. 参数优化：无特征选择 + 参数调优
    4. 双重优化：特征选择 + 参数调优
    """
    
    print("="*80)
    print("🔍 特征选择 + 参数优化 完整效能对比分析")
    print("="*80)
    
    print("\n📋 对比策略:")
    print("1. 📍 基线策略     - 全特征(42) + 默认参数")
    print("2. 🔧 特征选择     - 精选特征(20) + 默认参数") 
    print("3. ⚙️ 参数优化     - 全特征(42) + 调优参数")
    print("4. 🚀 双重优化     - 精选特征(20) + 调优参数")
    
    print("\n📊 评估维度:")
    print("- 🎯 预测性能 (AUROC, F1-Score)")
    print("- ⏱️ 计算时间 (特征选择 + 参数调优 + 训练)")
    print("- 🔢 模型复杂度 (特征数 + 参数数)")
    print("- 💡 综合效益 (性能/时间比)")
    
    strategies = [
        {
            'name': 'Baseline',
            'description': '基线：全特征+默认参数',
            'use_feature_selection': False,
            'use_hyperparameter_tuning': False,
            'k_best': 42
        },
        {
            'name': 'FeatureSelection',
            'description': '特征选择：精选特征+默认参数',
            'use_feature_selection': True,
            'use_hyperparameter_tuning': False,
            'k_best': 20
        },
        {
            'name': 'HyperparameterTuning',
            'description': '参数优化：全特征+调优参数',
            'use_feature_selection': False,
            'use_hyperparameter_tuning': True,
            'k_best': 42
        },
        {
            'name': 'DoubleOptimization',
            'description': '双重优化：精选特征+调优参数',
            'use_feature_selection': True,
            'use_hyperparameter_tuning': True,
            'k_best': 20
        }
    ]
    
    results = {}
    
    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"🔧 测试策略: {strategy['name']}")
        print(f"📝 描述: {strategy['description']}")
        print(f"{'='*60}")
        
        try:
            result = run_single_strategy(strategy)
            results[strategy['name']] = result
            
            print(f"✅ {strategy['name']} 完成")
            print(f"   特征数: {result['features']}")
            print(f"   最佳AUROC: {result['best_auroc']:.3f}")
            print(f"   总用时: {result['total_time']:.1f}秒")
            
        except Exception as e:
            print(f"❌ {strategy['name']} 失败: {e}")
            results[strategy['name']] = {'error': str(e)}
    
    # 生成综合分析报告
    generate_comprehensive_analysis(results)
    
    return results

def run_single_strategy(strategy):
    """运行单个优化策略"""
    
    # 记录开始时间
    start_time = time.time()
    
    # 初始化流水线
    pipeline = CESDPredictionPipeline(random_state=42)
    
    # 1. 数据预处理
    preprocessing_start = time.time()
    pipeline.load_and_preprocess_data("charls2018 20250722.csv", use_smote=False)
    preprocessing_time = time.time() - preprocessing_start
    
    result = {
        'strategy': strategy,
        'timing': {'preprocessing': preprocessing_time},
        'features': pipeline.X_train.shape[1],
        'samples': pipeline.X_train.shape[0]
    }
    
    # 2. 特征选择（如果启用）
    if strategy['use_feature_selection']:
        fs_start = time.time()
        pipeline.apply_feature_selection(k_best=strategy['k_best'])
        fs_time = time.time() - fs_start
        
        result['timing']['feature_selection'] = fs_time
        result['features'] = strategy['k_best']
    else:
        result['timing']['feature_selection'] = 0
    
    # 3. 模型训练
    training_start = time.time()
    if strategy['use_feature_selection']:
        models = pipeline.train_models(use_feature_selection=True)
    else:
        models = pipeline.train_models(use_feature_selection=False)
    training_time = time.time() - training_start
    
    result['timing']['training'] = training_time
    result['models_count'] = len(models)
    
    # 4. 参数调优（如果启用）
    if strategy['use_hyperparameter_tuning']:
        tuning_start = time.time()
        tuned_models, benchmark_df = pipeline.run_hyperparameter_tuning(
            search_method='random', n_iter=10  # 减少迭代次数以节省时间
        )
        tuning_time = time.time() - tuning_start
        
        result['timing']['hyperparameter_tuning'] = tuning_time
        
        # 使用调优后的模型
        if not benchmark_df.empty:
            best_model_name = benchmark_df.iloc[0]['Model']
            result['best_model'] = best_model_name
            result['best_auroc'] = benchmark_df.iloc[0]['Best_CV_AUC']
            result['tuning_improvement'] = result['best_auroc']
        else:
            result['best_auroc'] = 0
    else:
        result['timing']['hyperparameter_tuning'] = 0
        
        # 5. 模型评估（如果没有参数调优）
        eval_start = time.time()
        if strategy['use_feature_selection']:
            evaluation_results = pipeline.evaluate_models(use_feature_selection=True)
        else:
            evaluation_results = pipeline.evaluate_models(use_feature_selection=False)
        eval_time = time.time() - eval_start
        
        result['timing']['evaluation'] = eval_time
        
        # 计算平均AUROC
        if evaluation_results:
            aurocs = [res['auroc']['value'] for res in evaluation_results.values()]
            result['best_auroc'] = max(aurocs)
            result['avg_auroc'] = np.mean(aurocs)
        else:
            result['best_auroc'] = 0
            result['avg_auroc'] = 0
    
    # 计算总时间
    result['timing']['total'] = time.time() - start_time
    result['total_time'] = result['timing']['total']
    
    return result

def generate_comprehensive_analysis(results):
    """生成综合分析报告"""
    
    print(f"\n{'='*80}")
    print("📊 完整优化策略效能对比报告")
    print(f"{'='*80}")
    
    # 1. 性能对比表
    print(f"\n📈 性能指标对比")
    print("-" * 80)
    
    comparison_data = []
    
    for strategy_name, result in results.items():
        if 'error' not in result:
            comparison_data.append({
                'Strategy': strategy_name,
                'Features': result['features'],
                'Best_AUROC': result['best_auroc'],
                'Total_Time': result['total_time'],
                'FS_Time': result['timing'].get('feature_selection', 0),
                'HP_Time': result['timing'].get('hyperparameter_tuning', 0),
                'Training_Time': result['timing']['training']
            })
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('Best_AUROC', ascending=False)
        
        print(df.to_string(index=False, float_format='%.3f'))
        
        # 保存详细结果
        df.to_csv('complete_optimization_comparison.csv', index=False, encoding='utf-8-sig')
        
        # 2. 效率分析
        print(f"\n⚡ 效率分析")
        print("-" * 50)
        
        baseline_result = results.get('Baseline')
        if baseline_result and 'error' not in baseline_result:
            baseline_auroc = baseline_result['best_auroc']
            baseline_time = baseline_result['total_time']
            
            for _, row in df.iterrows():
                if row['Strategy'] != 'Baseline':
                    perf_improvement = ((row['Best_AUROC'] - baseline_auroc) / baseline_auroc * 100)
                    time_change = ((row['Total_Time'] - baseline_time) / baseline_time * 100)
                    efficiency = perf_improvement / (time_change + 1) if time_change > -100 else perf_improvement
                    
                    print(f"{row['Strategy']:20s}: AUROC {perf_improvement:+5.1f}%, "
                          f"时间 {time_change:+5.1f}%, 效率比 {efficiency:5.1f}")
        
        # 3. 最佳策略推荐
        print(f"\n💡 策略推荐")
        print("-" * 30)
        
        best_performance = df.iloc[0]
        fastest_strategy = df.loc[df['Total_Time'].idxmin()]
        
        print(f"🏆 最佳性能: {best_performance['Strategy']} "
              f"(AUROC: {best_performance['Best_AUROC']:.3f})")
        print(f"⚡ 最快策略: {fastest_strategy['Strategy']} "
              f"(用时: {fastest_strategy['Total_Time']:.1f}秒)")
        
        # 计算效率最优策略
        df['efficiency'] = df['Best_AUROC'] / (df['Total_Time'] / 60)  # AUROC per minute
        most_efficient = df.loc[df['efficiency'].idxmax()]
        
        print(f"⭐ 效率最优: {most_efficient['Strategy']} "
              f"(效率: {most_efficient['efficiency']:.4f} AUROC/分钟)")
        
        # 4. 应用建议
        print(f"\n🎯 应用建议")
        print("-" * 20)
        
        if best_performance['Strategy'] == 'DoubleOptimization':
            print("✅ 推荐使用双重优化策略 (特征选择+参数调优)")
            print("   理由: 性能最优，兼顾效率和效果")
        elif best_performance['Strategy'] == 'FeatureSelection':
            print("✅ 推荐使用特征选择策略")
            print("   理由: 显著提升性能，计算成本适中")
        elif best_performance['Strategy'] == 'HyperparameterTuning':
            print("✅ 推荐使用参数优化策略")
            print("   理由: 通过调参获得最佳性能")
        else:
            print("🤔 当前优化策略效果不明显，建议:")
            print("   - 检查数据质量")
            print("   - 调整特征选择参数")
            print("   - 扩展参数搜索空间")
    
    print(f"\n📁 结果文件:")
    print("   📊 complete_optimization_comparison.csv - 详细对比结果")

def quick_optimization_demo():
    """快速优化演示"""
    
    print("\n🚀 快速优化演示")
    print("-"*50)
    
    try:
        pipeline = CESDPredictionPipeline(random_state=42)
        
        # 加载数据
        pipeline.load_and_preprocess_data("charls2018 20250722.csv", use_smote=False)
        print(f"数据加载完成: {pipeline.X_train.shape}")
        
        # 策略1: 仅特征选择
        print("\n1️⃣ 测试特征选择效果...")
        start = time.time()
        pipeline.apply_feature_selection(k_best=15)
        pipeline.train_models(use_feature_selection=True)
        fs_results = pipeline.evaluate_models(use_feature_selection=True)
        fs_time = time.time() - start
        
        fs_aurocs = [r['auroc']['value'] for r in fs_results.values()]
        fs_best = max(fs_aurocs)
        
        # 策略2: 仅参数调优
        print("\n2️⃣ 测试参数调优效果...")
        pipeline2 = CESDPredictionPipeline(random_state=42)
        pipeline2.load_and_preprocess_data("charls2018 20250722.csv", use_smote=False)
        
        start = time.time()
        tuned_models, benchmark_df = pipeline2.run_hyperparameter_tuning(
            search_method='random', n_iter=5
        )
        hp_time = time.time() - start
        
        hp_best = benchmark_df.iloc[0]['Best_CV_AUC'] if not benchmark_df.empty else 0
        
        # 对比结果
        print(f"\n📊 快速对比结果:")
        print(f"特征选择策略: AUROC {fs_best:.3f}, 用时 {fs_time:.1f}秒, 特征 {pipeline.X_train_selected.shape[1]}个")
        print(f"参数调优策略: AUROC {hp_best:.3f}, 用时 {hp_time:.1f}秒, 特征 {pipeline2.X_train.shape[1]}个")
        
        improvement_fs = ((fs_best - 0.5) / 0.5) * 100  # 假设基线0.5
        improvement_hp = ((hp_best - 0.5) / 0.5) * 100
        
        if fs_best > hp_best:
            print(f"💡 建议: 特征选择效果更好 (+{improvement_fs:.1f}%)")
        else:
            print(f"💡 建议: 参数调优效果更好 (+{improvement_hp:.1f}%)")
            
        return {
            'feature_selection': {'auroc': fs_best, 'time': fs_time, 'features': 15},
            'hyperparameter_tuning': {'auroc': hp_best, 'time': hp_time, 'features': 42}
        }
        
    except Exception as e:
        print(f"❌ 演示失败: {e}")
        return None

if __name__ == "__main__":
    print("请选择运行模式:")
    print("1. 🔍 完整优化对比分析 (推荐，约20-30分钟)")
    print("2. ⚡ 快速优化演示 (约5分钟)")
    
    choice = input("\n请输入选择 (1/2，默认2): ").strip() or "2"
    
    if choice == "1":
        comprehensive_optimization_comparison()
    else:
        quick_optimization_demo()
    
    print("\n🎉 优化对比分析完成！") 