"""
基于策略管理器的智能优化脚本
Strategic Optimization Runner with Intelligent Management
"""

import sys
import os
import time
from pathlib import Path

# 添加模块路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cesd_depression_model.core.main_pipeline import CESDPredictionPipeline
from cesd_depression_model.strategy.strategy_manager import (
    StrategyManager, STRATEGY_PRESETS
)
import pandas as pd

def interactive_strategy_selection():
    """交互式策略选择"""
    
    print("🎯 CESD抑郁预测 - 智能策略优化系统")
    print("="*80)
    
    print("\n📋 请选择您的使用场景:")
    print("1. 🚀 快速启动 - 初次尝试，快速获得基本结果 (约13分钟)")
    print("2. ⚖️ 平衡优化 - 时间与性能平衡，适合一般研究 (约48分钟)")  
    print("3. 🏆 性能导向 - 追求最佳性能，适合竞赛应用 (约127分钟)")
    print("4. 🔬 科研级别 - 全面实验设计，适合高质量研究 (约185分钟)")
    print("5. 🛠️ 自定义策略 - 根据您的需求定制优化方案")
    print("6. 💡 智能推荐 - 基于您的约束条件自动推荐")
    
    choice = input("\n请输入您的选择 (1-6): ").strip()
    
    manager = StrategyManager()
    pipeline = CESDPredictionPipeline(random_state=42)
    
    if choice == "1":
        return run_preset_strategy("quick_start", manager, pipeline)
    elif choice == "2":
        return run_preset_strategy("balanced", manager, pipeline)
    elif choice == "3":
        return run_preset_strategy("performance_focused", manager, pipeline)
    elif choice == "4":
        return run_preset_strategy("research_grade", manager, pipeline)
    elif choice == "5":
        return run_custom_strategy(manager, pipeline)
    elif choice == "6":
        return run_intelligent_recommendation(manager, pipeline)
    else:
        print("❌ 无效选择，使用默认平衡优化方案")
        return run_preset_strategy("balanced", manager, pipeline)

def run_preset_strategy(preset_name: str, manager: StrategyManager, pipeline: CESDPredictionPipeline):
    """运行预设策略方案"""
    
    if preset_name not in STRATEGY_PRESETS:
        print(f"❌ 预设方案不存在: {preset_name}")
        return None
        
    preset = STRATEGY_PRESETS[preset_name]
    
    print(f"\n🎯 执行方案: {preset['name']}")
    print(f"📝 描述: {preset['description']}")
    print(f"⏱️ 预计时间: {preset['estimated_time']}分钟")
    print(f"👥 适用对象: {preset['target_users']}")
    
    confirm = input(f"\n确认执行此方案? (y/N): ").strip().lower()
    if confirm != 'y':
        print("❌ 已取消执行")
        return None
    
    # 执行策略序列
    results = manager.execute_strategy_sequence(
        strategy_names=preset['strategies'],
        pipeline=pipeline,
        train_path="charls2018 20250722.csv",
        test_path="klosa2018 20250722.csv"
    )
    
    # 显示执行摘要
    display_execution_summary(results, manager)
    
    return results

def run_custom_strategy(manager: StrategyManager, pipeline: CESDPredictionPipeline):
    """运行自定义策略"""
    
    print(f"\n🛠️ 自定义策略配置")
    print("-" * 50)
    
    print("\n📋 可用策略:")
    strategies_list = list(manager.strategies.keys())
    for i, (name, strategy) in enumerate(manager.strategies.items(), 1):
        print(f"{i:2d}. {name:25s} - {strategy.description}")
        print(f"     预计时间: {strategy.estimated_time}分钟, 预期提升: +{strategy.expected_improvement}%")
    
    print(f"\n请选择要执行的策略 (多个策略用逗号分隔，如: 1,3,5):")
    selection = input("策略编号: ").strip()
    
    try:
        indices = [int(x.strip()) - 1 for x in selection.split(',')]
        selected_strategies = [strategies_list[i] for i in indices if 0 <= i < len(strategies_list)]
        
        if not selected_strategies:
            print("❌ 未选择有效策略")
            return None
            
        print(f"\n✅ 已选择策略: {', '.join(selected_strategies)}")
        
        # 计算预估时间
        total_time = sum(manager.strategies[name].estimated_time for name in selected_strategies)
        total_improvement = sum(manager.strategies[name].expected_improvement for name in selected_strategies)
        
        print(f"📊 预估总时间: {total_time}分钟")
        print(f"🎯 预期性能提升: +{total_improvement:.1f}%")
        
        confirm = input(f"\n确认执行? (y/N): ").strip().lower()
        if confirm != 'y':
            print("❌ 已取消执行")
            return None
        
        # 执行策略序列
        results = manager.execute_strategy_sequence(
            strategy_names=selected_strategies,
            pipeline=pipeline,
            train_path="charls2018 20250722.csv",
            test_path="klosa2018 20250722.csv"
        )
        
        display_execution_summary(results, manager)
        return results
        
    except (ValueError, IndexError) as e:
        print(f"❌ 选择格式错误: {e}")
        return None

def run_intelligent_recommendation(manager: StrategyManager, pipeline: CESDPredictionPipeline):
    """运行智能推荐策略"""
    
    print(f"\n💡 智能策略推荐")
    print("-" * 50)
    
    # 收集用户约束条件
    print("请告诉我您的约束条件:")
    
    try:
        available_time = int(input("可用时间 (分钟，默认60): ").strip() or "60")
        performance_target = float(input("性能提升目标 (百分比，默认5.0): ").strip() or "5.0") 
        resource_constraint = input("资源约束 (低/中/高，默认中): ").strip() or "中"
        
        if resource_constraint not in ["低", "中", "高"]:
            resource_constraint = "中"
            
    except ValueError:
        print("❌ 输入格式错误，使用默认值")
        available_time = 60
        performance_target = 5.0
        resource_constraint = "中"
    
    # 获取智能推荐
    recommended_sequence = manager.recommend_strategy_sequence(
        available_time=available_time,
        performance_target=performance_target,
        resource_constraint=resource_constraint
    )
    
    if not recommended_sequence:
        print("❌ 无法在当前约束下找到合适的策略")
        return None
    
    print(f"\n🎯 推荐执行方案:")
    for i, strategy_name in enumerate(recommended_sequence, 1):
        strategy = manager.strategies[strategy_name]
        print(f"{i}. {strategy_name} - {strategy.description}")
    
    confirm = input(f"\n接受推荐并执行? (y/N): ").strip().lower()
    if confirm != 'y':
        print("❌ 已取消执行")
        return None
    
    # 执行推荐策略
    results = manager.execute_strategy_sequence(
        strategy_names=recommended_sequence,
        pipeline=pipeline,
        train_path="charls2018 20250722.csv",
        test_path="klosa2018 20250722.csv"
    )
    
    display_execution_summary(results, manager)
    return results

def display_execution_summary(results: dict, manager: StrategyManager):
    """显示执行摘要"""
    
    print(f"\n{'='*80}")
    print("📊 执行摘要报告")
    print(f"{'='*80}")
    
    if not results:
        print("❌ 没有执行结果")
        return
    
    # 统计信息
    total_strategies = len(results)
    successful_strategies = sum(1 for r in results.values() if r.success)
    total_time = sum(r.duration for r in results.values())
    
    print(f"\n📈 整体统计:")
    print(f"执行策略数: {total_strategies}")
    print(f"成功策略数: {successful_strategies}")
    print(f"实际总耗时: {total_time:.1f}分钟")
    
    # 性能提升表
    print(f"\n🏆 性能表现:")
    print("-" * 80)
    print(f"{'策略名称':<25} {'状态':<8} {'AUROC':<8} {'提升':<8} {'时间':<8}")
    print("-" * 80)
    
    baseline_auroc = None
    for name, result in results.items():
        if name == "baseline" and result.success:
            baseline_auroc = result.metrics.get('best_auroc', 0)
    
    for name, result in results.items():
        status = "✅ 成功" if result.success else "❌ 失败"
        auroc = result.metrics.get('best_auroc', 0) if result.success else 0
        
        if baseline_auroc and result.success and name != "baseline":
            improvement = ((auroc - baseline_auroc) / baseline_auroc) * 100
            improvement_str = f"+{improvement:.1f}%"
        else:
            improvement_str = "-"
            
        print(f"{name:<25} {status:<8} {auroc:<8.3f} {improvement_str:<8} {result.duration:<8.1f}")
    
    # 最佳策略
    if manager.best_strategy:
        print(f"\n🏆 最佳策略: {manager.best_strategy.strategy_name}")
        print(f"   最高AUROC: {manager.best_strategy.metrics.get('best_auroc', 0):.3f}")
        print(f"   性能提升: {manager.best_strategy.performance_gain:+.1f}%")
    
    # 生成的文件
    print(f"\n📁 生成文件:")
    all_artifacts = set()
    for result in results.values():
        if result.success:
            all_artifacts.update(result.artifacts)
    
    for artifact in sorted(all_artifacts):
        if Path(artifact).exists():
            print(f"   ✅ {artifact}")
        else:
            print(f"   📁 {artifact}")
    
    # 策略推荐
    print(f"\n💡 优化建议:")
    
    if manager.best_strategy and manager.best_strategy.performance_gain > 5:
        print("✨ 恭喜！您已获得显著的性能提升")
        print("   建议：将最佳策略应用到其他类似项目")
    elif manager.best_strategy and manager.best_strategy.performance_gain > 2:
        print("👍 获得了不错的性能提升")
        print("   建议：尝试更激进的优化策略以进一步提升")
    else:
        print("🤔 性能提升有限，建议：")
        print("   1. 检查数据质量和特征工程")
        print("   2. 尝试更多样化的模型")
        print("   3. 调整特征选择和超参数搜索范围")

def batch_comparison_mode():
    """批量对比模式 - 自动运行多个预设方案并对比"""
    
    print(f"\n🔄 批量对比模式")
    print("-" * 50)
    
    presets_to_compare = ["quick_start", "balanced", "performance_focused"]
    
    print("将自动运行以下方案进行对比:")
    for preset_name in presets_to_compare:
        preset = STRATEGY_PRESETS[preset_name]
        print(f"  📋 {preset['name']}: {preset['estimated_time']}分钟")
    
    total_time = sum(STRATEGY_PRESETS[name]['estimated_time'] for name in presets_to_compare)
    print(f"\n⏱️ 预计总时间: {total_time}分钟")
    
    confirm = input("确认运行批量对比? (y/N): ").strip().lower()
    if confirm != 'y':
        print("❌ 已取消执行")
        return
    
    manager = StrategyManager()
    pipeline = CESDPredictionPipeline(random_state=42)
    
    all_results = {}
    
    for preset_name in presets_to_compare:
        print(f"\n{'='*60}")
        print(f"🔧 运行方案: {STRATEGY_PRESETS[preset_name]['name']}")
        print(f"{'='*60}")
        
        # 为每个预设创建新的pipeline实例
        preset_pipeline = CESDPredictionPipeline(random_state=42)
        
        preset_results = manager.execute_strategy_sequence(
            strategy_names=STRATEGY_PRESETS[preset_name]['strategies'],
            pipeline=preset_pipeline,
            train_path="charls2018 20250722.csv",
            test_path="klosa2018 20250722.csv"
        )
        
        all_results[preset_name] = preset_results
    
    # 生成对比报告
    generate_comparison_report(all_results)

def generate_comparison_report(all_results: dict):
    """生成方案对比报告"""
    
    print(f"\n{'='*80}")
    print("📊 方案对比报告")
    print(f"{'='*80}")
    
    comparison_data = []
    
    for preset_name, results in all_results.items():
        preset_info = STRATEGY_PRESETS[preset_name]
        
        # 计算关键指标
        best_auroc = 0
        total_time = 0
        successful_count = 0
        
        for result in results.values():
            if result.success:
                successful_count += 1
                auroc = result.metrics.get('best_auroc', 0)
                if auroc > best_auroc:
                    best_auroc = auroc
            total_time += result.duration
        
        comparison_data.append({
            'Preset': preset_info['name'],
            'Strategies': len(results),
            'Success_Rate': f"{successful_count}/{len(results)}",
            'Best_AUROC': best_auroc,
            'Total_Time': total_time,
            'Efficiency': best_auroc / (total_time / 60) if total_time > 0 else 0  # AUROC per hour
        })
    
    # 生成对比表
    df = pd.DataFrame(comparison_data)
    df = df.sort_values('Best_AUROC', ascending=False)
    
    print("\n🏆 方案性能排名:")
    print(df.to_string(index=False, float_format='%.3f'))
    
    # 保存对比结果
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    df.to_csv(f'strategy_preset_comparison_{timestamp}.csv', index=False, encoding='utf-8-sig')
    
    # 推荐最佳方案
    best_preset = df.iloc[0]
    most_efficient = df.loc[df['Efficiency'].idxmax()]
    
    print(f"\n💡 方案推荐:")
    print(f"🏆 最佳性能: {best_preset['Preset']} (AUROC: {best_preset['Best_AUROC']:.3f})")
    print(f"⚡ 最高效率: {most_efficient['Preset']} (效率: {most_efficient['Efficiency']:.3f})")
    
    print(f"\n📁 对比报告已保存: strategy_preset_comparison_{timestamp}.csv")

if __name__ == "__main__":
    print("请选择运行模式:")
    print("1. 🎯 交互式策略选择 (推荐)")
    print("2. 🔄 批量方案对比")
    print("3. 📊 策略管理器演示")
    
    mode = input("\n请输入选择 (1-3，默认1): ").strip() or "1"
    
    if mode == "2":
        batch_comparison_mode()
    elif mode == "3":
        from cesd_depression_model.strategy.strategy_manager import create_strategy_manager_demo
        create_strategy_manager_demo()
    else:
        interactive_strategy_selection()
    
    print(f"\n🎉 系统运行完成！") 