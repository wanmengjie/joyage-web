"""
增强版CESD抑郁预测模型运行脚本
包含特征选择比较和增强SHAP分析功能
"""

import sys
import os

# 添加项目路径到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# 使用绝对导入
from cesd_depression_model.core.main_pipeline import CESDPredictionPipeline

def run_enhanced_analysis():
    """运行增强版分析"""
    
    print("🔥 开始运行CESD抑郁预测模型 - 增强版")
    print("=" * 80)
    print("📋 增强版功能:")
    print("  ✅ 全特征 vs 特征选择后的性能比较")
    print("  ✅ 自动选择最佳特征配置")
    print("  ✅ 增强SHAP分析（全特征和特征选择后的对比）")
    print("  ✅ CHARLS训练/测试集 + KLOSA外部验证")
    print("  ✅ 详细的对比分析报告")
    print("=" * 80)
    
    # 初始化增强版流水线
    pipeline = CESDPredictionPipeline(random_state=42)
    
    # 设置数据路径
    train_data_path = "charls2018 20250722.csv"
    external_validation_path = "klosa2018 20250722.csv"
    
    try:
        # 运行增强版完整流水线
        success = pipeline.run_full_pipeline(
            train_path=train_data_path,
            test_path=None,  # 使用数据分割，而非独立测试集
            use_smote=True,  # 使用SMOTE数据平衡
            use_feature_selection=True,  # 启用特征选择比较
            feature_selection_k=20  # 选择前20个特征
        )
        
        if success:
            print("\n🎉 增强版分析成功完成!")
            print("\n📊 主要输出文件:")
            print("1. feature_selection_comparison.csv - 特征选择效果对比")
            print("2. shap_plots_full_features/ - 全特征模型SHAP解释")
            print("3. shap_plots_selected_features/ - 特征选择后模型SHAP解释") 
            print("4. shap_plots_comparison/ - SHAP对比分析")
            print("5. plots/ - 性能比较可视化图表")
            print("6. saved_models/ - 保存的最佳模型")
            
            print("\n🔬 科研建议:")
            print("- 查看 feature_selection_comparison.csv 了解特征选择的效果")
            print("- 对比 shap_plots_full_features 和 shap_plots_selected_features 中的特征重要性")
            print("- 使用 plots/model_comparison.png 展示模型性能对比")
            print("- KLOSA外部验证结果可用于评估模型泛化性能")
            
        else:
            print("\n❌ 增强版分析失败")
            
    except Exception as e:
        print(f"\n❌ 运行过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

def run_feature_selection_only():
    """仅运行特征选择比较分析"""
    
    print("🔍 运行特征选择比较分析")
    print("=" * 50)
    
    pipeline = CESDPredictionPipeline(random_state=42)
    
    try:
        # 1. 加载和预处理数据
        pipeline.load_and_preprocess_data(
            train_path="charls2018 20250722.csv",
            use_smote=True
        )
        
        # 2. 运行特征选择比较
        comparison_results = pipeline.compare_with_and_without_feature_selection(
            k_best=20
        )
        
        print("\n📊 特征选择比较结果:")
        print(comparison_results)
        
        print("\n✅ 特征选择比较完成，结果已保存到 feature_selection_comparison.csv")
        
    except Exception as e:
        print(f"❌ 特征选择比较失败: {str(e)}")

def run_shap_analysis_only():
    """仅运行增强SHAP分析"""
    
    print("🔍 运行增强SHAP分析")
    print("=" * 50)
    
    pipeline = CESDPredictionPipeline(random_state=42)
    
    try:
        # 1. 加载和预处理数据
        pipeline.load_and_preprocess_data(
            train_path="charls2018 20250722.csv",
            use_smote=True
        )
        
        # 2. 运行增强SHAP分析
        shap_results = pipeline.generate_enhanced_shap_analysis(
            datasets=['train', 'test'],
            external_data_paths={'KLOSA': 'klosa2018 20250722.csv'},
            feature_selection_k=20
        )
        
        print("\n✅ 增强SHAP分析完成")
        print("📁 生成的文件:")
        print("  - shap_plots_full_features/ : 全特征模型SHAP解释")
        print("  - shap_plots_selected_features/ : 特征选择后模型SHAP解释")
        print("  - shap_plots_comparison/ : SHAP对比分析")
        
    except Exception as e:
        print(f"❌ 增强SHAP分析失败: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='CESD抑郁预测模型 - 增强版分析')
    parser.add_argument('--mode', choices=['full', 'feature_only', 'shap_only'], 
                       default='full', help='运行模式')
    
    args = parser.parse_args()
    
    if args.mode == 'full':
        run_enhanced_analysis()
    elif args.mode == 'feature_only':
        run_feature_selection_only()
    elif args.mode == 'shap_only':
        run_shap_analysis_only()
    else:
        print("❌ 未知的运行模式") 