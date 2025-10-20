"""
特征选择使用示例
"""

import sys
import os

# 添加模块路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cesd_depression_model.core.main_pipeline import CESDPredictionPipeline

def example_feature_selection():
    """特征选择示例"""
    
    print("="*80)
    print("🔍 特征选择示例")
    print("="*80)
    
    # 创建流水线
    pipeline = CESDPredictionPipeline(random_state=42)
    
    # 1. 基本特征选择流程
    print("\n📊 示例1: 基本特征选择流程")
    print("-"*50)
    
    try:
        # 加载和预处理数据
        pipeline.load_and_preprocess_data(
            train_path="charls2018 20250722.csv",
            test_path=None,  # 使用数据分割
            use_smote=False
        )
        
        # 应用特征选择
        summary = pipeline.apply_feature_selection(
            methods=['variance', 'univariate', 'rfe', 'model_based'],
            k_best=15,  # 选择15个最重要的特征
            variance_threshold=0.01
        )
        
        print("\n✅ 基本特征选择完成")
        if summary is not None:
            print(f"选择的特征:\n{summary['feature'].tolist()}")
        
    except Exception as e:
        print(f"❌ 基本特征选择失败: {e}")
    
    # 2. 特征选择效果比较
    print("\n📈 示例2: 特征选择效果比较")
    print("-"*50)
    
    try:
        # 重新初始化流水线
        pipeline2 = CESDPredictionPipeline(random_state=42)
        
        # 加载数据
        pipeline2.load_and_preprocess_data(
            train_path="charls2018 20250722.csv",
            use_smote=False
        )
        
        # 比较特征选择前后的效果
        comparison = pipeline2.compare_with_and_without_feature_selection(
            methods=['univariate', 'rfe', 'model_based'],
            k_best=20
        )
        
        print("\n✅ 特征选择效果比较完成")
        print("\n📊 改进效果:")
        print(comparison[['Model', 'Improvement', 'Improvement_Percent']].head())
        
    except Exception as e:
        print(f"❌ 特征选择效果比较失败: {e}")
    
    # 3. 完整流水线中使用特征选择
    print("\n🚀 示例3: 完整流水线中使用特征选择")
    print("-"*50)
    
    try:
        # 重新初始化流水线
        pipeline3 = CESDPredictionPipeline(random_state=42)
        
        # 运行包含特征选择的完整流水线
        success = pipeline3.run_full_pipeline(
            train_path="charls2018 20250722.csv",
            test_path="klosa2018 20250722.csv",
            use_smote=True,
            use_feature_selection=True,  # 启用特征选择
            feature_selection_k=25       # 选择25个特征
        )
        
        if success:
            print("✅ 包含特征选择的完整流水线运行成功")
        else:
            print("❌ 完整流水线运行失败")
            
    except Exception as e:
        print(f"❌ 完整流水线运行失败: {e}")

def example_custom_feature_selection():
    """自定义特征选择示例"""
    
    print("\n🛠️ 自定义特征选择示例")
    print("="*50)
    
    from cesd_depression_model.preprocessing import DataProcessor, FeatureSelector
    
    try:
        # 单独使用特征选择器
        data_processor = DataProcessor()
        feature_selector = FeatureSelector(random_state=42)
        
        # 加载和预处理数据
        data = data_processor.load_data("charls2018 20250722.csv")
        processed_data = data_processor.preprocess_data(data, is_training=True)
        
        # 分离特征和目标
        X = processed_data.drop(['depressed'], axis=1)
        y = processed_data['depressed']
        
        print(f"原始特征数: {X.shape[1]}")
        
        # 1. 只使用单变量选择
        print("\n📈 单变量特征选择:")
        selector1 = FeatureSelector(random_state=42)
        X_univariate = selector1.fit_transform(
            X, y, 
            methods=['univariate'], 
            k_best=10
        )
        print(f"选择特征数: {X_univariate.shape[1]}")
        print(f"选择的特征: {list(X_univariate.columns)}")
        
        # 2. 只使用基于模型的选择
        print("\n🌳 基于模型的特征选择:")
        selector2 = FeatureSelector(random_state=42)
        X_model_based = selector2.fit_transform(
            X, y,
            methods=['model_based'],
            k_best=15
        )
        print(f"选择特征数: {X_model_based.shape[1]}")
        print(f"选择的特征: {list(X_model_based.columns)}")
        
        # 3. 组合多种方法
        print("\n🎯 组合特征选择:")
        selector3 = FeatureSelector(random_state=42)
        X_ensemble = selector3.fit_transform(
            X, y,
            methods=['variance', 'univariate', 'model_based'],
            k_best=20,
            variance_threshold=0.005
        )
        print(f"选择特征数: {X_ensemble.shape[1]}")
        
        # 保存结果
        summary3 = selector3.save_results('custom_feature_selection.csv')
        print("✅ 自定义特征选择结果已保存")
        
    except Exception as e:
        print(f"❌ 自定义特征选择失败: {e}")

if __name__ == "__main__":
    # 运行示例
    example_feature_selection()
    example_custom_feature_selection()
    
    print("\n🎉 所有特征选择示例完成!")
    print("\n📁 生成的文件:")
    print("- feature_selection_results.csv: 特征选择详细结果")
    print("- feature_selection_results_detailed.json: 特征选择详细信息")
    print("- feature_selection_comparison.csv: 特征选择效果比较")
    print("- custom_feature_selection.csv: 自定义特征选择结果") 