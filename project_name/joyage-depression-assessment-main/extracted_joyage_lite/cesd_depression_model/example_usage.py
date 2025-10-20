"""
Example usage of CESD Depression Prediction Model

这个脚本演示了如何使用模块化的CESD抑郁预测模型
"""

from core.main_pipeline import CESDPredictionPipeline

def main():
    """主函数 - 演示完整的流水线使用"""
    
    # 初始化流水线
    pipeline = CESDPredictionPipeline(random_state=42)
    
    # 设置数据路径
    train_data_path = "charls2018 20250722.csv"  # 替换为实际的训练数据路径
    test_data_path = "klosa2018 20250722.csv"   # 替换为实际的测试数据路径（可选）
    
    print("🔥 开始运行CESD抑郁预测模型")
    print("=" * 80)
    
    try:
        # 方法1: 运行完整流水线（推荐）
        success = pipeline.run_full_pipeline(
            charls_file=train_data_path,
            klosa_file=test_data_path,  # 如果没有KLOSA数据，设为None
            enable_hyperparameter_tuning=True  # 是否启用超参数调优
        )
        
        if success:
            print("\n🎉 模型训练和评估成功完成!")
            
            # 可以继续进行新数据预测
            # new_predictions = pipeline.predict_new_data("new_data.csv")
            
        else:
            print("\n❌ 流水线运行失败")
            
    except Exception as e:
        print(f"\n❌ 运行过程中出现错误: {str(e)}")

def example_step_by_step():
    """分步骤运行的示例"""
    
    print("\n" + "="*80)
    print("📚 分步骤运行示例")
    print("="*80)
    
    # 初始化流水线
    pipeline = CESDPredictionPipeline(random_state=42)
    
    try:
        # 步骤1: 加载和预处理数据
        pipeline.load_and_preprocess_data(
            charls_file="charls2018 20250722.csv",
            use_smote=False
        )
        
        # 步骤2: 训练模型
        models = pipeline.train_models()
        print(f"训练了 {len(models)} 个模型")
        
        # 步骤3: 评估模型
        results = pipeline.evaluate_models()
        print(f"评估了 {len(results)} 个模型")
        
        # 步骤4: 生成可视化图表
        pipeline.generate_visualizations()
        
        # 步骤5: 交叉验证最佳模型
        cv_results = pipeline.cross_validate_best_model()
        
        # 步骤6: 保存模型和结果
        pipeline.save_models_and_results()
        
        print("\n✅ 所有步骤完成!")
        
    except Exception as e:
        print(f"\n❌ 分步骤运行出现错误: {str(e)}")

def example_custom_models():
    """自定义模型训练示例"""
    
    print("\n" + "="*80)
    print("🔧 自定义模型训练示例")
    print("="*80)
    
    from models.model_builder import ModelBuilder
    from preprocessing.data_processor import DataProcessor
    from evaluation.model_evaluator import ModelEvaluator
    
    # 单独使用各个模块
    data_processor = DataProcessor()
    model_builder = ModelBuilder()
    evaluator = ModelEvaluator()
    
    # 加载数据
    data = data_processor.load_data("charls2018 20250722.csv")
    processed_data = data_processor.preprocess_data(data, is_training=True)
    
    # 分离特征和目标
    X = processed_data.drop(['depressed'], axis=1)
    y = processed_data['depressed']
    
    # 训练特定模型
    rf_model = model_builder.tune_hyperparameters(
        X, y, model_type='random_forest', search_method='random'
    )
    
    # 评估模型
    results = evaluator.evaluate_model(rf_model, X, y, "RandomForest")
    
    print("✅ 自定义模型训练完成!")

if __name__ == "__main__":
    # 运行主要示例
    main()
    
    # 可选: 运行分步骤示例
    # example_step_by_step()
    
    # 可选: 运行自定义模型示例
    # example_custom_models() 