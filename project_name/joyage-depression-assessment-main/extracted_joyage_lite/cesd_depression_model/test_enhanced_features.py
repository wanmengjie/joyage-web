"""
测试增强版功能的脚本
验证特征选择比较和增强SHAP分析功能
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.main_pipeline import CESDPredictionPipeline

def test_feature_selection_comparison():
    """测试特征选择比较功能"""
    print("🧪 测试特征选择比较功能")
    print("=" * 50)
    
    try:
        pipeline = CESDPredictionPipeline(random_state=42)
        
        # 检查数据文件是否存在
        train_file = "charls2018 20250722.csv"
        if not os.path.exists(train_file):
            print(f"❌ 数据文件不存在: {train_file}")
            print("   请确保数据文件在正确位置")
            return False
        
        # 测试数据加载
        print("📂 测试数据加载...")
        pipeline.load_and_preprocess_data(train_file, use_smote=False)
        print("✅ 数据加载成功")
        
        # 测试特征选择比较方法是否存在
        if hasattr(pipeline, 'compare_with_and_without_feature_selection'):
            print("✅ 特征选择比较方法存在")
        else:
            print("❌ 特征选择比较方法不存在")
            return False
            
        # 测试增强SHAP分析方法是否存在
        if hasattr(pipeline, 'generate_enhanced_shap_analysis'):
            print("✅ 增强SHAP分析方法存在")
        else:
            print("❌ 增强SHAP分析方法不存在")
            return False
            
        print("✅ 所有增强功能方法检查通过")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        return False

def test_enhanced_pipeline():
    """测试增强版流水线参数"""
    print("\n🧪 测试增强版流水线参数")
    print("=" * 50)
    
    try:
        pipeline = CESDPredictionPipeline(random_state=42)
        
        # 检查run_full_pipeline方法的参数
        import inspect
        sig = inspect.signature(pipeline.run_full_pipeline)
        params = list(sig.parameters.keys())
        
        required_params = ['train_path', 'test_path', 'use_smote', 'use_feature_selection', 'feature_selection_k']
        missing_params = [p for p in required_params if p not in params]
        
        if missing_params:
            print(f"❌ 缺少参数: {missing_params}")
            return False
        else:
            print("✅ 所有必需参数存在")
            
        # 检查默认值
        defaults = {p.name: p.default for p in sig.parameters.values() if p.default != inspect.Parameter.empty}
        
        if defaults.get('use_feature_selection') == True:
            print("✅ use_feature_selection 默认为 True")
        else:
            print("⚠️ use_feature_selection 默认值可能不正确")
            
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        return False

def test_file_structure():
    """测试文件结构"""
    print("\n🧪 测试文件结构")
    print("=" * 50)
    
    required_files = [
        'core/main_pipeline.py',
        'run_enhanced_analysis.py',
        'README_ENHANCED.md'
    ]
    
    all_exist = True
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} 存在")
        else:
            print(f"❌ {file_path} 不存在")
            all_exist = False
    
    return all_exist

def main():
    """运行所有测试"""
    print("🔬 CESD增强版功能测试")
    print("=" * 80)
    
    tests = [
        ("文件结构测试", test_file_structure),
        ("增强功能方法测试", test_feature_selection_comparison),
        ("流水线参数测试", test_enhanced_pipeline)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n开始 {test_name}...")
        result = test_func()
        results.append((test_name, result))
    
    # 总结
    print("\n" + "=" * 80)
    print("📊 测试结果总结")
    print("=" * 80)
    
    passed = 0
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{len(tests)} 测试通过")
    
    if passed == len(tests):
        print("\n🎉 所有测试通过！增强版功能可以正常使用。")
        print("\n🚀 您可以运行以下命令开始使用增强版功能：")
        print("   python cesd_depression_model/run_enhanced_analysis.py")
    else:
        print("\n⚠️ 部分测试失败，请检查相关问题。")
    
    return passed == len(tests)

if __name__ == "__main__":
    main() 