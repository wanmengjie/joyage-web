# CESD抑郁症预测模型 - 增强版

## 📋 概述

本项目是一个基于CHARLS和KLOSA数据集的抑郁症预测模型，专注于**超参数调优**和模型性能优化。

## 🚀 主要功能

### 核心特性
- ✅ **数据预处理**: 自动数据清洗、缺失值处理、变量编码
- ✅ **模型训练**: 支持多种机器学习算法
- ✅ **超参数调优**: 自动化超参数优化，提升模型性能
- ✅ **外部验证**: KLOSA数据集独立验证
- ✅ **SHAP分析**: 模型可解释性分析
- ✅ **可视化**: 丰富的图表和分析报告

### 模型算法
- Random Forest (随机森林)
- Gradient Boosting (梯度提升)
- XGBoost
- LightGBM  
- Logistic Regression (逻辑回归)
- SVM (支持向量机)
- Voting Classifier (投票分类器)
- Stacking Classifier (堆叠分类器)

## 🔧 快速开始

### 基础使用

```python
from core.main_pipeline import CESDPredictionPipeline

# 初始化流水线
pipeline = CESDPredictionPipeline(random_state=42)

# 运行完整流水线 - 包含超参数调优
pipeline.run_full_pipeline(
    charls_file="charls2018.csv",
    klosa_file="klosa2018.csv",  # 可选，用于外部验证
    enable_hyperparameter_tuning=True  # 启用超参数调优
)
```

### 仅超参数调优

```python
# 如果只想运行超参数调优
pipeline.load_and_preprocess_data("charls2018.csv")
pipeline.train_models()
tuned_models, benchmark_df = pipeline.run_hyperparameter_tuning(
    search_method='random',  # 或 'grid'
    n_iter=20  # 随机搜索迭代次数
)
```

## 📊 流程说明

### 8步完整流程

1. **数据加载和预处理** - 数据清洗、编码、缺失值处理
2. **模型训练** - 使用全部特征训练多种算法
3. **超参数调优** - 自动优化模型超参数
4. **生成可视化** - 创建性能图表
5. **交叉验证** - 评估最佳模型稳定性
6. **解释性分析** - SHAP分析和模型诊断
7. **外部验证** - KLOSA数据集验证
8. **保存结果** - 保存模型和分析报告

## 📈 输出文件

### 模型文件
- `best_model.joblib` - 最佳模型
- `data_processor.joblib` - 数据处理器

### 分析报告
- `model_comparison_results.csv` - 模型对比结果
- `cross_validation_results.json` - 交叉验证结果
- `hyperparameter_tuning_results.json` - 超参数调优详情

### 可视化图表
- `plots/` - 各种性能图表
- `enhanced_shap_*/` - SHAP分析图表
- `diagnostics_*/` - 模型诊断图表

## ⚙️ 配置选项

### 超参数调优配置
```python
# 搜索方法
search_method = 'random'  # 'random' 或 'grid'

# 迭代次数（随机搜索）
n_iter = 20  # 建议10-50

# 交叉验证折数
cv_folds = 5  # 在config.py中配置
```

### 模型选择
可在`config.py`中配置要使用的模型类型和超参数网格。

## 📋 系统要求

### Python版本
- Python 3.8+

### 主要依赖
```
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=1.0.0
xgboost>=1.5.0
lightgbm>=3.2.0
shap>=0.40.0
matplotlib>=3.3.0
seaborn>=0.11.0
```

## 📖 API文档

### CESDPredictionPipeline 主要方法

#### `run_full_pipeline(charls_file, klosa_file=None, enable_hyperparameter_tuning=True)`
运行完整的模型开发流程

**参数:**
- `charls_file`: CHARLS数据文件路径
- `klosa_file`: KLOSA数据文件路径（可选）
- `enable_hyperparameter_tuning`: 是否启用超参数调优

#### `run_hyperparameter_tuning(search_method='random', n_iter=20)`
运行超参数调优

**参数:**
- `search_method`: 搜索方法 ('random' 或 'grid')
- `n_iter`: 随机搜索迭代次数

## 🔬 科研特性

### TRIPOD+AI合规
- 完全符合TRIPOD+AI预测模型报告指南
- 系统性记录所有方法学细节
- 提供完整的模型透明度

### 统计严谨性
- Bootstrap方法计算95%置信区间
- 交叉验证确保结果稳定性
- 外部验证评估泛化能力

### 可重现性
- 固定随机种子
- 完整的代码和数据流程记录
- 版本控制和环境管理

## 🎯 性能优化

### 超参数调优策略
- **随机搜索**: 适合快速探索，推荐用于初步优化
- **网格搜索**: 适合精细调优，计算量较大但更全面

### 内存和计算优化
- 智能并行计算配置
- 内存使用优化
- 进度监控和日志记录

## 💡 使用建议

1. **首次使用**: 建议使用默认配置运行完整流水线
2. **性能调优**: 根据初步结果调整超参数搜索范围
3. **科研发表**: 使用完整的TRIPOD+AI合规功能
4. **生产部署**: 使用最佳超参数重新训练最终模型

## 📞 技术支持

如有问题或建议，请查看代码注释或提交Issue。

---

**更新日期**: 2025-01-31
**版本**: 2.0 (移除特征选择，专注超参数调优) 