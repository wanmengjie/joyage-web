#!/bin/bash

# 设置工作目录
cd "$(dirname "$0")/.."

# 创建日志目录
mkdir -p 08_logs

# 记录开始时间
echo "Analysis started at $(date)" > 08_logs/training_log.txt

# 1. 数据清理
echo "Step 1: Data cleaning..."
python 03_scripts/01_cleaning.py

# 2. 缺失值插补
echo "Step 2: Missing value imputation..."
python 03_scripts/02_imputation.py

# 3. 特征工程
echo "Step 3: Feature engineering..."
python 03_scripts/03_feature_engineering.py

# 4. 基线分析
echo "Step 4: Baseline analysis..."
python 03_scripts/04_baseline_analysis.py

# 5. 超参数调优
echo "Step 5: Hyperparameter tuning..."
python 03_scripts/05a_hyperparam_search.py

# 6. 模型训练
echo "Step 6: Model training..."
python 03_scripts/05_model_training.py

# 7. 外部验证
echo "Step 7: External validation..."
python 03_scripts/06_validation.py

# 8. 指标计算
echo "Step 8: Metrics calculation..."
python 03_scripts/07_metrics_ci.py

# 9. 敏感性分析
echo "Step 9: Sensitivity analysis..."
python 03_scripts/08_sensitivity.py

# 10. SHAP分析
echo "Step 10: SHAP analysis..."
python 03_scripts/shap_plots.py

# 记录结束时间
echo "Analysis completed at $(date)" >> 08_logs/training_log.txt

echo "All analyses completed. Check results in 04_results/ directory." 