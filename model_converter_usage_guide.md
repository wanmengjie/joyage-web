# 模型转换与部署指南

## 概述

本文档说明如何将您使用14_final_eval_s7.py脚本生成的模型转换并部署到JoyAge抑郁风险评估平台。

## 准备工作

1. 首先运行14_final_eval_s7.py生成模型文件
2. 确定要使用的模型（如随机森林、XGBoost等）
3. 找到对应的校准模型文件，例如:
   - `10_experiments/[版本号]/final_eval_s7/models/RF_calibrated_isotonic.joblib`

## 方法1：使用模型转换脚本

我们提供了一个简单的脚本`convert_model_for_joyage.py`，可以直接将模型复制到JoyAge项目中。

### 使用方法

```bash
python convert_model_for_joyage.py --model_path "路径/到/RF_calibrated_isotonic.joblib" --joyage_dir "路径/到/joyage-depression-assessment-main"
```

参数说明:
- `--model_path`：14_final_eval_s7.py生成的模型路径（必需）
- `--joyage_dir`：JoyAge项目目录（可选，默认在当前目录下寻找）

### 注意事项

- 此脚本只替换模型文件，不会修改其他文件
- 请确保JoyAge项目中有有效的数据处理器文件（data_processor.joblib）
- 如果替换后应用无法正常运行，可能需要额外的数据处理步骤

## 方法2：创建完整部署包

对于更完整的解决方案，我们提供了`prepare_joyage_deployment.py`脚本，它会创建一个完整的部署包。

### 使用方法

```bash
python prepare_joyage_deployment.py --model_path "路径/到/RF_calibrated_isotonic.joblib" --joyage_source "路径/到/joyage-depression-assessment-main" --output_dir "输出目录路径"
```

参数说明:
- `--model_path`：14_final_eval_s7.py生成的模型路径（必需）
- `--joyage_source`：原始JoyAge项目目录，用于复制文件结构（必需）
- `--processor_path`：可选的数据处理器路径
- `--output_dir`：输出目录，默认为./joyage_deployment_[时间戳]

### 优点

- 创建完整的部署目录结构
- 复制所有必要的文件和目录
- 自动处理模型和数据处理器

## 常见问题

1. **模型不兼容怎么办？**
   
   如果直接替换的模型不兼容，建议使用第27号脚本(27_train_and_export_web_model.py)重新训练一个专门用于网页应用的模型。

   ```bash
   python 03_scripts/27_train_and_export_web_model.py --train "训练集.csv" --val "验证集.csv" --outdir "./web_model"
   ```

2. **如何确认模型已正确部署？**

   使用Streamlit本地运行测试:
   
   ```bash
   cd joyage_deployment_[时间戳]
   pip install -r requirements.txt
   streamlit run cesd_web_app_bilingual.py
   ```

3. **如何处理特征不一致问题？**

   确保您的模型使用的特征与JoyAge项目中定义的42个特征一致。如果不一致，可能需要修改`cesd_depression_model/config.py`中的特征定义。

## 其他说明

- JoyAge项目使用的是单一最佳模型文件，命名为`cesd_model_best_latest.joblib`
- 数据处理器文件名为`data_processor.joblib`
- 如需自定义配置，可修改`.streamlit/config.toml`文件
