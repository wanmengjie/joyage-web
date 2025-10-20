# JoyAge 抑郁风险评估平台 - 轻量版部署包

## 📋 包含文件
- cesd_web_app_bilingual.py - 主应用文件
- requirements.txt - Python依赖包
- model_downloader.py - 模型下载器（备用）
- DEPLOYMENT_GUIDE.md - 详细部署指南
- .streamlit/config.toml - Streamlit配置
- saved_models/ - 关键模型文件（已优化）
- cesd_depression_model/ - 核心模块

## 🚀 快速部署步骤
1. 将整个文件夹上传到GitHub仓库
2. 在Streamlit Cloud中选择该仓库
3. 主文件设置为: cesd_web_app_bilingual.py
4. 完成部署

## ⚠️ 轻量版说明
此版本只包含运行必需的模型文件，以确保：
- GitHub上传速度快
- Streamlit Cloud部署成功
- 应用运行正常

## 📊 打包信息
- 打包时间: 2025-08-09 19:07:25
- 打包类型: 轻量版（仅关键文件）
- 打包脚本: create_deployment_package_lite.py

详细部署说明请查看 DEPLOYMENT_GUIDE.md
