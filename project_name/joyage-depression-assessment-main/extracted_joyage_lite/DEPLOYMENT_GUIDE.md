# JoyAge 抑郁风险评估平台 - Streamlit Cloud 部署指南

## 📋 部署准备清单

### ✅ 必需文件已创建
- `cesd_web_app_bilingual.py` - 主应用文件
- `requirements.txt` - Python依赖包
- `.streamlit/config.toml` - Streamlit配置
- `saved_models/` - 模型文件目录
- `cesd_depression_model/` - 核心模块

### 📁 项目结构
```
20250731/
├── cesd_web_app_bilingual.py     # 主应用
├── requirements.txt              # 依赖包
├── model_downloader.py          # 模型下载器(备用)
├── .streamlit/
│   └── config.toml              # Streamlit配置
├── saved_models/
│   ├── cesd_model_best_latest.joblib
│   └── data_processor.joblib
└── cesd_depression_model/       # 核心模块
    ├── __init__.py
    ├── config.py
    └── ...
```

## 🚀 部署步骤

### 第1步：GitHub仓库准备
1. **创建新的GitHub仓库**
   ```bash
   # 在GitHub上创建名为 "joyage-depression-assessment" 的仓库
   ```

2. **上传项目文件**
   - 主应用文件：`cesd_web_app_bilingual.py`
   - 依赖文件：`requirements.txt`
   - 配置文件：`.streamlit/config.toml` 
   - 模型文件：`saved_models/` 目录下的所有 `.joblib` 文件
   - 核心模块：`cesd_depression_model/` 整个目录

### 第2步：Streamlit Cloud部署
1. **访问 Streamlit Cloud**
   - 前往：https://share.streamlit.io
   - 使用GitHub账号登录

2. **新建应用**
   - 点击 "New app"
   - 选择刚创建的GitHub仓库
   - **主文件路径**：`cesd_web_app_bilingual.py`
   - **分支**：`main` 或 `master`

3. **高级设置**（可选）
   - Python版本：3.9 或 3.10
   - 环境变量：无需特殊配置

### 第3步：部署验证
部署完成后，检查以下功能：
- ✅ 页面加载正常
- ✅ 三语言切换（中/英/韩）
- ✅ 模型加载成功
- ✅ 预测功能正常
- ✅ SHAP解释图显示

## 🔧 常见问题与解决方案

### 问题1：模型文件过大
**解决方案**：
- 当前模型文件较小（<1MB），可直接上传
- 如需要，使用Git LFS：
  ```bash
  git lfs track "*.joblib"
  git add .gitattributes
  ```

### 问题2：依赖包安装失败
**解决方案**：
- 检查 `requirements.txt` 版本兼容性
- 简化版本号（去掉具体版本，使用 `>=`）

### 问题3：内存不足
**解决方案**：
- 优化模型加载（使用 `@st.cache_resource`）
- 减少同时加载的模型数量

### 问题4：模型文件路径错误
**解决方案**：
- 确保相对路径正确
- 使用 `Path(__file__).parent` 获取相对路径

## 📊 性能优化建议

### 1. 缓存策略
```python
@st.cache_resource
def load_models():
    # 模型加载逻辑
    pass

@st.cache_data
def process_data(data):
    # 数据处理逻辑
    pass
```

### 2. 分步加载
- 主模型优先加载
- SHAP解释器按需加载
- 减少启动时间

### 3. 错误处理
- 添加模型加载失败的备选方案
- 提供清晰的错误提示信息

## 🌐 访问地址
部署成功后，您的应用将可通过以下地址访问：
```
https://your-app-name.streamlit.app
```

## 📝 部署后检查清单
- [ ] 应用正常启动
- [ ] 所有页面功能正常
- [ ] 多语言切换正常
- [ ] 模型预测准确
- [ ] SHAP解释正常显示
- [ ] 响应速度合理
- [ ] 移动端适配良好

## 🔄 更新部署
要更新应用：
1. 推送代码到GitHub仓库
2. Streamlit Cloud会自动重新部署
3. 通常需要2-5分钟完成

## 📞 技术支持
如遇到部署问题，可以：
- 查看Streamlit Cloud的部署日志
- 检查GitHub仓库的文件完整性
- 验证requirements.txt的依赖版本

---
**祝您部署顺利！🎉** 