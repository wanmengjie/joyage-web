# 错误记录与修正方法

## Streamlit应用运行记录

**Date**: 2025年1月31日

### 当前任务
运行CESD抑郁风险评估平台 - 双语Web应用

### 系统状态
- 工作目录: joyage-depression-assessment
- Python应用: cesd_web_app_bilingual v2.py
- 记忆系统: 已初始化

---

## 错误记录

**Mistake: 直接用python运行Streamlit应用**

**Wrong**:
```bash
python "cesd_web_app_bilingual v2.py"
```

**Correct**:
```bash
streamlit run "cesd_web_app_bilingual v2.py"
```

**错误说明**: 
直接用Python运行Streamlit应用会产生ScriptRunContext警告，因为Streamlit需要通过其专用的运行环境启动。警告信息"missing ScriptRunContext"表示缺少Streamlit运行上下文。

**解决方案**: 
使用`streamlit run`命令启动应用，这样才能正确初始化Streamlit的运行环境和会话状态。

---

**Mistake: 在错误目录运行Streamlit应用**

**Wrong**:
```bash
# 在根目录运行
streamlit run "cesd_web_app_bilingual v2.py"
# Error: File does not exist: cesd_web_app_bilingual v2.py
```

**Correct**:
```bash
# 先进入正确目录
cd joyage_lite_deployment_20250809_190725
streamlit run "cesd_web_app_bilingual v2.py"
```

**错误说明**: 
文件位于子目录中，但在根目录运行命令导致找不到文件。

**解决方案**: 
确保在包含Python文件的正确目录中运行streamlit命令。

---

**Mistake: Streamlit配置冲突**

**错误信息**:
```
Warning: the config option 'server.enableCORS=false' is not compatible with 'server.enableXsrfProtection=true'
```

**解决方案**: 
使用简化的启动命令，避免复杂的配置选项冲突。

---

**Mistake: Requirements文件重复和版本冲突**

**Wrong**:
```
# 存在两个冲突的requirements文件
requirements.txt (使用git版本的shap，不稳定)
requirements_web.txt (稳定版本但文件名错误)
```

**Correct**:
```
# 整合为单一的requirements.txt
# 使用稳定版本号，添加版本上限
# 避免使用git仓库版本的包
```

**错误说明**: 
- 重复的requirements文件会导致Streamlit Cloud部署混淆
- 使用git版本的shap包在生产环境不稳定
- 缺少版本上限可能导致依赖冲突

**解决方案**: 
1. 删除requirements_web.txt
2. 整合为统一的requirements.txt
3. 使用稳定版本号并添加上限
4. 避免git仓库依赖 