#!/usr/bin/env python3
"""
模型文件下载器 - 用于Streamlit Cloud部署
如果模型文件不存在，自动从指定位置下载
"""

import os
import requests
import streamlit as st
from pathlib import Path

def download_model_files():
    """下载必需的模型文件"""
    
    # 模型文件列表（如果需要从远程下载）
    model_files = {
        'cesd_model_best_latest.joblib': 'https://your-storage-url/cesd_model_best_latest.joblib',
        'data_processor.joblib': 'https://your-storage-url/data_processor.joblib'
    }
    
    saved_models_dir = Path('saved_models')
    saved_models_dir.mkdir(exist_ok=True)
    
    for filename, url in model_files.items():
        file_path = saved_models_dir / filename
        
        if not file_path.exists():
            st.info(f"正在下载模型文件: {filename}")
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        
                st.success(f"✅ 模型文件下载完成: {filename}")
                
            except Exception as e:
                st.error(f"❌ 下载失败 {filename}: {str(e)}")
                raise
    
    return True

def ensure_models_available():
    """确保模型文件可用"""
    saved_models_dir = Path('saved_models')
    
    # 检查是否存在任何模型文件
    model_files = list(saved_models_dir.glob('*.joblib'))
    
    if not model_files:
        st.warning("未找到模型文件，尝试下载...")
        download_model_files()
    
    return True

if __name__ == "__main__":
    ensure_models_available()
    print("模型文件检查完成") 