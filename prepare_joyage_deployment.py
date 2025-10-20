#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_joyage_deployment.py
- 使用14_final_eval_s7.py生成的模型创建joyage web部署包
- 支持自动复制模型文件和其他必要组件
- 创建完整的部署文件夹结构
"""

import os
import sys
import shutil
import joblib
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

def create_deployment_structure(target_dir):
    """创建部署所需的基本目录结构"""
    # 创建主目录
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建saved_models目录
    models_dir = target_dir / "saved_models"
    models_dir.mkdir(exist_ok=True)
    
    # 确保cesd_depression_model目录存在
    module_dir = target_dir / "cesd_depression_model"
    module_dir.mkdir(exist_ok=True)
    
    # 创建.streamlit目录
    streamlit_dir = target_dir / ".streamlit"
    streamlit_dir.mkdir(exist_ok=True)
    
    return {
        "root": target_dir,
        "models": models_dir,
        "module": module_dir,
        "streamlit": streamlit_dir
    }

def copy_joyage_structure(source_dir, target_dir):
    """从源joyage项目复制文件结构（除了模型文件）"""
    print(f"从 {source_dir} 复制项目结构到 {target_dir}...")
    
    # 需要复制的目录和文件
    dirs_to_copy = ["cesd_depression_model", ".streamlit"]
    files_to_copy = ["cesd_web_app_bilingual.py", "requirements.txt", 
                    "DEPLOYMENT_GUIDE.md", "README.md", "model_downloader.py"]
    
    # 复制目录（除了saved_models）
    for dirname in dirs_to_copy:
        src = source_dir / dirname
        dst = target_dir / dirname
        
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"- 已复制目录: {dirname}")
        else:
            print(f"- 警告: 目录不存在，跳过: {dirname}")
    
    # 复制根目录文件
    for filename in files_to_copy:
        src = source_dir / filename
        dst = target_dir / filename
        
        if src.exists():
            shutil.copy2(src, dst)
            print(f"- 已复制文件: {filename}")
        else:
            print(f"- 警告: 文件不存在，跳过: {filename}")

def prepare_and_copy_model(model_path, target_dirs, model_name="cesd_model_best_latest.joblib"):
    """准备模型并复制到目标目录"""
    print(f"\n正在处理模型: {model_path}")
    
    try:
        # 加载源模型
        model = joblib.load(model_path)
        print("- 模型加载成功")
        
        # 保存到各个目标位置
        for name, dir_path in target_dirs.items():
            save_path = dir_path / model_name
            joblib.dump(model, save_path)
            print(f"- 已保存到: {save_path}")
        
        return True
    except Exception as e:
        print(f"❌ 模型处理失败: {e}")
        return False

def create_data_processor(source_processor, target_dirs):
    """创建或复制数据处理器"""
    processor_name = "data_processor.joblib"
    
    if source_processor and Path(source_processor).exists():
        # 复制现有处理器
        try:
            processor = joblib.load(source_processor)
            print("- 数据处理器加载成功")
            
            for name, dir_path in target_dirs.items():
                save_path = dir_path / processor_name
                joblib.dump(processor, save_path)
                print(f"- 已保存到: {save_path}")
            
            return True
        except Exception as e:
            print(f"❌ 数据处理器复制失败: {e}")
    else:
        print("⚠️ 未提供数据处理器，将在部署前需要补充此文件")
    
    return False

def main():
    parser = argparse.ArgumentParser(description="准备joyage web部署包")
    parser.add_argument("--model_path", required=True, help="14_final_eval_s7.py生成的模型路径（RF或其他校准模型）")
    parser.add_argument("--joyage_source", required=True, help="原始joyage项目路径，用于复制文件结构")
    parser.add_argument("--processor_path", default=None, help="数据处理器路径（如果有）")
    parser.add_argument("--output_dir", default=None, help="输出目录，默认为./joyage_deployment_YYYYMMDD_HHMMSS")
    args = parser.parse_args()
    
    # 检查模型文件
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"错误: 模型文件不存在: {model_path}")
        return 1
    
    # 检查joyage源目录
    joyage_src = Path(args.joyage_source)
    if not joyage_src.exists() or not (joyage_src / "cesd_depression_model").exists():
        print(f"错误: joyage源目录无效: {joyage_src}")
        return 1
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(f"joyage_deployment_{timestamp}")
    
    print(f"\n=== 开始准备joyage部署包 ===")
    print(f"目标目录: {output_dir}")
    
    # 创建部署结构
    dirs = create_deployment_structure(output_dir)
    
    # 从源joyage项目复制文件结构
    copy_joyage_structure(joyage_src, output_dir)
    
    # 处理模型文件
    model_dirs = {
        "main": dirs["models"]
    }
    model_success = prepare_and_copy_model(model_path, model_dirs)
    
    # 处理数据处理器
    processor_success = False
    if args.processor_path:
        processor_success = create_data_processor(args.processor_path, model_dirs)
    else:
        # 尝试从源joyage项目复制数据处理器
        source_processor = joyage_src / "saved_models" / "data_processor.joblib"
        if source_processor.exists():
            print(f"\n找到源数据处理器，正在复制...")
            processor_success = create_data_processor(source_processor, model_dirs)
        else:
            print("\n⚠️ 未找到数据处理器，需在部署前补充")
    
    print("\n=== 部署包准备完成 ===")
    print(f"位置: {output_dir}")
    print(f"模型处理: {'✓' if model_success else '❌'}")
    print(f"数据处理器: {'✓' if processor_success else '❌'}")
    
    if not processor_success:
        print("\n⚠️ 警告: 部署包可能不完整，缺少数据处理器")
        print("请在使用前确保saved_models目录包含data_processor.joblib文件")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
