#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
转换14_final_eval_s7.py生成的模型为joyage web应用所需格式
"""

import os
import sys
import shutil
import joblib
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="转换并复制模型到joyage项目")
    parser.add_argument("--model_path", required=True, help="14_final_eval_s7.py生成的模型路径（例如：RF_calibrated_isotonic.joblib）")
    parser.add_argument("--joyage_dir", default=None, help="joyage项目目录，默认为同级目录的joyage-depression-assessment-main")
    args = parser.parse_args()
    
    # 处理路径
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"错误：模型文件不存在: {model_path}")
        return 1
    
    # 确定joyage项目目录
    if args.joyage_dir:
        joyage_dir = Path(args.joyage_dir)
    else:
        # 默认在当前脚本目录的同级找joyage目录
        joyage_dir = Path(__file__).parent / "project_name" / "joyage-depression-assessment-main"
    
    # 检查目录是否存在
    if not joyage_dir.exists():
        print(f"错误：joyage项目目录不存在: {joyage_dir}")
        return 1
    
    # joyage主目录和lite目录下的模型目标路径
    main_models_dir = joyage_dir / "saved_models"
    lite_models_dir = joyage_dir / "extracted_joyage_lite" / "saved_models"
    
    # 确保目录存在
    main_models_dir.mkdir(exist_ok=True)
    lite_models_dir.mkdir(exist_ok=True)
    
    # 加载源模型
    print(f"正在加载模型：{model_path}")
    try:
        model = joblib.load(model_path)
        print("模型加载成功！")
    except Exception as e:
        print(f"错误：无法加载模型: {e}")
        return 1
    
    # 重新保存为joyage格式
    main_model_path = main_models_dir / "cesd_model_best_latest.joblib"
    lite_model_path = lite_models_dir / "cesd_model_best_latest.joblib"
    
    print(f"正在保存模型到: {main_model_path}")
    joblib.dump(model, main_model_path)
    
    print(f"正在保存模型到: {lite_model_path}")
    joblib.dump(model, lite_model_path)
    
    print("模型转换和复制完成！")
    
    # 检查是否需要生成数据处理器（如果不存在）
    if not (main_models_dir / "data_processor.joblib").exists():
        print("\n警告：data_processor.joblib不存在，请确保在joyage项目中有有效的数据处理器文件")
        print("提示：您可能需要从原始joyage项目复制data_processor.joblib文件")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
