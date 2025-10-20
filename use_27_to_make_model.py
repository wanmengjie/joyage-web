#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
use_27_to_make_model.py
- 使用第27号脚本(27_train_and_export_web_model.py)创建web适用的模型
- 专为joyage项目准备，确保完全兼容
"""

import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path

def run_script_27(script_path, train_path, val_path, output_dir, test_path=None):
    """运行27_train_and_export_web_model.py脚本生成模型"""
    # 构建命令
    cmd = [
        sys.executable,
        str(script_path),
        "--train", str(train_path),
        "--val", str(val_path),
        "--outdir", str(output_dir)
    ]
    
    if test_path:
        cmd.extend(["--test", str(test_path)])
    
    # 执行命令
    print(f"运行命令: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("脚本执行成功:")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"脚本执行失败:")
        print(f"错误代码: {e.returncode}")
        print(f"标准输出: {e.stdout}")
        print(f"错误输出: {e.stderr}")
        return False

def copy_model_to_joyage(source_dir, joyage_dir):
    """复制生成的模型文件到joyage项目"""
    # 检查源文件
    model_src = source_dir / "final_rf_pipeline.joblib"
    if not model_src.exists():
        print(f"错误: 模型文件不存在: {model_src}")
        return False
    
    # joyage目录中的目标路径
    target_paths = [
        joyage_dir / "saved_models" / "cesd_model_best_latest.joblib",
        joyage_dir / "extracted_joyage_lite" / "saved_models" / "cesd_model_best_latest.joblib"
    ]
    
    # 创建目录并复制文件
    success = True
    for target_path in target_paths:
        try:
            # 确保目录存在
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 复制文件
            shutil.copy2(model_src, target_path)
            print(f"已复制到: {target_path}")
        except Exception as e:
            print(f"复制到 {target_path} 失败: {e}")
            success = False
    
    return success

def main():
    parser = argparse.ArgumentParser(description="使用27号脚本生成web适用的模型并部署到joyage项目")
    parser.add_argument("--train", required=True, help="训练集CSV路径")
    parser.add_argument("--val", required=True, help="验证集CSV路径")
    parser.add_argument("--test", default=None, help="测试集CSV路径（可选）")
    parser.add_argument("--scripts_dir", default=None, help="脚本目录，默认为./project_name/03_scripts")
    parser.add_argument("--joyage_dir", default=None, help="joyage项目目录，默认为./project_name/joyage-depression-assessment-main")
    args = parser.parse_args()
    
    # 检查训练和验证集
    train_path = Path(args.train)
    val_path = Path(args.val)
    if not train_path.exists() or not val_path.exists():
        print("错误: 训练集或验证集文件不存在")
        return 1
    
    # 检查测试集（如果提供）
    test_path = None
    if args.test:
        test_path = Path(args.test)
        if not test_path.exists():
            print(f"警告: 测试集文件不存在: {test_path}")
            test_path = None
    
    # 确定脚本和joyage目录
    if args.scripts_dir:
        scripts_dir = Path(args.scripts_dir)
    else:
        scripts_dir = Path(__file__).parent / "project_name" / "03_scripts"
    
    if args.joyage_dir:
        joyage_dir = Path(args.joyage_dir)
    else:
        joyage_dir = Path(__file__).parent / "project_name" / "joyage-depression-assessment-main"
    
    # 检查目录和脚本
    script_path = scripts_dir / "27_train_and_export_web_model.py"
    if not script_path.exists():
        print(f"错误: 27号脚本不存在: {script_path}")
        return 1
    
    if not joyage_dir.exists():
        print(f"警告: joyage项目目录不存在: {joyage_dir}")
    
    # 创建临时输出目录
    output_dir = Path(__file__).parent / "web_model_output"
    output_dir.mkdir(exist_ok=True)
    
    print("\n=== 开始使用27号脚本生成web模型 ===")
    
    # 运行27号脚本
    success = run_script_27(script_path, train_path, val_path, output_dir, test_path)
    if not success:
        print("模型生成失败")
        return 1
    
    print("\n=== 模型生成成功，正在复制到joyage项目 ===")
    
    # 复制模型到joyage项目
    if joyage_dir.exists():
        copy_success = copy_model_to_joyage(output_dir, joyage_dir)
        if copy_success:
            print("\n✓ 模型已成功复制到joyage项目")
        else:
            print("\n⚠ 复制模型到joyage项目时出现问题")
    else:
        print(f"\n⚠ 未复制模型：joyage项目目录不存在")
    
    print("\n生成的模型文件保存在: {output_dir}")
    print("您可以手动复制以下文件到joyage项目:")
    print(f"- {output_dir}/final_rf_pipeline.joblib → saved_models/cesd_model_best_latest.joblib")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
