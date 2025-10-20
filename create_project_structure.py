import os
import pathlib

# 项目根目录
project_root = "project_name"

# 主要目录
main_dirs = [
    "00_docs",
    "01_raw_data",
    "02_processed_data",
    "03_scripts",
    "04_results",
    "05_supplement",
    "06_lit_review",
    "07_config",
    "08_logs",
    "09_deployment",
    "10_experiments"
]

# 子目录
sub_dirs = {
    "02_processed_data": ["logs"],
    "04_results": ["tables", "figures", "reports"],
    "05_supplement": ["extra_figures"],
    "09_deployment": ["screenshots"],
    "10_experiments": ["sensitivity", "ablation", "fairness", "threshold", "results"]
}

# 脚本文件
script_files = [
    "00_utils.py",
    "01_cleaning.py",
    "02_imputation.py",
    "03_feature_engineering.py",
    "04_baseline_analysis.py",
    "05a_hyperparam_search.py",
    "05_model_training.py",
    "06_validation.py",
    "07_metrics_ci.py",
    "08_sensitivity.py",
    "shap_plots.py",
    "run_all.sh"
]

# README内容模板
readme_template = """# {dir_name}

{description}

## 内容

{content}
"""

# 目录描述
dir_descriptions = {
    "00_docs": "研究文档，包括研究笔记、会议纪要、决策记录等。",
    "01_raw_data": "原始数据存储，包括CHARLS和KLOSA数据集。",
    "02_processed_data": "处理后的数据，包括清洗、插补和特征工程后的数据。",
    "03_scripts": "分析脚本，包括数据处理、模型训练、评估等。",
    "04_results": "分析结果，包括表格、图形和报告。",
    "05_supplement": "补充材料，包括额外的表格和图形。",
    "06_lit_review": "文献综述相关材料。",
    "07_config": "配置文件，包括模型参数、随机种子等。",
    "08_logs": "运行日志，记录训练和验证过程。",
    "09_deployment": "部署相关文件，包括Web应用。",
    "10_experiments": "实验文件，包括敏感性分析、消融实验等。"
}

# 创建目录结构
def create_structure():
    # 创建项目根目录
    os.makedirs(project_root, exist_ok=True)
    
    # 创建主要目录
    for dir_name in main_dirs:
        dir_path = os.path.join(project_root, dir_name)
        os.makedirs(dir_path, exist_ok=True)
        
        # 创建README.md
        readme_path = os.path.join(dir_path, "README.md")
        description = dir_descriptions.get(dir_name, "该目录的描述。")
        content = ""
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_template.format(
                dir_name=dir_name,
                description=description,
                content=content
            ))
        
        # 创建子目录
        if dir_name in sub_dirs:
            for sub_dir in sub_dirs[dir_name]:
                sub_dir_path = os.path.join(dir_path, sub_dir)
                os.makedirs(sub_dir_path, exist_ok=True)
    
    # 创建脚本文件
    scripts_dir = os.path.join(project_root, "03_scripts")
    for script_file in script_files:
        script_path = os.path.join(scripts_dir, script_file)
        pathlib.Path(script_path).touch()
    
    # 创建项目根目录的README.md
    root_readme_path = os.path.join(project_root, "README.md")
    with open(root_readme_path, 'w', encoding='utf-8') as f:
        f.write("""# 多疾病预测系统

基于CHARLS数据集的15种疾病同时预测系统，使用多标签学习方法。

## 项目概述

本项目使用CHARLS(中国健康与养老追踪调查)数据集构建多疾病预测模型，可以同时预测15种常见疾病的风险。使用KLOSA(韩国老年人纵向研究)数据集进行外部验证。

## 目录结构

- `00_docs/`: 研究文档
- `01_raw_data/`: 原始数据
- `02_processed_data/`: 处理后的数据
- `03_scripts/`: 分析脚本
- `04_results/`: 分析结果
- `05_supplement/`: 补充材料
- `06_lit_review/`: 文献综述
- `07_config/`: 配置文件
- `08_logs/`: 运行日志
- `09_deployment/`: 部署文件
- `10_experiments/`: 实验文件
""")
    
    print("项目结构创建完成！")

if __name__ == "__main__":
    create_structure() 