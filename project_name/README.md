# 多疾病预测系统

基于CHARLS数据集的15种疾病同时预测系统，使用多标签学习方法。

## 项目概述

本项目使用CHARLS(中国健康与养老追踪调查)数据集构建多疾病预测模型，可以同时预测15种常见疾病的风险。使用KLOSA(韩国老年人纵向研究)数据集进行外部验证。

## 目录结构

project_name/
│
├── 00_docs/                         
│   ├── project_notes.md              # 研究笔记/会议纪要/决策记录
│   ├── data_dictionary.pdf           # CHARLS/KLOSA 数据字典
│   ├── analysis_plan.md              # 分析方案 (Methods 对应的SOP)
│   └── references.bib                # 文献库 (RIS/BibTex/EndNote)
│
├── 01_raw_data/                     
│   ├── CHARLS_2018.dta
│   ├── KLOSA_2018.dta
│   └── readme.txt                    # 数据来源/下载日期/版本号
│
├── 02_processed_data/               
│   ├── baseline_clean.csv            # 清洗后数据
│   ├── imputed.csv                   # 缺失值插补数据
│   ├── analysis_ready.csv            # 建模数据 (标准化/哑变量化)
│   └── logs/                         
│       ├── missing_report.xlsx       # 缺失率/删除样本记录
│       └── baseline_distribution.pdf # 基线分布可视化
│
├── 03_scripts/                      
│   ├── 00_utils.py                   # 公共函数 (绘图、评估、CI计算)
│   ├── 01_cleaning.py                # 数据清理 (异常值/逻辑冲突)
│   ├── 02_imputation.py              # 缺失值插补 (MICE/众数/中位数)
│   ├── 03_feature_engineering.py     # 特征工程 (标准化/哑变量化)
│   ├── 04_baseline_analysis.py       # 基线描述 (Table 1)
│   ├── 05a_hyperparam_search.py      # 超参数调优 (CV/随机/贝叶斯搜索)
│   ├── 05_model_training.py          # 最终模型训练 (训练+内部验证)
│   ├── 06_validation.py              # 外部验证 (KLOSA)
│   ├── 07_metrics_ci.py              # 各类指标+95%CI计算
│   ├── 08_sensitivity.py             # 敏感性分析入口
│   ├── shap_plots.py                 # SHAP解释性分析 (全局+个体)
│   └── run_all.sh                    # 一键运行pipeline
│
├── 04_results/                      
│   ├── tables/
│   │   ├── training_metrics.xlsx       # 训练集指标 (均值+95%CI)
│   │   ├── cv_metrics_charls.xlsx      # 内部验证指标 (均值+95%CI)
│   │   ├── external_metrics_klosa.xlsx # 外部验证指标 (点估计+95%CI)
│   │   ├── model_compare_charls.xlsx   # 多模型对比 (含95%CI)
│   │   ├── threshold_metrics.xlsx      # 不同cutoff下的阈值指标
│   │   └── tuning_summary.xlsx         # 调优过程汇总 (Top-k参数)
│   │
│   ├── tuning/
│   │   ├── search_log.csv              # 调优日志 (参数+分数)
│   │   └── search_space.yaml           # 搜索空间定义
│   │
│   ├── figures/
│   │   ├── roc_curves.pdf              # ROC曲线 (训练/内部/外部)
│   │   ├── pr_curves.pdf               # PR曲线
│   │   ├── calibration_curves.pdf      # 校准曲线
│   │   ├── dca_curves.pdf              # 决策曲线 (DCA)
│   │   ├── shap_beeswarm.png           # SHAP beeswarm图
│   │   ├── shap_bar.png                # SHAP条形图
│   │   └── patient_explanations/       # 个体化SHAP解释 (force/waterfall)
│   │
│   └── reports/                        # 自动生成的结果报告
│       └── model_report.html
│
├── 05_supplement/                    
│   ├── table_s1_missingness.xlsx       # 缺失值表
│   ├── table_sx_metrics.xlsx           # 全指标+95%CI汇总 (训练/验证/外部)
│   ├── figure_s1_flowchart.png         # 样本筛选流程图
│   ├── tripid_ai_checklist.docx        # TRIPOD-AI检查表
│   ├── methods_details.docx            # 方法学细节 (CI算法等)
│   └── extra_figures/                  # 补充图表 (敏感性/消融/非线性)
│
├── 06_lit_review/                    
│   ├── search_strategy_pubmed.txt      # 文献检索式
│   ├── reference_list.ris              # RIS文件 (EndNote/Zotero导入)
│   └── top_papers_notes.md             # 核心文献笔记
│
├── 07_config/                        
│   ├── requirements.txt                # Python包版本
│   ├── config.yaml                     # 模型参数配置
│   └── seeds.txt                       # 随机种子
│
├── 08_logs/                          
│   ├── training_log.txt                # 训练日志
│   └── validation_log.txt              # 验证日志
│
├── 09_deployment/                    
│   ├── app.py                          # Streamlit主程序
│   ├── requirements_app.txt            # 部署环境
│   ├── screenshots/                    # 工具截图 (Supplement)
│   └── README_app.md                   # 部署说明
│
├── 10_experiments/                   
│   ├── sensitivity/                    
│   │   ├── cesd_cutoffs.py             # CES-D cutoff敏感性
│   │   ├── missing_data_methods.py     # 缺失值处理对比
│   │   ├── subgroup_analysis.py        # 亚组分析 (性别/年龄/城乡/教育)
│   │   └── tautology_check.py          # 排除与结局高度相关变量
│   │
│   ├── ablation/                       
│   │   ├── single_feature_ablation.py  # 单变量消融
│   │   ├── domain_ablation.py          # 域级消融
│   │   └── feature_ranking_compare.py  # SHAP vs permutation 排名
│   │
│   ├── fairness/                       # 公平性分析
│   │   └── subgroup_calibration.py     # 不同人群的校准/性能
│   │
│   ├── threshold/                      # 阈值优化
│   │   └── cutoff_analysis.py          # Cutoff下的敏感性/特异性/PPV/NPV
│   │
│   └── results/                        
│       ├── sensitivity_tables.xlsx
│       ├── ablation_tables.xlsx
│       ├── fairness_tables.xlsx
│       ├── threshold_tables.xlsx
│       ├── sensitivity_figures.pdf
│       ├── ablation_figures.pdf
│       ├── fairness_figures.pdf
│       └── threshold_figures.pdf
│
└── README.md                           # 项目说明文档
