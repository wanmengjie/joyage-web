# 01_raw_data

原始数据存储，包括CHARLS和KLOSA数据集。

## 数据文件

1. **CHARLS数据集**: 
   - 文件名: `charls2018 20250906.csv`
   - 描述: 中国健康与养老追踪调查(China Health and Retirement Longitudinal Study)2018年数据
   - 用途: 模型训练和内部验证
   - 更新日期: 2025-09-06

2. **KLOSA数据集**:
   - 文件名: `klosa2018 20250722 - 20250905.csv`
   - 描述: 韩国老年人纵向研究(Korean Longitudinal Study of Aging)2018年数据
   - 用途: 外部验证
   - 更新日期: 2025-09-05

## 数据字典

数据字典和变量说明请参考 `00_docs/data_dictionary.pdf`。

## 注意事项

- 这些原始数据文件不应被直接修改
- 数据预处理和清洗的结果将保存在 `02_processed_data/` 目录中
- 请确保数据处理过程中的所有变更都有记录，以确保研究的可重复性


