"""
数据映射常量管理器 - 统一管理数据转换映射
"""

import pandas as pd
import numpy as np

class DataMappings:
    """数据映射常量管理，提供高效的向量化转换方法"""
    
    # KLOSA到CHARLS的adlfive变量映射
    ADLFIVE_MAPPING = {
        0: '0.Fully Independent',
        1: '1.Mild Dependence', 
        2: '2.Moderate Dependence',
        3: '3.Significant Dependence',
        4: '4.Severe Dependence',
        5: '5.Total Dependence'
    }
    
    # 目标变量映射
    DEPRESSION_MAPPING = {
        '0.No': 0,
        '1.Yes': 1,
        0: 0,
        1: 1
    }
    
    # 性别变量映射
    GENDER_MAPPING = {
        '1.man': 0,
        '2.woman': 1,
        1: 0,
        2: 1
    }
    
    @classmethod
    def convert_adlfive_klosa_to_charls(cls, series):
        """
        高效的向量化转换KLOSA的adlfive数值为CHARLS格式
        
        参数:
        ----
        series : pd.Series
            KLOSA的adlfive数据列
            
        返回:
        ----
        pd.Series : 转换后的数据列
        """
        return series.map(cls.ADLFIVE_MAPPING).fillna(series)
    
    @classmethod
    def convert_depression_to_numeric(cls, series):
        """
        将抑郁症目标变量转换为数值格式
        
        参数:
        ----
        series : pd.Series
            抑郁症变量数据列
            
        返回:
        ----
        pd.Series : 转换后的数值数据列
        """
        return series.map(cls.DEPRESSION_MAPPING).fillna(series)
    
    @classmethod
    def convert_gender_to_numeric(cls, series):
        """
        将性别变量转换为数值格式
        
        参数:
        ----
        series : pd.Series
            性别变量数据列
            
        返回:
        ----
        pd.Series : 转换后的数值数据列
        """
        return series.map(cls.GENDER_MAPPING).fillna(series)
    
    @classmethod
    def apply_mapping_safely(cls, series, mapping_dict, default_value=None):
        """
        安全地应用映射，处理缺失值和异常情况
        
        参数:
        ----
        series : pd.Series
            要转换的数据列
        mapping_dict : dict
            映射字典
        default_value : any, optional
            默认值，如果映射失败则使用此值
            
        返回:
        ----
        pd.Series : 转换后的数据列
        """
        if default_value is not None:
            return series.map(mapping_dict).fillna(default_value)
        else:
            return series.map(mapping_dict).fillna(series)
    
    @classmethod
    def get_unique_unmapped_values(cls, series, mapping_dict):
        """
        获取无法映射的唯一值，用于调试
        
        参数:
        ----
        series : pd.Series
            数据列
        mapping_dict : dict
            映射字典
            
        返回:
        ----
        list : 无法映射的唯一值列表
        """
        unique_values = series.dropna().unique()
        unmapped = [val for val in unique_values if val not in mapping_dict]
        return unmapped
    
    @classmethod
    def validate_mapping_completeness(cls, series, mapping_dict, column_name=""):
        """
        验证映射的完整性
        
        参数:
        ----
        series : pd.Series
            数据列
        mapping_dict : dict
            映射字典
        column_name : str
            列名（用于错误信息）
            
        返回:
        ----
        bool : 映射是否完整
        """
        unmapped = cls.get_unique_unmapped_values(series, mapping_dict)
        
        if unmapped:
            print(f"⚠️ {column_name} 存在无法映射的值: {unmapped}")
            return False
        else:
            print(f"✅ {column_name} 映射完整")
            return True 