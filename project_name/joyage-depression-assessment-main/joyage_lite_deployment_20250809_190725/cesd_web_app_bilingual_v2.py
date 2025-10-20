#!/usr/bin/env python3
"""
CESD抑郁预测模型 双语Web应用 (中英文切换)
实现个人预测 + SHAP解释可视化 + 完整42个特征
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from cesd_depression_model.utils.helpers import load_model
from cesd_depression_model.config import CATEGORICAL_VARS, NUMERICAL_VARS

# 页面配置
st.set_page_config(
    page_title="JoyAge悦龄抑郁风险评估平台 (45+)",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 语言配置
LANGUAGES = {
    'zh': {
        'app_title': '🧠 JoyAge悦龄抑郁风险评估平台 (45+)',
        'language_select': '选择语言',
        'personal_info': '📝 个人信息输入',
        'basic_info': '👤 基本信息',
        'health_status': '🏥 健康状况',
        'lifestyle': '🚭 生活方式',
        'social_support': '👥 社会支持',
        'economic_status': '💰 经济状况',
        'family_info': '👨‍👩‍👧‍👦 家庭信息',
        'predict_button': '🔮 开始预测',
        'prediction_results': '📋 预测结果',
        'assessment_result': '📊 评估结果',
        'depression_probability': '抑郁风险概率',
        'risk_classification': '风险分级标准',
        'low_risk_desc': '低风险 (<30%): 常规监测',
        'medium_risk_desc': '中等风险 (30-70%): 加强关注',
        'high_risk_desc': '高风险 (>70%): 立即干预',
        'personal_analysis': '📊 个人化解释分析',
        'core_features': '🔍 核心重要特征',
        'high_risk': '⚠️ 高抑郁风险',
        'low_risk': '✅ 低抑郁风险',
        'risk_probability': '风险概率',
        'seek_help': '建议寻求专业心理健康支持',
        'maintain_health': '继续保持良好的心理健康状态',
        'input_info_prompt': '👈 请在左侧输入个人信息并点击预测按钮',
        'computing_explanation': '正在计算个人化解释...',
        'increase_risk': '增加风险',
        'decrease_risk': '降低风险',
        'current_value': '当前值',
        'impact': '影响',
        'shap_value': 'SHAP值',
        'usage_instructions': 'ℹ️ 使用说明',
        'left_panel_desc': '左侧：输入个人基本信息、健康状况、社会支持等特征',
        'right_panel_desc': '右侧：查看个人化的SHAP解释，了解哪些因素对您的抑郁风险影响最大',
        'disclaimer': '注意：此预测仅供参考，不能替代专业医疗诊断',
        'model_info': '模型信息：基于CHARLS数据训练，外部验证使用KLOSA数据',
        'shap_chart_title': '🎯 个人SHAP特征重要性分析',
        'shap_x_axis': 'SHAP值 (对预测的贡献)',
        'shap_y_axis': '特征'
    },
    'en': {
        'app_title': '🧠 JoyAge Depression Risk Assessment Platform (45+)',
        'language_select': 'Select Language',
        'personal_info': '📝 Personal Information Input',
        'basic_info': '👤 Basic Information',
        'health_status': '🏥 Health Status',
        'lifestyle': '🚭 Lifestyle',
        'social_support': '👥 Social Support',
        'economic_status': '💰 Economic Status',
        'family_info': '👨‍👩‍👧‍👦 Family Information',
        'predict_button': '🔮 Start Prediction',
        'prediction_results': '📋 Prediction Results',
        'assessment_result': '📊 Assessment Result',
        'depression_probability': 'Depression Probability',
        'risk_classification': 'Risk Classification Criteria',
        'low_risk_desc': 'Low Risk (<30%): Routine monitoring',
        'medium_risk_desc': 'Medium Risk (30-70%): Enhanced surveillance',
        'high_risk_desc': 'High Risk (>70%): Immediate intervention',
        'personal_analysis': '📊 Personalized Explanation Analysis',
        'core_features': '🔍 Core Important Features',
        'high_risk': '⚠️ High Depression Risk',
        'low_risk': '✅ Low Depression Risk',
        'risk_probability': 'Risk Probability',
        'seek_help': 'Recommend seeking professional mental health support',
        'maintain_health': 'Continue maintaining good mental health',
        'input_info_prompt': '👈 Please input personal information on the left and click predict',
        'computing_explanation': 'Computing personalized explanation...',
        'increase_risk': 'Increase Risk',
        'decrease_risk': 'Decrease Risk',
        'current_value': 'Current Value',
        'impact': 'Impact',
        'shap_value': 'SHAP Value',
        'usage_instructions': 'ℹ️ Usage Instructions',
        'left_panel_desc': 'Left Panel: Input personal basic information, health status, social support and other features',
        'right_panel_desc': 'Right Panel: View personalized SHAP explanations to understand which factors have the greatest impact on your depression risk',
        'disclaimer': 'Note: This prediction is for reference only and cannot replace professional medical diagnosis',
        'model_info': 'Model Info: Trained on CHARLS data, externally validated using KLOSA data',
        'shap_chart_title': '🎯 Personal SHAP Feature Importance Analysis',
        'shap_x_axis': 'SHAP Value (Contribution to Prediction)',
        'shap_y_axis': 'Features'
    },
    'ko': {
        'app_title': '🧠 JoyAge 우울 위험 평가 플랫폼 (45+)',
        'language_select': '언어 선택',
        'personal_info': '📝 개인정보 입력',
        'basic_info': '👤 기본 정보',
        'health_status': '🏥 건강 상태',
        'lifestyle': '🚭 생활 습관',
        'social_support': '👥 사회적 지원',
        'economic_status': '💰 경제적 상태',
        'family_info': '👨‍👩‍👧‍👦 가족 정보',
        'predict_button': '🔮 예측 시작',
        'prediction_results': '📋 예측 결과',
        'assessment_result': '📊 평가 결과',
        'depression_probability': '우울 위험 확률',
        'risk_classification': '위험 분류 기준',
        'low_risk_desc': '낮은 위험 (<30%): 정기 모니터링',
        'medium_risk_desc': '중간 위험 (30-70%): 강화 관찰',
        'high_risk_desc': '높은 위험 (>70%): 즉시 개입',
        'personal_analysis': '📊 개인화된 설명 분석',
        'core_features': '🔍 핵심 중요 특성',
        'high_risk': '⚠️ 높은 우울증 위험',
        'low_risk': '✅ 낮은 우울증 위험',
        'risk_probability': '위험 확률',
        'seek_help': '전문적인 정신건강 지원을 받을 것을 권합니다',
        'maintain_health': '좋은 정신건강 상태를 계속 유지하세요',
        'input_info_prompt': '👈 왼쪽에 개인정보를 입력하고 예측 버튼을 클릭하세요',
        'computing_explanation': '개인화된 설명을 계산 중...',
        'increase_risk': '위험 증가',
        'decrease_risk': '위험 감소',
        'current_value': '현재 값',
        'impact': '영향',
        'shap_value': 'SHAP 값',
        'usage_instructions': 'ℹ️ 사용 방법',
        'left_panel_desc': '왼쪽 패널: 개인 기본정보, 건강상태, 사회적 지원 등 특성을 입력',
        'right_panel_desc': '오른쪽 패널: 개인화된 SHAP 설명을 보고 우울증 위험에 가장 큰 영향을 미치는 요인을 파악',
        'disclaimer': '참고: 이 예측은 참고용일 뿐이며 전문적인 의료 진단을 대체할 수 없습니다',
        'model_info': '모델 정보: CHARLS 데이터로 훈련, KLOSA 데이터로 외부 검증',
        'shap_chart_title': '🎯 개인 SHAP 특성 중요도 분석',
        'shap_x_axis': 'SHAP 값 (예측에 대한 기여도)',
        'shap_y_axis': '특성'
    }
}

# 完整特征标签（中英文）
FEATURE_LABELS = {
    'zh': {
        # 基本信息
        'agey': '年龄',
        'ragender': '性别',
        'raeducl': '教育水平',
        'mstath': '婚姻状况',
        'rural': '居住地区',
        'child': '子女数量',
        'hhres': '家庭人数',
        
        # 健康状况
        'shlta': '自评健康状况',
        'hlthlm': '健康问题限制工作',
        'adlfive': '日常生活活动量表',
        'hibpe': '曾患高血压',
        'diabe': '曾患糖尿病',
        'cancre': '曾患癌症',
        'lunge': '曾患肺病',
        'hearte': '曾患心脏病',
        'stroke': '曾患中风',
        'arthre': '曾患关节炎',
        'livere': '曾患肝病',
        
        # 生活方式
        'drinkev': '曾经饮酒',
        'smokev': '曾经吸烟',
        'stayhospital': '去年住院经历',
        'fall': '近年跌倒经历',
        
        # 社会支持
        'hipriv': '私人医疗保险',
        'momliv': '母亲健在',
        'dadliv': '父亲健在',
        'ramomeducl': '母亲教育水平',
        'radadeducl': '父亲教育水平',
        'lvnear': '居住在子女附近',
        'kcntf': '每周与子女面对面接触',
        'socwk': '参与社会活动',
        
        # 经济状况
        'work': '目前工作中',
        'pubpen': '领取公共养老金',
        'peninc': '领取私人退休金',
        'ftrsp': '配偶能帮助未来ADL需求',
        'ftrkids': '子女/孙辈能帮助未来ADL需求',
        'painfr': '身体疼痛困扰',
        
        # 收入支出
        'comparable_hexp': '住院自付费用(2010年国际美元)',
        'comparable_exp': '家庭人均消费(2010年国际美元)',
        'comparable_itearn': '税后收入(2010年国际美元)',
        'comparable_frec': '转移支付收入总额(2010年国际美元)',
        'comparable_tgiv': '转移支付支出总额(2010年国际美元)',
        'comparable_ipubpen': '公共养老金收入(2010年国际美元)'
    },
    'en': {
        # Basic Information
        'agey': 'Age',
        'ragender': 'Gender',
        'raeducl': 'Education Level',
        'mstath': 'Marital Status',
        'rural': 'Residence Area',
        'child': 'Number of Children',
        'hhres': 'Household Size',
        
        # Health Status
        'shlta': 'Self-report of Health',
        'hlthlm': 'Health Problems Limit Work',
        'adlfive': 'Some Diff 5-item ADL Scale',
        'hibpe': 'Ever Had High Blood Pressure',
        'diabe': 'Ever Had Diabetes',
        'cancre': 'Ever Had Cancer',
        'lunge': 'Ever Had Lung Disease',
        'hearte': 'Ever Had Heart Problem',
        'stroke': 'Ever Had Stroke',
        'arthre': 'Ever Had Arthritis',
        'livere': 'Ever Had Liver Disease',
        
        # Lifestyle
        'drinkev': 'Ever Drinks Any Alcohol Before',
        'smokev': 'Smoke Ever',
        'stayhospital': 'Hospital Stay Last Year',
        'fall': 'Fallen Last Years',
        
        # Social Support
        'hipriv': 'Cover by Private Health Insurance',
        'momliv': 'Mother Alive',
        'dadliv': 'Father Alive',
        'ramomeducl': 'Mother Harmonized Education Levels',
        'radadeducl': 'Father Harmonized Education Levels',
        'lvnear': 'Live Near Children',
        'kcntf': 'Any Weekly Contact w/ Children in Person',
        'socwk': 'Participate in Social Activities',
        
        # Economic Status
        'work': 'Currently Working',
        'pubpen': 'Receives Public Pension',
        'peninc': 'Receives Private Pension',
        'ftrsp': 'Spouse Able to Help with Future ADL Needs',
        'ftrkids': 'Child/Grandchild Able to Help with Future ADL Needs',
        'painfr': 'Troubled with Body Pain',
        
        # Income & Expenditure
        'comparable_hexp': 'Hospitalization Out-of-pocket Expenditure (2010 Intl. Dollars)',
        'comparable_exp': 'Household Per Capita Consumption (2010 Intl. Dollars)',
        'comparable_itearn': 'Earnings After Tax (2010 Intl. Dollars)',
        'comparable_frec': 'Total Amount of Transfers Received (2010 Intl. Dollars)',
        'comparable_tgiv': 'Total Amount of Transfers Given (2010 Intl. Dollars)',
        'comparable_ipubpen': 'Public Pension Income (2010 Intl. Dollars)'
    },
    'ko': {
        # 기본 정보
        'agey': '나이',
        'ragender': '성별',
        'raeducl': '교육 수준',
        'mstath': '혼인 상태',
        'rural': '거주 지역',
        'child': '자녀 수',
        'hhres': '가구원 수',
        
        # 건강 상태
        'shlta': '자가평가 건강상태',
        'hlthlm': '건강 문제로 인한 업무 제한',
        'adlfive': '일상생활동작 척도',
        'hibpe': '고혈압 병력',
        'diabe': '당뇨병 병력',
        'cancre': '암 병력',
        'lunge': '폐질환 병력',
        'hearte': '심장병 병력',
        'stroke': '뇌졸중 병력',
        'arthre': '관절염 병력',
        'livere': '간질환 병력',
        
        # 생활 습관
        'drinkev': '음주 경험',
        'smokev': '흡연 경험',
        'stayhospital': '작년 입원 경험',
        'fall': '최근 낙상 경험',
        
        # 사회적 지원
        'hipriv': '민간 의료보험',
        'momliv': '어머니 생존',
        'dadliv': '아버지 생존',
        'ramomeducl': '어머니 교육 수준',
        'radadeducl': '아버지 교육 수준',
        'lvnear': '자녀 근처 거주',
        'kcntf': '자녀와의 주간 대면 접촉',
        'socwk': '사회활동 참여',
        
        # 경제적 상태
        'work': '현재 근무 중',
        'pubpen': '공적연금 수급',
        'peninc': '사적연금 수급',
        'ftrsp': '배우자의 미래 ADL 도움 가능성',
        'ftrkids': '자녀/손자녀의 미래 ADL 도움 가능성',
        'painfr': '신체 통증 고민',
        
        # 수입과 지출
        'comparable_hexp': '입원 본인부담금 (2010년 국제달러)',
        'comparable_exp': '가구 1인당 소비 (2010년 국제달러)',
        'comparable_itearn': '세후 근로소득 (2010년 국제달러)',
        'comparable_frec': '받은 이전소득 총액 (2010년 국제달러)',
        'comparable_tgiv': '제공한 이전소득 총액 (2010년 국제달러)',
        'comparable_ipubpen': '공적연금 소득 (2010년 국제달러)'
    }
}

# 选项标签字典 - 根据语言分离
OPTION_LABELS = {
    'zh': {
        'gender': {0: "男性", 1: "女性"},
        'education': {
            0: "无正规教育", 1: "未完成小学", 2: "小学毕业", 3: "初中毕业", 
            4: "高中毕业", 5: "职业技术学校", 6: "大专", 7: "本科", 8: "硕士", 9: "博士"
        },
        'marital': {
            0: "已婚或同居", 1: "已婚配偶不在", 
            2: "分居", 3: "离异", 4: "丧偶", 5: "从未结婚"
        },
        'rural': {"0.Urban Community": "城市", "1.Rural Village": "农村"},
        'health': {0: "很好", 1: "好", 2: "一般", 3: "差", 4: "很差"},
        'adl': {
            0: "完全独立", 1: "轻度依赖", 2: "中度依赖", 
            3: "重度依赖", 4: "严重依赖", 5: "完全依赖"
        },
        'parent_education': {
            0: "低于高中", 1: "高中及职业培训", 2: "高等教育"
        },
        'work': {"0.Not working for pay": "不工作", "1.Working for pay": "工作"},
        'yes_no': "是",
        'yes_no_false': "否"
    },
    'en': {
        'gender': {0: "Male", 1: "Female"},
        'education': {
            0: "No formal education", 1: "Did not finish primary school", 
            2: "Primary school", 3: "Middle school", 4: "High school", 
            5: "Vocational school", 6: "Junior college", 7: "University", 8: "Master", 9: "PhD"
        },
        'marital': {
            0: "Married", 1: "Married spouse absent", 
            2: "Separated", 3: "Divorced", 4: "Widowed", 5: "Never married"
        },
        'rural': {"0.Urban Community": "Urban", "1.Rural Village": "Rural"},
        'health': {0: "Very good", 1: "Good", 2: "Fair", 3: "Poor", 4: "Very Poor"},
        'adl': {
            0: "Fully Independent", 1: "Mild Dependence", 2: "Moderate Dependence", 
            3: "Significant Dependence", 4: "Severe Dependence", 5: "Total Dependence"
        },
        'parent_education': {
            0: "Less than upper secondary", 1: "Upper secondary & vocational", 2: "Tertiary"
        },
        'work': {"0.Not working for pay": "Not Working", "1.Working for pay": "Working"},
        'yes_no': "Yes",
        'yes_no_false': "No"
    },
    'ko': {
        'gender': {0: "남성", 1: "여성"},
        'education': {
            0: "정규교육 없음", 1: "초등학교 미졸업", 2: "초등학교 졸업", 3: "중학교 졸업", 
            4: "고등학교 졸업", 5: "직업기술학교", 6: "전문대학", 7: "대학교", 8: "석사", 9: "박사"
        },
        'marital': {
            0: "기혼 또는 동거", 1: "기혼 배우자 부재", 
            2: "별거", 3: "이혼", 4: "사별", 5: "미혼"
        },
        'rural': {"0.Urban Community": "도시", "1.Rural Village": "농촌"},
        'health': {0: "매우 좋음", 1: "좋음", 2: "보통", 3: "나쁨", 4: "매우 나쁨"},
        'adl': {
            0: "완전 독립", 1: "경미한 의존", 2: "중등도 의존", 
            3: "상당한 의존", 4: "심각한 의존", 5: "완전 의존"
        },
        'parent_education': {
            0: "고등학교 미만", 1: "고등학교 및 직업훈련", 2: "고등교육"
        },
        'work': {"0.Not working for pay": "근무하지 않음", "1.Working for pay": "근무 중"},
        'yes_no': "예",
        'yes_no_false': "아니요"
    }
}

# 优化的自定义CSS样式
st.markdown("""
<style>
/* 全局样式优化 */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* 主标题优化 */
.main-header {
    font-size: 3.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 2.5rem;
    padding: 1rem 0;
    text-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

/* 卡片容器样式 */
.card-container {
    background: white;
    border-radius: 15px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07), 0 1px 3px rgba(0, 0, 0, 0.1);
    border: 1px solid rgba(0, 0, 0, 0.05);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.card-container:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1), 0 3px 6px rgba(0, 0, 0, 0.08);
}

/* 预测结果框优化 */
.prediction-box {
    padding: 2rem 1.5rem;
    border-radius: 15px;
    text-align: center;
    margin: 1.5rem 0;
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
    position: relative;
    overflow: hidden;
}

.prediction-box::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.6), transparent);
}

.high-risk {
    background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
    border: 1px solid #ef5350;
    color: #d32f2f;
}

.medium-risk {
    background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
    border: 1px solid #ff9800;
    color: #f57c00;
}

.low-risk {
    background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
    border: 1px solid #4caf50;
    color: #2e7d32;
}

.prediction-box h3 {
    font-size: 1.8rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}

.prediction-box p {
    font-size: 1.25rem;
    margin: 0.3rem 0;
}

/* 特征重要性卡片 */
.feature-card {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
    border-left: 4px solid #2196f3;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    transition: all 0.2s ease;
}

.feature-card:hover {
    transform: translateX(5px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}

.feature-card .feature-name {
    font-weight: 600;
    color: #2c3e50;
    font-size: 1rem;
}

.feature-card .feature-value {
    color: #7f8c8d;
    font-size: 0.9rem;
}

.feature-card .shap-value {
    font-weight: 500;
    font-size: 0.95rem;
}

.feature-card .impact-positive {
    color: #27ae60;
}

.feature-card .impact-negative {
    color: #e74c3c;
}

/* 侧边栏样式优化 */
.css-1d391kg {
    background-color: #f8f9fa;
}

.css-1d391kg .stSelectbox > div > div {
    background-color: white;
    border-radius: 8px;
}

/* 按钮样式优化 */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    padding: 0.75rem 1.5rem;
    border-radius: 10px;
    font-weight: 600;
    font-size: 1rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 6px rgba(102, 126, 234, 0.25);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(102, 126, 234, 0.35);
}

/* 加载动画优化 */
.stSpinner > div {
    border-color: #667eea transparent #667eea transparent;
}

/* 分组标题样式 */
.section-header {
    font-size: 1.5rem;
    font-weight: 600;
    color: #2c3e50;
    margin: 1.5rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #e9ecef;
    display: flex;
    align-items: center;
}

.section-header::before {
    content: '';
    width: 4px;
    height: 1.5rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    margin-right: 0.75rem;
    border-radius: 2px;
}

/* 信息提示框 */
.info-box {
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    border: 1px solid #2196f3;
    border-radius: 10px;
    padding: 1rem;
    margin: 1rem 0;
    font-size: 0.95rem;
    color: #1565c0;
}

/* 响应式设计 */
@media (max-width: 768px) {
    .main-header {
        font-size: 2rem;
    }
    
    .prediction-box {
        padding: 1.5rem 1rem;
    }
    
    .card-container {
        margin: 0.5rem 0;
        padding: 1rem;
    }
}

/* 滚动条美化 */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #5a6fd8 0%, #6a4c93 100%);
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """优化的模型加载器 - 支持多重备选策略"""
    # 智能模型目录定位 - 同时支持本地和Streamlit Cloud
    script_dir = Path(__file__).parent
    candidate_dirs = [
        script_dir / "saved_models",  # 脚本同级目录（推荐）
        Path.cwd() / "saved_models",  # 工作目录（兼容）
        Path("saved_models")  # 相对路径（备选）
    ]
    
    models_dir = None
    for candidate in candidate_dirs:
        if candidate.exists() and candidate.is_dir():
            models_dir = candidate
            break
    
    if models_dir is None:
        return None, None, "❌ Models directory not found / 模型目录不存在 / 모델 디렉토리를 찾을 수 없습니다"
        
    # 模型优先级策略
    model_priorities = [
        ("cesd_model_best_*hyperparameter_tuned*.joblib", "🎯 调优模型 / Tuned Model / 조정된 모델"),
        ("cesd_model_best_*.joblib", "📊 最佳模型 / Best Model / 최고 모델"), 
        ("cesd_model_*.joblib", "⚙️ 备用模型 / Backup Model / 백업 모델")
    ]
    
    model, model_info = None, None
    
    for pattern, description in model_priorities:
        model_files = list(models_dir.glob(pattern))
        if model_files:
            # 选择最新的文件
            latest_file = max(model_files, key=lambda p: p.stat().st_mtime)
            try:
                model = joblib.load(latest_file)
                model_info = f"{description}: {latest_file.name}"
                break
            except Exception as e:
                continue
    
    if model is None:
        return None, None, "❌ No valid model found / 未找到有效模型 / 유효한 모델을 찾을 수 없습니다"
    
    # 加载数据处理器
    processor_path = models_dir / "data_processor.joblib"
    data_processor = None
    
    if processor_path.exists():
        try:
            data_processor = joblib.load(processor_path)
        except Exception as e:
            pass
    
    return model, data_processor, model_info

def get_language():
    """获取当前语言设置"""
    if 'language' not in st.session_state:
        st.session_state.language = 'zh'
    return st.session_state.language

def get_text(key):
    """获取当前语言的文本"""
    lang = get_language()
    return LANGUAGES[lang].get(key, key)

def get_feature_label(feature_name):
    """获取特征的当前语言标签"""
    lang = get_language()
    return FEATURE_LABELS[lang].get(feature_name, feature_name)

@st.cache_data
def get_feature_groups():
    """获取特征分组配置"""
    return {
        'basic': ['agey', 'ragender', 'raeducl', 'mstath', 'rural', 'child', 'hhres'],
        'health': ['shlta', 'hlthlm', 'adlfive', 'hibpe', 'diabe', 'cancre', 'lunge', 'hearte', 'stroke', 'arthre', 'livere', 'painfr'],
        'lifestyle': ['drinkev', 'smokev', 'stayhospital', 'fall'],
        'social': ['hipriv', 'momliv', 'dadliv', 'lvnear', 'kcntf', 'socwk', 'ramomeducl', 'radadeducl', 'ftrsp', 'ftrkids'],
        'economic': ['work', 'pubpen', 'peninc'],
        'financial': ['comparable_hexp', 'comparable_exp', 'comparable_itearn', 'comparable_frec', 'comparable_tgiv', 'comparable_ipubpen']
    }

def _create_basic_info_inputs(inputs, lang):
    """创建基本信息输入"""
    # 年龄
    inputs['agey'] = st.sidebar.slider(get_feature_label('agey'), 45, 100, 65)
    
    # 性别
    gender_options = [0, 1]
    inputs['ragender'] = st.sidebar.selectbox(
        get_feature_label('ragender'), gender_options,
        format_func=lambda x: OPTION_LABELS[lang]['gender'][x]
    )
    
    # 教育水平
    edu_options = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    inputs['raeducl'] = st.sidebar.selectbox(
        get_feature_label('raeducl'), edu_options,
        format_func=lambda x: OPTION_LABELS[lang]['education'][x]
    )
    
    # 婚姻状况
    marriage_options = [0, 1, 2, 3, 4, 5]
    inputs['mstath'] = st.sidebar.selectbox(
        get_feature_label('mstath'), marriage_options,
        format_func=lambda x: OPTION_LABELS[lang]['marital'][x]
    )
    
    # 居住地区 - 使用数值编码
    rural_options = [0, 1]  # 0=Urban, 1=Rural
    inputs['rural'] = st.sidebar.selectbox(
        get_feature_label('rural'), rural_options,
        format_func=lambda x: OPTION_LABELS[lang]['rural'][f"{x}.Urban Community" if x == 0 else f"{x}.Rural Village"]
    )
    
    # 子女数量和家庭人数
    inputs['child'] = st.sidebar.slider(get_feature_label('child'), 0, 10, 2)
    inputs['hhres'] = st.sidebar.slider(get_feature_label('hhres'), 1, 10, 3)

def _create_health_inputs(inputs, lang):
    """创建健康状况输入"""
    # 健康状况
    health_options = [0, 1, 2, 3, 4]
    inputs['shlta'] = st.sidebar.selectbox(
        get_feature_label('shlta'), health_options,
        format_func=lambda x: OPTION_LABELS[lang]['health'][x]
    )
    
    # 健康限制和ADL
    inputs['hlthlm'] = 1 if st.sidebar.checkbox(get_feature_label('hlthlm')) else 0
    
    adl_options = [0, 1, 2, 3, 4, 5]
    inputs['adlfive'] = st.sidebar.selectbox(
        get_feature_label('adlfive'), adl_options,
        format_func=lambda x: OPTION_LABELS[lang]['adl'][x]
    )
    
    # 疾病史
    diseases = ['hibpe', 'diabe', 'cancre', 'lunge', 'hearte', 'stroke', 'arthre', 'livere']
    for disease in diseases:
        inputs[disease] = 1 if st.sidebar.checkbox(get_feature_label(disease)) else 0
    
    # 身体疼痛
    inputs['painfr'] = 1 if st.sidebar.checkbox(get_feature_label('painfr')) else 0

def _create_other_inputs(inputs, lang):
    """创建其他类别输入（生活方式、社会支持、经济状况）"""
    # 生活方式
    lifestyle_vars = ['drinkev', 'smokev', 'stayhospital', 'fall']
    for var in lifestyle_vars:
        inputs[var] = 1 if st.sidebar.checkbox(get_feature_label(var)) else 0
    
    # 社会支持
    social_vars = ['hipriv', 'momliv', 'dadliv', 'lvnear', 'kcntf', 'socwk']
    for var in social_vars:
        inputs[var] = 1 if st.sidebar.checkbox(get_feature_label(var)) else 0
    
    # 未来照护支持
    future_care_vars = ['ftrsp', 'ftrkids']
    for var in future_care_vars:
        inputs[var] = 1 if st.sidebar.checkbox(get_feature_label(var)) else 0
    
    # 父母教育
    parent_education_options = [0, 1, 2]
    for var in ['ramomeducl', 'radadeducl']:
        inputs[var] = st.sidebar.selectbox(
            get_feature_label(var), parent_education_options,
            format_func=lambda x: OPTION_LABELS[lang]['parent_education'][x]
        )
    
    # 工作状态 - 使用数值编码
    work_options = [0, 1]  # 0=Not working, 1=Working
    inputs['work'] = st.sidebar.selectbox(
        get_feature_label('work'), work_options,
        format_func=lambda x: OPTION_LABELS[lang]['work'][f"{x}.Not working for pay" if x == 0 else f"{x}.Working for pay"]
    )
    
    # 收入相关 - 修复编码格式
    pension_vars = ['pubpen', 'peninc']
    for var in pension_vars:
        inputs[var] = 1 if st.sidebar.checkbox(get_feature_label(var)) else 0
    
    # 收入支出变量
    income_title = {'zh': "💰 收入支出", 'en': "💰 Income & Expenditure", 'ko': "💰 수입과 지출"}
    st.sidebar.subheader(income_title[lang])
    
    financial_vars = {
        'comparable_hexp': (0, 50000, 0), 'comparable_exp': (0, 100000, 1000),
        'comparable_itearn': (0, 100000, 0), 'comparable_frec': (0, 200000, 2000),
        'comparable_tgiv': (0, 50000, 0), 'comparable_ipubpen': (0, 50000, 0)
    }
    
    for var, (min_val, max_val, default_val) in financial_vars.items():
        inputs[var] = st.sidebar.slider(get_feature_label(var), min_val, max_val, default_val)

def create_complete_input_form():
    """优化的表单创建 - 模块化和进度显示"""
    lang = get_language()
    
    # 语言切换器已移动到页面右上角（见 main() 顶部），这里移除以避免重复
    
    # 表单头部
    st.sidebar.header(get_text('personal_info'))
    
    inputs = {}
    
    # 基本信息
    with st.sidebar.expander(f"📋 {get_text('basic_info')}", expanded=True):
        _create_basic_info_inputs(inputs, lang)
    
    # 健康状况
    with st.sidebar.expander(f"🏥 {get_text('health_status')}", expanded=False):
        _create_health_inputs(inputs, lang)
    
    # 其他信息（生活方式、社会支持、经济状况）
    section_names = {
        'lifestyle': f"🚭 {get_text('lifestyle')}",
        'social': f"👥 {get_text('social_support')}",
        'economic': f"💰 {get_text('economic_status')}"
    }
    
    with st.sidebar.expander("🔗 其他信息", expanded=False):
        _create_other_inputs(inputs, lang)
    
    return inputs

@st.cache_data
def validate_and_clean_inputs(inputs):
    """验证和清理输入数据"""
    cleaned_inputs = {}
    
    # 数值型变量验证
    for var in NUMERICAL_VARS:
        if var in inputs:
            try:
                cleaned_inputs[var] = float(inputs[var])
            except (ValueError, TypeError):
                cleaned_inputs[var] = 0.0  # 默认值
        else:
            cleaned_inputs[var] = 0.0
    
    # 分类变量验证
    for var in CATEGORICAL_VARS:
        if var in inputs:
            cleaned_inputs[var] = inputs[var]
        else:
            # 提供合理的默认值
            if var in ['ragender']:
                cleaned_inputs[var] = 0  # 默认男性
            elif var in ['rural']:
                cleaned_inputs[var] = "0.Urban Community"  # 默认城市
            elif var.endswith('e'):  # 疾病变量
                cleaned_inputs[var] = "0.No"  # 默认无疾病
            else:
                cleaned_inputs[var] = 0
    
    return cleaned_inputs

def preprocess_input(inputs, data_processor):
    """优化的数据预处理 - 支持缓存和错误恢复"""
    try:
        # Step 1: 输入验证和清理
        cleaned_inputs = validate_and_clean_inputs(inputs)
        
        # Step 2: 转换为DataFrame
        df = pd.DataFrame([cleaned_inputs])
        
        # Step 3: 确保特征顺序一致
        all_features = CATEGORICAL_VARS + NUMERICAL_VARS
        
        # 验证必需特征
        missing_features = set(all_features) - set(df.columns)
        if missing_features:
            st.warning(f"⚠️ Missing features detected: {list(missing_features)[:5]}...")
            # 添加缺失特征的默认值
            for feature in missing_features:
                df[feature] = 0
        
        # 重新排序列
        df = df.reindex(columns=all_features, fill_value=0)
        
        # Step 4: 数据处理器预处理
        if data_processor:
            try:
                # 填充缺失值
                df = data_processor.impute_features(df, is_training=False)
                
                # 应用分类变量标准化
                if hasattr(data_processor, '_standardize_categorical_formats'):
                    df = data_processor._standardize_categorical_formats(df)
            except Exception as e:
                st.warning(f"⚠️ Data processor failed, using fallback: {e}")
                df = _fallback_preprocessing(df)
        else:
            # 备选预处理方法
            df = _fallback_preprocessing(df)
        
        # Step 5: 最终验证
        if df.isnull().any().any():
            st.warning("⚠️ Remaining null values detected, filling with defaults")
            df = df.fillna(0)
        
        return df
        
    except Exception as e:
        st.error(f"❌ Data preprocessing failed / 数据预处理失败: {str(e)}")
        return None

def _fallback_preprocessing(df):
    """备选预处理方法"""
    # 简单的分类变量编码
    for col in CATEGORICAL_VARS:
        if col in df.columns and df[col].dtype == 'object':
            # 字符串到数值的映射
            unique_vals = df[col].unique()
            if len(unique_vals) > 0:
                mapping = {val: i for i, val in enumerate(unique_vals)}
                df[col] = df[col].map(mapping).fillna(0)
    
    # 确保数值变量为数值型
    for col in NUMERICAL_VARS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return df

@st.cache_data
def get_model_type(_model):
    """检测模型类型用于SHAP选择"""
    model_name = _model.__class__.__name__.lower()
    
    if any(tree_name in model_name for tree_name in ['forest', 'tree', 'gradient', 'xgb', 'lgb', 'catboost']):
        return 'tree'
    elif any(linear_name in model_name for linear_name in ['linear', 'logistic', 'svm', 'ridge']):
        return 'linear'
    else:
        return 'auto'

def compute_shap_explanation(model, X, feature_names):
    """优化的SHAP计算 - 支持智能解释器选择和错误恢复"""
    try:
        # 数据验证
        if X is None or X.empty:
            st.error("❌ Invalid input data for SHAP computation")
            return None, None
        
        # 智能选择SHAP解释器
        model_type = get_model_type(model)
        
        with st.spinner("🔍 Computing SHAP explanations..."):
            if model_type == 'tree' and hasattr(model, 'predict_proba'):
                try:
                    # 优先使用TreeExplainer（更快更准确）
                    explainer = shap.TreeExplainer(model)
                    
                except Exception as e:
                    st.warning(f"⚠️ TreeExplainer failed: {e}, falling back to Explainer")
                    explainer = shap.Explainer(model)
            elif model_type == 'linear':
                try:
                    # 线性模型使用LinearExplainer
                    explainer = shap.LinearExplainer(model, X)
                    st.info("📊 Using LinearExplainer for linear model")
                except Exception as e:
                    st.warning(f"⚠️ LinearExplainer failed: {e}, using general Explainer")
                    explainer = shap.Explainer(model)
            else:
                # 通用解释器
                explainer = shap.Explainer(model)
                st.info("🔧 Using general Explainer")
            
            # 计算SHAP值
            shap_values = explainer.shap_values(X)
            
            # 处理不同格式的SHAP值
            if isinstance(shap_values, list):
                if len(shap_values) == 2:
                    # 二分类情况，取正类
                    shap_values = shap_values[1]
                    st.info("✅ Binary classification SHAP values extracted")
                else:
                    # 多分类情况，可能需要特殊处理
                    st.warning(f"⚠️ Multi-class SHAP values detected ({len(shap_values)} classes)")
                    shap_values = shap_values[0]  # 暂时使用第一类
            
            # 获取基准值
            if hasattr(explainer, 'expected_value'):
                expected_value = explainer.expected_value
                if isinstance(expected_value, list):
                    expected_value = expected_value[1] if len(expected_value) == 2 else expected_value[0]
            else:
                expected_value = 0.0
                st.warning("⚠️ Expected value not available, using 0.0")
            
            # 验证结果
            if shap_values is None:
                st.error("❌ SHAP computation returned None")
                return None, None
            
            return shap_values, expected_value
        
    except Exception as e:
        st.error(f"❌ SHAP computation failed / SHAP计算失败: {str(e)}")
        st.error("💡 Tip: This might be due to model compatibility. Try using a different model.")
        return None, None

def create_shap_waterfall_plot(shap_values, expected_value, feature_names, feature_values):
    """创建SHAP瀑布图"""
    try:
        lang = get_language()
        
        # 获取单个样本的SHAP值
        if len(shap_values.shape) > 1:
            shap_vals = shap_values[0]
        else:
            shap_vals = shap_values
            
        # 计算绝对值排序
        abs_importance = np.abs(shap_vals)
        top_indices = np.argsort(abs_importance)[-10:][::-1]  # 取前10个重要特征
        
        # 准备数据
        top_features = [feature_names[i] for i in top_indices]
        top_values = [shap_vals[i] for i in top_indices]
        top_labels = [get_feature_label(f) for f in top_features]
        
        # 创建瀑布图
        fig = go.Figure()
        
        # 基线配色：粉/蓝（higher/lower）
        palette_up = '#FF0D57'   # higher → red/pink
        palette_down = '#1F88F5' # lower  → blue
        colors = [palette_up if v > 0 else palette_down for v in top_values]
        
        # 添加条形图
        fig.add_trace(go.Bar(
            y=top_labels,
            x=top_values,
            orientation='h',
            marker_color=colors,
            text=[f"{v:.3f}" for v in top_values],
            textposition='outside'
        ))
        
        fig.update_layout(
            title=get_text('shap_chart_title'),
            title_font=dict(size=22, family='Arial Black'),
            font=dict(size=18),
            xaxis_title=get_text('shap_x_axis'),
            yaxis_title=get_text('shap_y_axis'),
            height=500,
            showlegend=False
        )
        # 放大坐标轴与标签字体
        fig.update_yaxes(tickfont=dict(size=16, family='Arial Black'))
        fig.update_xaxes(tickfont=dict(size=14))
        
        return fig
        
    except Exception as e:
        st.error(f"SHAP visualization failed / SHAP可视化失败: {str(e)}")
        return None

def risk_gauge(probability, age, gender):
    """Plotly风险仪表盘，红黄绿配色，含年龄基准阈值"""
    import plotly.graph_objects as go
    age_baseline = 0.15 * (age / 60)
    color = "#2ecc71" if probability <= 0.4 else ("#f39c12" if probability <= 0.7 else "#e74c3c")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={'suffix': '%', 'font': {'size': 42, 'color': color}},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': color, 'thickness': 0.7},
            'steps': [
                {'range': [0, 40], 'color': 'rgba(46,204,113,0.25)'},
                {'range': [40, 70], 'color': 'rgba(243,156,18,0.25)'},
                {'range': [70, 100], 'color': 'rgba(231,76,60,0.25)'}
            ],
            'threshold': {'line': {'color': 'black', 'width': 4}, 'thickness': 0.75, 'value': age_baseline * 100}
        }
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=10), paper_bgcolor='rgba(0,0,0,0)')
    return fig

RISK_THRESHOLDS = {'low': 0.3, 'high': 0.7}

def get_risk_style(prob, lang='en'):
    """根据概率获取风险样式 - 支持多语言"""
    # 风险级别翻译
    risk_levels = {
        'zh': {'low': '低风险', 'medium': '中等风险', 'high': '高风险'},
        'en': {'low': 'Low Risk', 'medium': 'Medium Risk', 'high': 'High Risk'},
        'ko': {'low': '낮은 위험', 'medium': '중간 위험', 'high': '높은 위험'}
    }
    
    if prob <= RISK_THRESHOLDS['low']:
        return (risk_levels[lang]['low'], "#2ecc71", "✅")
    elif prob <= RISK_THRESHOLDS['high']:
        return (risk_levels[lang]['medium'], "#f1c40f", "⚠️")
    else:
        return (risk_levels[lang]['high'], "#e74c3c", "🚨")

def render_result_card(prob):
    lang = get_language()
    risk_level, color, icon = get_risk_style(prob, lang)
    st.subheader(get_text("assessment_result"))
    st.markdown(f"""
    <div style="border:2px solid {color}; border-radius:10px; padding:20px;">
        <h3 style="color:{color}; text-align:center; font-size:28px;">{icon} {risk_level}</h3>
        <p style="text-align:center; font-size:28px; font-weight:700;">{get_text("depression_probability")}: {prob*100:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)
    with st.expander(get_text("risk_classification")):
        st.markdown(f"""
        - {get_text("low_risk_desc")}
        - {get_text("medium_risk_desc")}  
        - {get_text("high_risk_desc")}
        """)

def render_key_factors_table(shap_values, feature_names):
    st.subheader("Key Contributing Factors")
    shap_row = shap_values[0] if len(shap_values.shape)>1 else shap_values
    df = pd.DataFrame({
        'Clinical Factor': feature_names,
        'Impact Direction': ['Positive' if v>0 else 'Negative' for v in shap_row],
        'Impact Magnitude': np.abs(shap_row)
    }).sort_values('Impact Magnitude', ascending=False).head(5)
    st.dataframe(
        df[['Clinical Factor','Impact Direction']],
        column_config={
            'Impact Direction': st.column_config.TextColumn(
                help='Positive: Higher values increase risk; Negative: Higher values decrease risk'
            )
        },
        hide_index=True
    )
    return df

# Hero 文案（顶置于 render_hero 之前）
JOYAGE_INTRO = {
    'zh': """ 
JoyAge悦龄平台致力于实现联合国“健康老龄化十年行动计划”的愿景，通过科学、精准的抑郁风险预测，为全球45岁以上中老年人群提供个性化心理健康评估服务。我们相信，每一位长者都应该拥有快乐、有尊严的晚年生活。""",
    'en': """
JoyAge platform is dedicated to advancing the UN's "Decade of Healthy Ageing" vision by providing scientific and precise depression risk prediction for individuals aged 45+ globally. We believe every senior deserves a joyful and dignified later life.""",
    'ko': """
JoyAge 플랫폼은 UN 'Healthy Ageing 10년 행동계획'의 비전을 실현하기 위해 45세 이상을 대상으로 과학적·정밀한 우울 위험 예측을 제공합니다."""
}

def render_hero(lang: str):
    title_cn = "目标愿景"
    title_en = "Goal & Mission"
    title_ko = "목표 비전"
    st.markdown(
        """
        <style>
          .hero-wrap{background:linear-gradient(135deg,#f5f7ff 0%,#eef2ff 100%);border-radius:16px;padding:18px 22px;margin-bottom:16px;border:1px solid rgba(102,126,234,.15)}
          .hero-title{font-size:26px;font-weight:800;color:#2c3e50;line-height:1.3}
          .hero-sub{margin-top:8px;font-size:15px;color:#475569}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="hero-wrap">
          <div class="hero-title">🧠 {title_cn if lang == 'zh' else (title_ko if lang == 'ko' else title_en)}</div>
          <div class="hero-sub">{JOYAGE_INTRO.get(lang, JOYAGE_INTRO['en'])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def main():
    """主函数"""
    lang = get_language()
    
    # 顶部右上角语言切换器
    top_c1, top_c2 = st.columns([6,1])
    with top_c1:
        st.markdown(f'<h1 class="main-header">{get_text("app_title")}</h1>', unsafe_allow_html=True)
    with top_c2:
        language_options = ['zh', 'en', 'ko']
        language_names = {'zh': '🇨🇳 中文', 'en': '🇺🇸 English', 'ko': '🇰🇷 한국어'}
        current_index = language_options.index(lang) if lang in language_options else 0
        new_lang = st.selectbox(get_text('language_select'), language_options, index=current_index,
                                format_func=lambda x: language_names[x], key="language_selector_top")
        if new_lang != lang:
            st.session_state.language = new_lang
            st.rerun()
    
    # 顶部 Hero（标题+平台愿景）
    render_hero(get_language())
 
    # 加载模型
    model, data_processor, model_info = load_models()
    
    if model is None:
        # 动态错误消息
        error_messages = {
            'zh': "❌ 无法加载模型，请检查模型文件是否存在",
            'en': "❌ Unable to load model, please check if model files exist", 
            'ko': "❌ 모델을 로드할 수 없습니다. 모델 파일이 존재하는지 확인하세요"
        }
        st.error(error_messages[lang])
        return
    
    # 暂存模型信息，稍后在页面底部显示
    st.session_state.model_info = model_info
    st.session_state.data_processor = data_processor
    
    # 创建两列布局
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # 使用卡片容器包装预测结果
        st.markdown(f'''
        <div class="card-container">
            <div class="section-header">{get_text("prediction_results")}</div>
        ''', unsafe_allow_html=True)
        
        # 获取用户输入（完整42个特征）
        inputs = create_complete_input_form()
        
        # 添加使用说明
        st.markdown(f"""
        <div class="info-box">
            <strong>ℹ️ {get_text('usage_instructions')}</strong><br>
            {get_text('left_panel_desc')}<br>
            <em>{get_text('disclaimer')}</em>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button(get_text('predict_button'), type="primary"):
            # 预处理数据
            X = preprocess_input(inputs, data_processor)
            
            if X is not None:
                try:
                    # 进行预测
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X)[0]
                        risk_prob = proba[1]  # 抑郁风险概率
                    else:
                        risk_prob = 0.5  # 默认值
                    
                    pred = model.predict(X)[0]
                    
                    # 结果卡片整行展示，宽度与"开始预测"一致
                    render_result_card(risk_prob)
                    
                    # 计算SHAP（不展示表格）
                    shap_values, expected_value = compute_shap_explanation(model, X, X.columns)
                    if shap_values is not None:
                        pass
                    
                    # 存储结果
                    st.session_state['prediction_done'] = True
                    st.session_state['X'] = X
                    st.session_state['risk_prob'] = risk_prob
                    # 🔧 修复SHAP缓存错误：每次预测都更新SHAP值，确保动态变化
                    st.session_state['shap_values'] = shap_values
                    st.session_state['expected_value'] = expected_value
                    
                except Exception as e:
                    st.error(f"Prediction failed / 预测失败: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)  # 结束卡片容器
    
    with col2:
        # 使用卡片容器包装SHAP分析
        st.markdown(f'''
        <div class="card-container">
            <div class="section-header">{get_text("personal_analysis")}</div>
        ''', unsafe_allow_html=True)
        
        if st.session_state.get('prediction_done', False):
            X = st.session_state['X']
            shap_values = st.session_state.get('shap_values')
            expected_value = st.session_state.get('expected_value')
            if shap_values is not None:
                fig = create_shap_waterfall_plot(shap_values, expected_value, X.columns, X.iloc[0])
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown(f"""
            <div class="info-box">
                <strong>👈 {get_text('input_info_prompt')}</strong><br>
                {get_text('right_panel_desc')}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)  # 结束卡片容器
    
    # 底部信息 - 使用卡片样式
    st.markdown("---")
    
    # 多语言模型信息
    model_info_text = {
        'zh': {
            'architecture': '📊 模型架构：基于CHARLS 2018年数据训练，外部验证使用KLOSA数据',
            'target': '🎯 预测目标：中老年人群抑郁症风险评估',
            'performance': '📈 模型性能：经过严格的交叉验证和超参数优化',
            'explanation': '🔬 解释方法：使用SHAP提供个性化特征重要性分析',
            'disclaimer': '⚠️ 此预测结果仅供参考，不能替代专业医疗诊断。如有心理健康担忧，请咨询专业医疗人员。'
        },
        'en': {
            'architecture': '📊 Model Architecture: Trained on CHARLS 2018 data, externally validated using KLOSA data',
            'target': '🎯 Prediction Target: Depression risk assessment for middle-aged and elderly populations',
            'performance': '📈 Model Performance: Rigorous cross-validation and hyperparameter optimization',
            'explanation': '🔬 Explanation Method: SHAP for personalized feature importance analysis',
            'disclaimer': '⚠️ This prediction is for reference only and cannot replace professional medical diagnosis. If you have mental health concerns, please consult a healthcare professional.'
        },
        'ko': {
            'architecture': '📊 모델 구조: CHARLS 2018 데이터로 훈련, KLOSA 데이터로 외부 검증',
            'target': '🎯 예측 목표: 중고령층 우울증 위험 평가',
            'performance': '📈 모델 성능: 엄격한 교차검증 및 하이퍼파라미터 최적화',
            'explanation': '🔬 설명 방법: 개인화된 특성 중요도 분석을 위한 SHAP 사용',
            'disclaimer': '⚠️ 이 예측은 참고용일 뿐이며 전문적인 의료 진단을 대체할 수 없습니다. 정신건강에 대한 우려가 있으시면 의료 전문가와 상담하시기 바랍니다.'
        }
    }
    
    lang = get_language()
    info = model_info_text[lang]
    
    st.markdown(f"""
    <div class="card-container">
        <div class="section-header">{get_text('model_info')}</div>
        <div class="info-box">
            <strong>{info['architecture']}</strong><br>
            <strong>{info['target']}</strong><br>
            <strong>{info['performance']}</strong><br>
            <strong>{info['explanation']}</strong>
        </div>
        <div style="margin-top: 1rem; font-size: 0.9rem; color: #7f8c8d; text-align: center;">
            {info['disclaimer']}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ================== 模型状态信息（页面底部） ==================
    st.divider()
    
    # 模型状态信息标题
    status_titles = {
        'zh': "🔧 系统状态信息",
        'en': "🔧 System Status Information", 
        'ko': "🔧 시스템 상태 정보"
    }
    
    st.subheader(status_titles[lang])
    
    # 检查是否有存储的模型信息
    if hasattr(st.session_state, 'model_info') and st.session_state.model_info:
        # 显示模型信息
        st.success(f"✅ {st.session_state.model_info}")
        
        # 数据处理器状态
        if hasattr(st.session_state, 'data_processor') and st.session_state.data_processor:
            processor_messages = {
                'zh': "📋 数据处理器已加载",
                'en': "📋 Data processor loaded",
                'ko': "📋 데이터 프로세서가 로드되었습니다"
            }
            st.info(processor_messages[lang])
        else:
            fallback_messages = {
                'zh': "⚠️ 未找到数据处理器，使用备选方案",
                'en': "⚠️ Data processor not found, using fallback",
                'ko': "⚠️ 데이터 프로세서를 찾을 수 없어 대체 방안을 사용합니다"
            }
            st.warning(fallback_messages[lang])
        
        # 模型验证信息
        # 从load_models中获取的模型重新获取验证信息
        model, _, _ = load_models()
        if model:
            model_validation_info = []
            if hasattr(model, 'feature_count_'):
                if lang == 'zh':
                    model_validation_info.append(f"特征数: {model.feature_count_}")
                elif lang == 'en':
                    model_validation_info.append(f"Features: {model.feature_count_}")
                else:
                    model_validation_info.append(f"특성 수: {model.feature_count_}")
            
            if hasattr(model, 'predict_proba'):
                validation_messages = {
                    'zh': "支持概率预测",
                    'en': "Probability prediction supported",
                    'ko': "확률 예측 지원"
                }
                model_validation_info.append(validation_messages[lang])
            
            if hasattr(model, 'feature_names_in_'):
                if lang == 'zh':
                    model_validation_info.append(f"训练特征: {len(model.feature_names_in_)}")
                elif lang == 'en':
                    model_validation_info.append(f"Training features: {len(model.feature_names_in_)}")
                else:
                    model_validation_info.append(f"훈련 특성: {len(model.feature_names_in_)}")
            
            if model_validation_info:
                st.info("📊 " + " | ".join(model_validation_info))
    else:
        # 如果没有模型信息，显示简单提示
        loading_messages = {
            'zh': "ℹ️ 系统正在初始化，模型状态信息将在加载完成后显示",
            'en': "ℹ️ System initializing, model status will be displayed after loading",
            'ko': "ℹ️ 시스템 초기화 중, 로딩 완료 후 모델 상태가 표시됩니다"
        }
        st.info(loading_messages[lang])

if __name__ == "__main__":
    # 初始化session state
    if 'prediction_done' not in st.session_state:
        st.session_state['prediction_done'] = False
    if 'language' not in st.session_state:
        st.session_state['language'] = 'zh'
    
    main() 