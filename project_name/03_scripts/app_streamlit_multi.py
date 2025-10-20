#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JoyAge | CESD抑郁预测模型 · 三语Web应用
- 复用你给的UI/多语言/样式
- 直接读取 10_experiments/<VER>/web_model_rf 导出的工件
- 个人预测 + SHAP解释（对RF的OneHot后特征做SHAP，并按原始变量聚合展示Top影响因素）
"""

import os, json, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import shap
import streamlit as st
import plotly.graph_objects as go

# =========================
# 配置：模型工件查找
# =========================
def _first_exist(paths):
    for p in paths:
        if p and Path(p).exists():
            return Path(p)
    return None

def locate_artifacts():
    """
    优先级：
    1) 环境变量 JOYAGE_MODEL_DIR
    2) 脚本同级 ./web_model_rf
    3) 项目结构 10_experiments/<VER>/web_model_rf （常见 v2025-10-03 / v2025-10-01）
    """
    here = Path(__file__).resolve().parent
    env_dir = os.getenv("JOYAGE_MODEL_DIR", "").strip() or None
    candidates = [
        env_dir,
        here / "web_model_rf",
        here.parents[1] / "10_experiments" / "v2025-10-03" / "web_model_rf",
        here.parents[1] / "10_experiments" / "v2025-10-01" / "web_model_rf",
    ]
    root = _first_exist(candidates)
    if not root:
        raise FileNotFoundError(
            "未找到模型工件目录。请设置环境变量 JOYAGE_MODEL_DIR 或将 web_model_rf 放在脚本同级/10_experiments/<VER>/web_model_rf。"
        )
    files = {
        "pipe": root / "final_rf_pipeline.joblib",
        "schema": root / "schema.json",
        "threshold": root / "threshold.json",
        "train_cols": root / "train_columns.json",
        "bg": root / "X_bg_raw.npy",
    }
    for k, p in files.items():
        if not p.exists():
            raise FileNotFoundError(f"缺少工件：{p}")
    return files

ART = locate_artifacts()

# =========================
# 多语言字典（来自你给的版本，略有删改）
# =========================
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
        'shap_y_axis': '特征',
        'status': "🔧 系统状态信息",
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
        'shap_y_axis': 'Features',
        'status': "🔧 System Status Information",
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
        'right_panel_desc': '오른쪽 패널: SHAP 설명을 통해 영향 요인 확인',
        'disclaimer': '참고: 이 예측은 참고용이며 의료 진단을 대체할 수 없습니다',
        'model_info': '모델 정보: CHARLS 데이터로 학습, KLOSA 데이터로 외부 검증',
        'shap_chart_title': '🎯 개인 SHAP 특성 중요도 분석',
        'shap_x_axis': 'SHAP 값 (기여도)',
        'shap_y_axis': '특성',
        'status': "🔧 시스템 상태 정보",
    }
}

# （可选）中文/英/韩的人物/变量标签，可按需扩展
FEATURE_LABELS = {
    'zh': {'agey': '年龄', 'ragender': '性别', 'raeducl': '教育水平', 'mstath': '婚姻状况', 'rural': '居住地区'},
    'en': {'agey': 'Age', 'ragender': 'Gender', 'raeducl': 'Education Level', 'mstath': 'Marital Status', 'rural': 'Residence Area'},
    'ko': {'agey': '나이', 'ragender': '성별', 'raeducl': '교육 수준', 'mstath': '혼인 상태', 'rural': '거주 지역'},
}

JOYAGE_INTRO = {
    'zh': "JoyAge悦龄平台：为45岁以上人群提供个性化抑郁风险评估与解释。",
    'en': "JoyAge platform provides personalized depression risk assessment and explanations for adults aged 45+.",
    'ko': "JoyAge 플랫폼은 45세 이상을 위한 개인 맞춤형 우울 위험 평가를 제공합니다.",
}

RISK_THRESHOLDS = {'low': 0.3, 'high': 0.7}

# =========================
# 页面配置 & 样式（复用你给的CSS）
# =========================
st.set_page_config(
    page_title=LANGUAGES['zh']['app_title'],
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""<style>
.main .block-container {padding-top:2rem;padding-bottom:2rem;max-width:1200px;}
.main-header {font-size:3.2rem;font-weight:700;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
-webkit-background-clip:text;-webkit-text-fill-color:transparent;text-align:center;margin-bottom:2.5rem;padding:1rem 0;}
.card-container {background:white;border-radius:15px;padding:1.5rem;margin:1rem 0;box-shadow:0 4px 6px rgba(0,0,0,.07),0 1px 3px rgba(0,0,0,.1);
border:1px solid rgba(0,0,0,.05);}
.section-header {font-size:1.5rem;font-weight:600;color:#2c3e50;margin:1.5rem 0 1rem 0;padding-bottom:.5rem;border-bottom:2px solid #e9ecef;display:flex;align-items:center;}
.section-header::before {content:'';width:4px;height:1.5rem;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);margin-right:.75rem;border-radius:2px;}
.info-box {background:linear-gradient(135deg,#e3f2fd 0%,#bbdefb 100%);border:1px solid #2196f3;border-radius:10px;padding:1rem;margin:1rem 0;font-size:.95rem;color:#1565c0;}
.stButton>button {width:100%;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;border:none;padding:.75rem 1.5rem;border-radius:10px;font-weight:600;font-size:1rem;}
</style>""", unsafe_allow_html=True)

# =========================
# 加载模型与schema
# =========================
@st.cache_resource
def load_artifacts():
    pipe = joblib.load(ART["pipe"])
    schema = json.loads(Path(ART["schema"]).read_text(encoding="utf-8"))
    thr = json.loads(Path(ART["threshold"]).read_text(encoding="utf-8"))
    train_cols = json.loads(Path(ART["train_cols"]).read_text(encoding="utf-8"))
    Xbg = np.load(ART["bg"])
    # 重建背景DF（用于范围与默认值）
    bg_df = pd.DataFrame(Xbg, columns=train_cols)
    return pipe, schema, thr, bg_df

pipe, schema, thr_json, bg_df = load_artifacts()
num_cols = schema.get("num_cols", [])
cat_cols = schema.get("cat_cols", [])
cat_values = schema.get("cat_values", {})  # dict[col] = list of allowed (训练观察到的)取值（注意：训练时是原始类型）

# 取出OHE训练时的类别类型，保证UI选择值能还原为训练类型
def get_ohe_category_type_map():
    # pre -> ("num", ...), ("cat", Pipeline(... OneHotEncoder))
    pre = pipe.named_steps["pre"]
    cat_pipeline = None
    for name, trans, cols in pre.transformers_:
        if name == "cat":
            cat_pipeline = trans
    ohe = cat_pipeline.named_steps["ohe"]
    type_map = {}  # col -> python type (int/str/float...)
    for i, c in enumerate(cat_cols):
        cats = ohe.categories_[i]
        if len(cats) > 0:
            type_map[c] = type(cats[0])
        else:
            type_map[c] = str
    return type_map
CAT_TYPE = get_ohe_category_type_map()

# =========================
# 多语言工具
# =========================
def get_language():
    if 'language' not in st.session_state:
        st.session_state.language = 'zh'
    return st.session_state.language

def T(key):  # 文案
    lang = get_language()
    return LANGUAGES.get(lang, LANGUAGES['en']).get(key, key)

def flabel(col):  # 特征标签
    lang = get_language()
    return FEATURE_LABELS.get(lang, {}).get(col, col)

# =========================
# 输入表单（自动根据 schema 渲染）
# 范围/默认值：从 bg_df 的分位数估一个合理范围
# =========================
def default_num_range(col):
    s = pd.to_numeric(bg_df[col], errors="coerce")
    s = s.dropna()
    if len(s) == 0:
        return (0.0, 100.0, 50.0)
    q1, q5, q95, q99 = s.quantile([.25, .05, .95, .99])
    lo = float(np.floor(min(q1, q5)))
    hi = float(np.ceil(max(q95, q99)))
    mid = float(np.median(s))
    if lo == hi:
        hi = lo + 1.0
    return (lo, hi, mid)

def cast_cat_value(col, val):
    """把UI选中的字符串/数值，转换成训练时的类型（与OHE categories_一致）"""
    typ = CAT_TYPE.get(col, str)
    try:
        if typ is int:
            return int(val)
        if typ is float:
            return float(val)
        return str(val)
    except Exception:
        return val

def build_input_form():
    st.sidebar.header(T('personal_info'))
    inputs = {}
    # 基本信息分组里优先放 agey 等常用字段
    group_order = [
        ('basic', ['agey', 'ragender', 'raeducl', 'mstath', 'rural', 'child', 'hhres']),
        ('health', None),
        ('lifestyle', None),
        ('social', None),
        ('economic', None),
        ('financial', None),
    ]

    # 可见的所有列（schema中的列交集）
    all_cols = [c for c in (cat_cols + num_cols) if c in bg_df.columns]

    with st.sidebar.expander(f"📋 {T('basic_info')}", expanded=True):
        # 年龄（如果存在）
        if 'agey' in all_cols:
            lo, hi, mid = default_num_range('agey')
            inputs['agey'] = st.slider(flabel('agey'), float(lo), float(hi), float(mid))
        # 其它按schema生成
        for col in ['ragender','raeducl','mstath','rural','child','hhres']:
            if col not in all_cols: 
                continue
            if col in cat_cols:
                options = cat_values.get(col, sorted(pd.Series(bg_df[col]).dropna().unique().tolist()))
                # options 可能是字符串，按需显示
                sel = st.selectbox(flabel(col), options, index=0)
                inputs[col] = cast_cat_value(col, sel)
            else:
                lo, hi, mid = default_num_range(col)
                inputs[col] = st.number_input(flabel(col), value=float(mid), min_value=float(lo), max_value=float(hi))
    # 其它分组：自动把剩余列渲染（按类型）
    remaining = [c for c in all_cols if c not in inputs]
    with st.sidebar.expander(f"🏥 {T('health_status')}", expanded=False):
        for col in remaining:
            if col.startswith(("hibpe","diabe","cancre","lunge","hearte","stroke","arthre","livere","painfr","shlta","hlthlm","adlfive")):
                if col in cat_cols:
                    options = cat_values.get(col, sorted(pd.Series(bg_df[col]).dropna().unique().tolist()))
                    sel = st.selectbox(flabel(col), options, index=0)
                    inputs[col] = cast_cat_value(col, sel)
    with st.sidebar.expander(f"🚭 {T('lifestyle')}", expanded=False):
        for col in remaining:
            if col in ['drinkev','smokev','stayhospital','fall']:
                if col in cat_cols:
                    options = cat_values.get(col, sorted(pd.Series(bg_df[col]).dropna().unique().tolist()))
                    sel = st.selectbox(flabel(col), options, index=0)
                    inputs[col] = cast_cat_value(col, sel)
    with st.sidebar.expander(f"👥 {T('social_support')}", expanded=False):
        for col in remaining:
            if col in ['hipriv','momliv','dadliv','lvnear','kcntf','socwk','ftrsp','ftrkids','ramomeducl','radadeducl']:
                if col in cat_cols:
                    options = cat_values.get(col, sorted(pd.Series(bg_df[col]).dropna().unique().tolist()))
                    sel = st.selectbox(flabel(col), options, index=0)
                    inputs[col] = cast_cat_value(col, sel)
    with st.sidebar.expander(f"💰 {T('economic_status')}", expanded=False):
        for col in remaining:
            if col in ['work','pubpen','peninc']:
                if col in cat_cols:
                    options = cat_values.get(col, sorted(pd.Series(bg_df[col]).dropna().unique().tolist()))
                    sel = st.selectbox(flabel(col), options, index=0)
                    inputs[col] = cast_cat_value(col, sel)
        # 数值型收入支出
        for col in ['comparable_hexp','comparable_exp','comparable_itearn','comparable_frec','comparable_tgiv','comparable_ipubpen']:
            if col in num_cols:
                lo, hi, mid = default_num_range(col)
                inputs[col] = st.number_input(flabel(col), value=float(mid), min_value=float(lo), max_value=float(hi))

    # 对其余未覆盖的列做兜底渲染
    covered = set(inputs.keys())
    others = [c for c in all_cols if c not in covered]
    if others:
        with st.sidebar.expander("🧩 Others / 其它", expanded=False):
            for col in others:
                if col in cat_cols:
                    options = cat_values.get(col, sorted(pd.Series(bg_df[col]).dropna().unique().tolist()))
                    sel = st.selectbox(flabel(col), options, index=0, key=f"other_{col}")
                    inputs[col] = cast_cat_value(col, sel)
                else:
                    lo, hi, mid = default_num_range(col)
                    inputs[col] = st.number_input(flabel(col), value=float(mid), min_value=float(lo), max_value=float(hi), key=f"other_{col}")
    return inputs

# =========================
# 预测 & SHAP
# =========================
def predict_proba_df(xrow: dict) -> float:
    df = pd.DataFrame([xrow], columns=bg_df.columns)  # 确保列顺序
    # 缺失列补0
    for c in [c for c in (cat_cols+num_cols) if c not in df.columns]:
        df[c] = 0
    df = df[bg_df.columns.intersection(cat_cols + num_cols)]
    proba = pipe.predict_proba(df)[0,1]
    return float(proba), df

def compute_shap(df_one: pd.DataFrame):
    """对RF进行TreeExplainer；在预处理后空间计算SHAP；再按原始列聚合可视化"""
    pre = pipe.named_steps["pre"]
    rf  = pipe.named_steps["rf"]
    X_enc = pre.transform(df_one)  # numpy
    # 变换后特征名
    try:
        feat_enc_names = pre.get_feature_names_out()
    except Exception:
        feat_enc_names = [f"f{i}" for i in range(X_enc.shape[1])]
    # SHAP
    explainer = shap.TreeExplainer(rf)
    shap_vals = explainer.shap_values(X_enc)
    # 二分类：取正类
    if isinstance(shap_vals, list) and len(shap_vals) == 2:
        shap_pos = shap_vals[1]
    elif isinstance(shap_vals, np.ndarray):
        shap_pos = shap_vals
    else:
        shap_pos = shap_vals
    row = shap_pos[0]  # 单样本
    # 聚合到原始列：对 OHE 的同一前缀求和
    def orig_name(enc):
        # OneHotEncoder 输出通常是 "colname_value"；数字列保持 "colname"
        return enc.split("_", 1)[0]
    agg = {}
    for v, n in zip(row, feat_enc_names):
        k = orig_name(n)
        agg[k] = agg.get(k, 0.0) + float(v)
    # 排序Top10
    agg_items = sorted(agg.items(), key=lambda kv: abs(kv[1]), reverse=True)[:10]
    return agg_items

def risk_card(prob):
    lang = get_language()
    if prob <= RISK_THRESHOLDS['low']:
        level, color, icon = ('低风险','green','✅') if lang=='zh' else ('Low Risk','green','✅') if lang=='en' else ('낮은 위험','green','✅')
    elif prob <= RISK_THRESHOLDS['high']:
        level, color, icon = ('中等风险','#f1c40f','⚠️') if lang=='zh' else ('Medium Risk','#f1c40f','⚠️') if lang=='en' else ('중간 위험','#f1c40f','⚠️')
    else:
        level, color, icon = ('高风险','#e74c3c','🚨') if lang=='zh' else ('High Risk','#e74c3c','🚨') if lang=='en' else ('높은 위험','#e74c3c','🚨')
    st.markdown(f"""
    <div class="card-container" style="border-left:6px solid {color}">
      <div style="text-align:center;">
        <h3 style="margin:8px 0;">{icon} {T('assessment_result')} · {level}</h3>
        <p style="font-size:26px;font-weight:700;">{T('depression_probability')}: {prob*100:.1f}%</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

def shap_bar(agg_items):
    # agg_items: [(feature, shap_sum), ...]
    labels = [flabel(k) for k,_ in agg_items]
    vals   = [v for _,v in agg_items]
    colors = ['#FF0D57' if v>0 else '#1F88F5' for v in vals]
    fig = go.Figure(go.Bar(
        y=labels[::-1], x=vals[::-1],
        orientation='h', marker_color=colors[::-1],
        text=[f"{v:+.3f}" for v in vals[::-1]], textposition='outside'
    ))
    fig.update_layout(
        title=T('shap_chart_title'),
        xaxis_title=T('shap_x_axis'),
        yaxis_title=T('shap_y_axis'),
        height=520, showlegend=False, margin=dict(l=20,r=20,t=30,b=20)
    )
    return fig

# =========================
# 页面
# =========================
def render_hero():
    lang = get_language()
    title = {"zh":"目标愿景", "en":"Goal & Mission", "ko":"목표 비전"}[lang]
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg,#f5f7ff 0%,#eef2ff 100%);border-radius:16px;padding:18px 22px;margin-bottom:16px;border:1px solid rgba(102,126,234,.15)">
          <div style="font-size:26px;font-weight:800;color:#2c3e50;">🧠 {title}</div>
          <div style="margin-top:8px;font-size:15px;color:#475569">{JOYAGE_INTRO.get(lang, JOYAGE_INTRO['en'])}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def main():
    lang = get_language()
    # 右上角语言切换
    c1, c2 = st.columns([6,1])
    with c1:
        st.markdown(f'<h1 class="main-header">{T("app_title")}</h1>', unsafe_allow_html=True)
    with c2:
        options = ['zh','en','ko']
        names = {'zh':'🇨🇳 中文','en':'🇺🇸 English','ko':'🇰🇷 한국어'}
        idx = options.index(lang) if lang in options else 0
        new_lang = st.selectbox(T('language_select'), options, index=idx, format_func=lambda x: names[x], key="lang_sel")
        if new_lang != lang:
            st.session_state.language = new_lang
            st.rerun()

    render_hero()

    # 两列布局：左输入，右解释
    col1, col2 = st.columns([1,2])

    with col1:
        st.markdown(f'<div class="card-container"><div class="section-header">{T("prediction_results")}</div>', unsafe_allow_html=True)
        inputs = build_input_form()
        st.markdown(f"""
        <div class="info-box">
          <strong>ℹ️ {T('usage_instructions')}</strong><br>
          {T('left_panel_desc')}<br>
          <em>{T('disclaimer')}</em>
        </div>
        """, unsafe_allow_html=True)

        if st.button(T('predict_button'), type="primary"):
            try:
                prob, xdf = predict_proba_df(inputs)
                st.session_state["prob"] = prob
                st.session_state["xdf"]  = xdf
                # SHAP
                agg_items = compute_shap(xdf)
                st.session_state["shap_agg"] = agg_items
            except Exception as e:
                st.error(f"预测失败：{e}")

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(f'<div class="card-container"><div class="section-header">{T("personal_analysis")}</div>', unsafe_allow_html=True)
        if "prob" in st.session_state:
            risk_card(st.session_state["prob"])
            if "shap_agg" in st.session_state and st.session_state["shap_agg"]:
                fig = shap_bar(st.session_state["shap_agg"])
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown(f"""
            <div class="info-box">
              <strong>👈 {T('input_info_prompt')}</strong><br>{T('right_panel_desc')}
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader(T('status'))
    st.success(f"✅ 模型：{ART['pipe'].name} | schema: {ART['schema'].name} | ver={schema.get('ver_out','?')}")
    st.info(f"📈 阈值(Youden from val)：{thr_json.get('threshold', 'N/A'):.3f} | AUC(train)={thr_json.get('auc_train','?')} | AUC(val)={thr_json.get('auc_val','?')}")

if __name__ == "__main__":
    if 'language' not in st.session_state: st.session_state['language'] = 'zh'
    main()
