# -*- coding: utf-8 -*-
r"""
14_final_eval_s7.py — 最终评估（S7）
改动要点：
1) 只开一层并行：外层 GridSearchCV(n_jobs=N_JOBS)，模型内部统一单线程（n_jobs=1 / thread_count=1），避免并行套娃
2) 可选启用 GPU（从 config.yaml 的 eval.use_gpu 读取；XGB/CatBoost/LGBM 自动切GPU）
3) 树模型用 OrdinalEncoder，线性模型(LR)用 OneHotEncoder
4) LightGBM 'large' 网格精简为 large-lite，显著减少组合数
5) 防“暗并行”：限制 OMP/MKL/OPENBLAS 线程为 1
6) 可选断点续跑（eval.resume=true 时，已完成的模型会跳过）
"""

# ---- 限制数值库多线程，避免暗并行（放在最前）----
import os
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings, json, joblib
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import yaml
import multiprocessing as mp
from scipy import stats

# ===================== 统一读取配置 =====================
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

CFG = yaml.safe_load(open(repo_root() / "07_config" / "config.yaml", "r", encoding="utf-8"))

PROJECT_ROOT = Path(CFG["paths"]["project_root"])
VER_IN  = CFG.get("run_id_in",  CFG.get("run_id"))
VER_OUT = CFG.get("run_id_out", CFG.get("run_id"))

DATA_DIR   = PROJECT_ROOT / "02_processed_data" / VER_IN
FROZEN_DIR = DATA_DIR / "frozen"
SPLIT_DIR  = DATA_DIR / "splits"
S4_DIR     = PROJECT_ROOT / "10_experiments" / VER_OUT / "feature_sweep"

# 固定插补：mean_mode（带 y）
MEAN_MODE_XY = FROZEN_DIR / "charls_mean_mode_Xy.csv"

# 外部集（可由 YAML.paths 覆写）
ext_cfg = CFG.get("paths", {}) or {}
EXT_MAIN     = Path(ext_cfg.get("external_main_Xy",     FROZEN_DIR / "klosa_main_Xy.csv"))
EXT_TRANSFER = Path(ext_cfg.get("external_transfer_Xy", FROZEN_DIR / "klosa_transfer_Xy.csv"))

# 索引
TRAIN_IDX = SPLIT_DIR / "charls_train_idx.csv"
VAL_IDX   = SPLIT_DIR / "charls_val_idx.csv"
TEST_IDX  = SPLIT_DIR / "charls_test_idx.csv"

# 标签与 ID 列
Y_NAME  = (CFG.get("outcome", {}) or {}).get("name", "depression_bin")
ID_COLS = list(CFG.get("id_cols") or []) or ["ID", "householdID"]

# 模型列表（禁用 catboost）
_models_cfg = list((CFG.get("eval", {}) or {}).get("models", []))
if not _models_cfg:
    _models_cfg = ["lr", "rf", "extra_trees", "gb", "adaboost", "lgb", "xgb"]
MODEL_LIST = [m for m in _models_cfg if str(m).lower() != "catboost"]

# 评估参数
eval_cfg   = (CFG.get("eval", {}) or {})
final_cfg  = (eval_cfg.get("final", {}) or {})
GRID_LEVEL = "medium"  # 默认使用medium级别的参数网格
CV_N_SPLITS= int(final_cfg.get("cv_splits", 5))
RESUME     = True  # 启用断点续跑；若需强制重算请设为 False 或删除已有cv结果
USE_GPU    = True  # 强制启用GPU加速

_cpu = mp.cpu_count() or 4
N_JOBS = int(final_cfg.get("n_jobs", max(1, min(4, _cpu - 1))))  # 外层并行（保守，Windows更稳）

OUT_DIR = PROJECT_ROOT / "10_experiments" / VER_OUT / "final_eval_s7"
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "plots").mkdir(exist_ok=True)
(OUT_DIR / "models").mkdir(exist_ok=True)

print(f"[info] CPU={_cpu} | N_JOBS(GridSearchCV)={N_JOBS} | GRID_LEVEL={GRID_LEVEL} | USE_GPU={USE_GPU} | RESUME={RESUME}")

# ===================== sklearn / ML 基础 =====================
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, precision_score,
    recall_score, f1_score, brier_score_loss, roc_curve
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression

def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # sklearn >=1.4
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)         # sklearn <1.4

from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
)

# 可选库
try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except Exception:
    _HAS_LGBM = False

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

try:
    from catboost import CatBoostClassifier
    _HAS_CAT = True
except Exception:
    _HAS_CAT = False


# ===================== 工具 =====================
def load_idx_csv(path: Optional[Path]) -> Optional[np.ndarray]:
    if path and Path(path).exists():
        s = pd.read_csv(path).iloc[:, 0]
        return s.values.astype(int)
    return None

def safe_read_csv(p: Optional[Path]) -> Optional[pd.DataFrame]:
    if p and Path(p).exists():
        return pd.read_csv(p)
    return None

def read_yaml_variable_types() -> Tuple[List[str], List[str], List[str]]:
    vt = (CFG.get("preprocess", {}) or {}).get("variable_types", {}) or {}
    num_cols = list(vt.get("continuous", []) or [])
    mul_cols = list(vt.get("categorical", []) or [])
    return num_cols, [], mul_cols

def infer_feature_types(df: pd.DataFrame, y_name: str,
                        init_num: Optional[List[str]]=None,
                        init_mul: Optional[List[str]]=None,
                        id_cols: Optional[List[str]]=None) -> Tuple[List[str], List[str], List[str]]:
    if id_cols is None: id_cols = []
    exclude = set(id_cols + [y_name])
    feats = [c for c in df.columns if c not in exclude]

    init_num = set(init_num or [])
    init_mul = set(init_mul or [])

    num_cols, bin_cols, mul_cols = [], [], []
    for c in feats:
        s = df[c]
        if c in init_num:
            num_cols.append(c); continue
        if c in init_mul:
            nunq = s.dropna().nunique()
            (bin_cols if nunq == 2 else mul_cols).append(c); continue

        if pd.api.types.is_numeric_dtype(s):
            nunq = s.dropna().nunique()
            if nunq == 2:
                bin_cols.append(c)
            elif pd.api.types.is_integer_dtype(s) and nunq <= 10:
                (bin_cols if nunq == 2 else mul_cols).append(c)
            else:
                num_cols.append(c)
        else:
            mul_cols.append(c)
    return num_cols, bin_cols, mul_cols

def build_preprocessor(all_num: List[str], all_bin: List[str], all_mul: List[str],
                       selected: Optional[List[str]] = None,
                       use_ohe: bool = True) -> ColumnTransformer:
    """
    预处理策略：
    - use_ohe=True（线性模型，如 LR）: 数值标准化 + 类别 One-Hot
    - use_ohe=False（树模型）: 数值直通 + 类别 Ordinal 编码；二值直通
    """
    if selected is not None:
        s = set(selected)
        num_cols = [c for c in all_num if c in s]
        bin_cols = [c for c in all_bin if c in s]
        mul_cols = [c for c in all_mul if c in s]
    else:
        num_cols, bin_cols, mul_cols = all_num, all_bin, all_mul

    cat_transformer = (make_ohe() if use_ohe
                       else OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    # 数值列：仅在线性模型中标准化；树模型直通，避免不必要的变换
    num_transformer = (StandardScaler(with_mean=False) if use_ohe else "passthrough")

    return ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
            ("bin", "passthrough", bin_cols),
            ("mul", cat_transformer, mul_cols),
        ],
        remainder="drop"
    )

# 让不同数据集在列上与“选中特征”对齐：数值/二值补0，分类型补 "UNK"
def align_columns(X: Optional[pd.DataFrame],
                  sel_num: List[str], sel_bin: List[str], sel_mul: List[str]) -> Optional[pd.DataFrame]:
    if X is None: return None
    Xc = X.copy()
    for c in sel_num + sel_bin + sel_mul:
        if c not in Xc.columns:
            if c in sel_mul:
                Xc[c] = "UNK"
            else:
                Xc[c] = 0.0
    cols = sel_num + sel_bin + sel_mul
    return Xc[cols]


# ===================== 模型与网格 =====================
def _class_ratio(y: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    pos = y.sum(); neg = len(y) - pos
    return (neg / max(pos, 1))

def make_model(model_key: str, y_train: np.ndarray):
    """只开一层并行：模型内部统一单线程；可选启用GPU。"""
    mk = model_key.lower()
    if mk == "lr":
        return LogisticRegression(solver="liblinear", penalty="l2",
                                  max_iter=2000, class_weight="balanced", random_state=42)

    elif mk == "rf":
        return RandomForestClassifier(class_weight="balanced", n_jobs=1, random_state=42)

    elif mk == "extra_trees":
        return ExtraTreesClassifier(class_weight="balanced", n_jobs=1, random_state=42)

    elif mk == "gb":
        return GradientBoostingClassifier(random_state=42)

    elif mk == "adaboost":
        return AdaBoostClassifier(random_state=42)

    elif mk == "lgb":
        if _HAS_LGBM:
            params = dict(objective="binary", class_weight="balanced",
                          random_state=42, n_jobs=1)
            if USE_GPU:
                # 不同版本可能是 device_type 或 device
                params["device_type"] = "gpu"
                # params["device"] = "gpu"  # 若上面报错可换这一行
            return LGBMClassifier(**params)
        return GradientBoostingClassifier(random_state=42)

    elif mk == "xgb":
        if _HAS_XGB:
            params = dict(
                objective="binary:logistic",
                random_state=42,
                n_jobs=1,
                scale_pos_weight=_class_ratio(y_train),
                verbosity=0
            )
            if USE_GPU:
                params.update(tree_method="gpu_hist", predictor="gpu_predictor")
            else:
                # Use fast histogram algorithm on CPU to speed up training
                params.update(tree_method="hist")
            return XGBClassifier(**params)
        return GradientBoostingClassifier(random_state=42)

    elif mk == "catboost":
        if _HAS_CAT:
            w1 = _class_ratio(y_train)
            params = dict(
                loss_function="Logloss", eval_metric="AUC",
                class_weights=[1.0, w1], random_seed=42,
                verbose=100, od_type="Iter", od_wait=50
            )
            if USE_GPU:
                params.update(task_type="GPU", devices="0")
            else:
                params.update(thread_count=1)   # CPU 模式限单线程
            return CatBoostClassifier(**params)
        return None

    else:
        raise ValueError(f"Unsupported model: {model_key}")

def build_param_grid(model_key: str, level: str) -> Dict[str, List]:
    # 特殊处理：LightGBM 与 CatBoost 均强制使用 small 网格；其余模型使用指定级别
    if model_key in ["lgb", "catboost"]:
        lv = "small"
        print(f"[info] 对 {model_key} 强制使用 small 参数网格，其他模型使用 {level} 网格")
    else:
        lv = (level or "small").lower()

    if model_key == "lr":
        C = {"small":[0.1,1,10],"medium":[0.05,0.1,0.5,1,5,10],
             "large":[0.005,0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20]}[lv]
        return {"clf__penalty":["l1","l2"], "clf__C":C}

    if model_key == "rf":
        return {"clf__n_estimators":[300,600,900,1200] if lv!="small" else [300,600,900],
                "clf__max_depth":[None,10,20,30] if lv!="small" else [None,10,20],
                "clf__min_samples_leaf":[1,2,4],
                "clf__min_samples_split":[2,5,10] if lv!="small" else [2,5],
                "clf__max_features":["sqrt","log2",0.5,1.0] if lv=="large" else ["sqrt","log2",0.5]}

    if model_key == "extra_trees":
        return {"clf__n_estimators":[300,600,900,1200] if lv!="small" else [300,600,900],
                "clf__max_depth":[None,10,20,30] if lv!="small" else [None,10,20],
                "clf__min_samples_leaf":[1,2,4],
                "clf__min_samples_split":[2,5,10] if lv!="small" else [2,5],
                "clf__max_features":["sqrt","log2",0.5,1.0] if lv=="large" else ["sqrt","log2",0.5]}

    if model_key == "gb":
        return {"clf__n_estimators":[200,400,600,800,1000] if lv=="large" else ([200,400,800] if lv=="small" else [200,400,600,800]),
                "clf__learning_rate":[0.01,0.03,0.05,0.1] if lv!="small" else [0.05,0.1],
                "clf__max_depth":[2,3,4,5] if lv!="small" else [2,3,4],
                "clf__subsample":[0.7,0.8,0.9,1.0] if lv!="small" else [0.8,1.0],
                "clf__min_samples_leaf":[1,2,3,5] if lv=="large" else [1,3,5]}

    if model_key == "adaboost":
        return {"clf__n_estimators":[200,400,600,800,1000,1200] if lv=="large" else ([200,400,800] if lv=="small" else [200,400,600,800,1000]),
                "clf__learning_rate":[0.01,0.03,0.05,0.1,0.2] if lv!="small" else [0.05,0.1,0.2],
                "clf__algorithm":["SAMME","SAMME.R"]}

    if model_key == "lgb" and _HAS_LGBM:
        if lv == "small":
            return {"clf__n_estimators":[300,600],        # 2个值
                    "clf__num_leaves":[31,63],            # 2个值
                    "clf__max_depth":[-1,10],             # 2个值
                    "clf__learning_rate":[0.05,0.1],      # 2个值
                    "clf__subsample":[0.8,1.0],           # 2个值
                    "clf__colsample_bytree":[0.8,1.0],    # 2个值
                    "clf__reg_lambda":[0,5],              # 2个值
                    "clf__min_child_samples":[10,30]}     # 2个值
        if lv == "medium":
            return {"clf__n_estimators":[300,600,900],
                    "clf__num_leaves":[31,63,127],
                    "clf__max_depth":[-1,5,10,15],
                    "clf__learning_rate":[0.01,0.03,0.05,0.1],
                    "clf__subsample":[0.7,0.8,0.9,1.0],
                    "clf__colsample_bytree":[0.6,0.8,1.0],
                    "clf__reg_lambda":[0,1,3,5,10],
                    "clf__min_child_samples":[10,20,30]}
        # large-lite（精简）
        return {"clf__n_estimators":[300,600],
                "clf__num_leaves":[31,63],
                "clf__max_depth":[-1,8,12],
                "clf__learning_rate":[0.03,0.1],
                "clf__subsample":[0.8,1.0],
                "clf__colsample_bytree":[0.8,1.0],
                "clf__reg_lambda":[0,5],
                "clf__min_child_samples":[10,20]}

    if model_key == "xgb" and _HAS_XGB:
        # XGB grids trimmed to speed up search while keeping key knobs
        if lv == "small":
            return {
                "clf__n_estimators": [200, 400],
                "clf__max_depth": [3, 4],
                "clf__learning_rate": [0.05, 0.1],
                "clf__subsample": [0.8, 1.0],
                "clf__colsample_bytree": [0.8, 1.0],
                "clf__reg_lambda": [0, 3],
                "clf__min_child_weight": [1, 3],
                "clf__gamma": [0]
            }
        if lv == "medium":
            return {
                "clf__n_estimators": [300, 600],
                "clf__max_depth": [3, 4, 5],
                "clf__learning_rate": [0.05, 0.1],
                "clf__subsample": [0.8, 1.0],
                "clf__colsample_bytree": [0.8, 1.0],
                "clf__reg_lambda": [0, 3],
                "clf__reg_alpha": [0, 0.1],
                "clf__min_child_weight": [1, 3],
                "clf__gamma": [0, 0.1]
            }
        # large-lite (trimmed)
        return {
            "clf__n_estimators": [300, 600],
            "clf__max_depth": [3, 4],
            "clf__learning_rate": [0.05, 0.1],
            "clf__subsample": [0.8, 1.0],
            "clf__colsample_bytree": [0.8, 1.0],
            "clf__reg_lambda": [0, 3],
            "clf__reg_alpha": [0],
            "clf__min_child_weight": [1, 3],
            "clf__gamma": [0]
        }

    if model_key == "catboost" and _HAS_CAT:
        if lv == "small":
            # 进一步提速：降低 iterations 候选以缩短单次拟合时间
            return {
                "clf__iterations": [300, 600],
                "clf__depth": [4, 6],
                "clf__learning_rate": [0.03, 0.06],
                "clf__l2_leaf_reg": [1, 5],
                "clf__bagging_temperature": [0, 1],
            }
        if lv == "medium":
            return {"clf__iterations":[600,1000,1400],
                    "clf__depth":[4,6,8,10],
                    "clf__learning_rate":[0.02,0.04,0.06],
                    "clf__l2_leaf_reg":[1,3,5,7],
                    "clf__bagging_temperature":[0,0.5,1,2]}
        return {"clf__iterations":[800,1200,1600],
                "clf__depth":[4,6,8,10],
                "clf__learning_rate":[0.01,0.03,0.06],
                "clf__l2_leaf_reg":[1,3,5,7],
                "clf__bagging_temperature":[0,0.5,1,2]}
    return {}

# ===================== 指标与校准 =====================
def compute_9_metrics(y_true: np.ndarray, p: np.ndarray, thr: float = 0.5, compute_ci: bool = False) -> Dict[str, Union[float, str]]:
    """
    计算9种评估指标，可选计算95%置信区间
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p).clip(1e-8, 1-1e-8)
    yhat = (p >= thr).astype(int)
    tn = np.sum((y == 0) & (yhat == 0))
    fp = np.sum((y == 0) & (yhat == 1))
    fn = np.sum((y == 1) & (yhat == 0))
    tp = np.sum((y == 1) & (yhat == 1))

    def _safe(a, b): return float(a) / max(float(b), 1e-12)

    try:
        auroc = roc_auc_score(y, p)
    except Exception:
        auroc = float("nan")
    try:
        auprc = average_precision_score(y, p)
    except Exception:
        auprc = float("nan")

    metrics = {
        "AUROC": auroc,
        "AUPRC": auprc,
        "Accuracy": accuracy_score(y, yhat),
        "Precision": precision_score(y, yhat, zero_division=0),
        "Recall": recall_score(y, yhat, zero_division=0),
        "F1_Score": f1_score(y, yhat, zero_division=0),
        "Brier_Score": brier_score_loss(y, p),
        "Specificity": _safe(tn, tn + fp),
        "NPV": _safe(tn, tn + fn),
    }

    # 如果需要计算置信区间
    if compute_ci and len(y) > 50:  # 样本量充足时才计算CI
        bootstrap_ci = compute_bootstrap_ci(y, p, thr, n_bootstrap=1000)
        
        # 合并置信区间到结果中
        for metric, value in metrics.items():
            if metric in bootstrap_ci and not np.isnan(value):
                low, high = bootstrap_ci[metric]
                metrics[metric] = f"{value:.4f} [{low:.4f}, {high:.4f}]"
            else:
                metrics[metric] = f"{value:.4f}"
    
    return metrics

def compute_bootstrap_ci(y: np.ndarray, p: np.ndarray, thr: float = 0.5, n_bootstrap: int = 1000, 
                        ci_level: float = 0.95) -> Dict[str, Tuple[float, float]]:
    """
    使用Bootstrap方法计算评估指标的置信区间
    """
    n_samples = len(y)
    results = {
        "AUROC": [], "AUPRC": [], "Accuracy": [], "Precision": [], 
        "Recall": [], "F1_Score": [], "Brier_Score": [], 
        "Specificity": [], "NPV": []
    }
    
    # 运行bootstrap
    rng = np.random.RandomState(42)  # 固定随机种子以保证可复现
    
    for _ in range(n_bootstrap):
        # 有放回采样
        indices = rng.randint(0, n_samples, n_samples)
        y_boot = y[indices]
        p_boot = p[indices]
        
        # 计算所有指标
        try:
            metrics = compute_9_metrics(y_boot, p_boot, thr, compute_ci=False)
            
            # 保存每个bootstrap样本的指标
            for metric, value in metrics.items():
                if not np.isnan(value):
                    results[metric].append(value)
        except Exception:
            pass  # 忽略可能的错误（如类别缺失等）
    
    # 计算置信区间
    ci_results = {}
    alpha = (1 - ci_level) / 2
    
    for metric, values in results.items():
        if len(values) > n_bootstrap * 0.9:  # 至少90%的bootstrap样本有有效结果
            ci_lower = np.percentile(values, alpha * 100)
            ci_upper = np.percentile(values, (1 - alpha) * 100)
            ci_results[metric] = (ci_lower, ci_upper)
    
    return ci_results

def calib_slope_intercept(y, p):
    eps = 1e-8
    z = np.log(np.clip(p, eps, 1-eps) / np.clip(1-p, eps, 1-eps))
    lr = LogisticRegression(solver="liblinear").fit(z.reshape(-1,1), np.asarray(y).astype(int))
    return float(lr.coef_[0,0]), float(lr.intercept_[0])

def make_isotonic_calibrated_pipe(fitted_pipe: Pipeline, X_cal: pd.DataFrame, y_cal: np.ndarray) -> Pipeline:
    pre = fitted_pipe.named_steps["pre"]
    clf = fitted_pipe.named_steps["clf"]
    try:
        cal = CalibratedClassifierCV(estimator=clf, method="isotonic", cv="prefit")
    except TypeError:
        cal = CalibratedClassifierCV(base_estimator=clf, method="isotonic", cv="prefit")
    X_cal_t = pre.transform(X_cal)
    cal.fit(X_cal_t, y_cal)
    return Pipeline([("pre", pre), ("cal", cal)])

def _as_proba(model_or_pipe, X):
    if hasattr(model_or_pipe, "predict_proba"):
        return model_or_pipe.predict_proba(X)[:, 1]
    if hasattr(model_or_pipe, "decision_function"):
        z = model_or_pipe.decision_function(X)
        return 1.0 / (1.0 + np.exp(-z))
    return model_or_pipe.predict(X).astype(float)

# ===================== 绘图（多模型：ROC/校准/DCA） =====================
import matplotlib.pyplot as plt

def decision_curve_analysis(y_true: np.ndarray, p: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p)
    N = len(y)
    nb = []
    for t in thresholds:
        yhat = (p >= t).astype(int)
        TP = np.sum((yhat == 1) & (y == 1))
        FP = np.sum((yhat == 1) & (y == 0))
        nb.append((TP/N) - (FP/N) * (t/(1.0 - t)))
    return np.asarray(nb, dtype=float)

def plot_roc_multi(preds, out_png: Path, title: str):
    fig = plt.figure(figsize=(7.2, 5.2))
    ax = fig.add_subplot(111)
    for model, scen_map in preds.items():
        if not scen_map: continue
        (y, p) = list(scen_map.values())[0]
        try:
            fpr, tpr, _ = roc_curve(y.astype(int), p)
            auc = roc_auc_score(y, p)
            ax.plot(fpr, tpr, label=f"{model.upper()} (AUC={auc:.3f})")
        except Exception:
            continue
    ax.plot([0,1],[0,1],'--', color='gray', linewidth=1)
    ax.set_xlabel("1 - Specificity"); ax.set_ylabel("Sensitivity")
    ax.set_title(title); ax.grid(True, linestyle="--", alpha=0.35); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(out_png, dpi=220); fig.savefig(out_png.with_suffix(".pdf")); plt.close(fig)

def plot_calibration_multi(preds, out_png: Path, title: str, n_bins: int = 10):
    fig = plt.figure(figsize=(7.2, 5.2)); ax = fig.add_subplot(111)
    drew = False
    for model, scen_map in preds.items():
        if not scen_map: continue
        (y, p) = list(scen_map.values())[0]
        try:
            frac_pos, mean_pred = calibration_curve(y.astype(int), p, n_bins=n_bins, strategy="quantile")
            ax.plot(mean_pred, frac_pos, marker='o', label=model.upper())
            drew = True
        except Exception:
            continue
    ax.plot([0,1],[0,1],'--', color='gray', linewidth=1)
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed fraction")
    ax.set_title(title); ax.grid(True, linestyle="--", alpha=0.35)
    if drew: ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(out_png, dpi=220); fig.savefig(out_png.with_suffix(".pdf")); plt.close(fig)

def plot_dca_multi(preds, out_png: Path, title: str):
    thresholds = np.linspace(0.01, 0.99, 99)
    fig = plt.figure(figsize=(7.2, 5.2)); ax = fig.add_subplot(111)
    ax.plot(thresholds, np.zeros_like(thresholds), '--', color='gray', linewidth=1, label="Treat None")
    first = next(iter(preds.values()), None)
    if first:
        y_base = list(first.values())[0][0].astype(int)
        prev = float(y_base.mean()) if len(y_base) else 0.0
        treat_all = prev - (1 - prev) * thresholds/(1 - thresholds)
        ax.plot(thresholds, treat_all, ':', color='black', linewidth=1, label="Treat All")
    for model, scen_map in preds.items():
        if not scen_map: continue
        (y, p) = list(scen_map.values())[0]
        try:
            nb = decision_curve_analysis(y, p, thresholds)
            ax.plot(thresholds, nb, label=model.upper())
        except Exception:
            continue
    ax.set_xlabel("Threshold probability"); ax.set_ylabel("Net benefit"); ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35); ax.legend(loc="best")
    fig.tight_layout(); fig.savefig(out_png, dpi=220); fig.savefig(out_png.with_suffix(".pdf")); plt.close(fig)

# ===================== 读取 S4 最佳特征 =====================
def pick_features_for_model(model_key: str, all_feats: List[str]) -> List[str]:
    f_best = S4_DIR / f"S4_RFE_{model_key}_best_features.csv"
    if f_best.exists():
        feats = pd.read_csv(f_best)["feature"].dropna().astype(str).tolist()
        feats = [f for f in feats if f in all_feats]
        if len(feats): return feats

    f_curve = S4_DIR / f"rfe_curve_{model_key}.csv"
    if f_curve.exists():
        dfc = pd.read_csv(f_curve)
        if "scenario" in dfc.columns:
            sub = dfc[dfc["scenario"]=="internal_val"].copy()
            if not len(sub): sub = dfc[dfc["scenario"]=="internal_test"].copy()
            if len(sub):
                sub = sub.sort_values(["AUROC","AUPRC","k"], ascending=[False,False,True])
                k_best = int(sub.iloc[0]["k"])
                return all_feats[:k_best]
    return list(all_feats)

# ===================== 主流程 =====================
def _youden_threshold(y: np.ndarray, p: np.ndarray) -> float:
    fpr, tpr, th = roc_curve(y.astype(int), p)
    j = tpr - fpr
    k = int(np.nanargmax(j))
    return float(th[k])

def main():
    # 读数据
    df = pd.read_csv(MEAN_MODE_XY)
    assert Y_NAME in df.columns, f"{Y_NAME} 不在 {MEAN_MODE_XY}"

    tr_idx = load_idx_csv(TRAIN_IDX); va_idx = load_idx_csv(VAL_IDX); te_idx = load_idx_csv(TEST_IDX)
    ext_main = safe_read_csv(EXT_MAIN)
    ext_tran = safe_read_csv(EXT_TRANSFER)

    # 变量类型：YAML 种子 + 自动
    try:
        init_num, _bin_unused, init_mul = read_yaml_variable_types()
    except Exception:
        init_num, init_mul = None, None

    all_num, all_bin, all_mul = infer_feature_types(df, Y_NAME, init_num, init_mul, ID_COLS)
    all_feats = all_num + all_bin + all_mul

    # 划分
    X_train_all = df.loc[tr_idx, all_feats]; y_train = df.loc[tr_idx, Y_NAME].astype(int).values
    X_val_all   = df.loc[va_idx, all_feats] if va_idx is not None else None
    y_val       = df.loc[va_idx, Y_NAME].astype(int).values if va_idx is not None else None
    X_test_all  = df.loc[te_idx, all_feats] if te_idx is not None else None
    y_test      = df.loc[te_idx, Y_NAME].astype(int).values if te_idx is not None else None

    # 创建数据保存目录
    data_dir = OUT_DIR / "data"
    data_dir.mkdir(exist_ok=True)
    
    # 收集器
    metrics_file = data_dir / "metrics_rows.json"
    probs_file = data_dir / "prob_collectors.json"
    thresholds_file = data_dir / "thresholds_from_val.json"
    models_params_file = data_dir / "model_params_record.json"
    
    # 尝试加载已有数据（用于断点续跑）
    metrics_rows = []
    prob_collectors = {k: [] for k in ["internal_train","internal_val","internal_test","external_main","external_transfer"]}
    thresholds_from_val = {}
    model_params_record = {}
    
    if RESUME and metrics_file.exists():
        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                metrics_rows = json.load(f)
            print(f"[info] 已加载现有指标数据：{len(metrics_rows)}条记录")
        except Exception as e:
            print(f"[warn] 加载指标数据失败：{e}")
    
    if RESUME and probs_file.exists():
        try:
            with open(probs_file, "r", encoding="utf-8") as f:
                prob_data = json.load(f)
                # 转换回原始格式
                for scen, rows in prob_data.items():
                    prob_collectors[scen] = []
                    for row in rows:
                        prob_collectors[scen].append({
                            "model": row["model"],
                            "y": np.array(row["y"]),
                            "p": np.array(row["p"])
                        })
            print(f"[info] 已加载现有概率数据：{len(prob_collectors)}个场景")
        except Exception as e:
            print(f"[warn] 加载概率数据失败：{e}")
    
    if RESUME and thresholds_file.exists():
        try:
            with open(thresholds_file, "r", encoding="utf-8") as f:
                thresholds_from_val = json.load(f)
            print(f"[info] 已加载现有阈值数据：{len(thresholds_from_val)}个模型")
        except Exception as e:
            print(f"[warn] 加载阈值数据失败：{e}")
    
    if RESUME and models_params_file.exists():
        try:
            with open(models_params_file, "r", encoding="utf-8") as f:
                model_params_record = json.load(f)
            print(f"[info] 已加载现有模型参数：{len(model_params_record)}个模型")
        except Exception as e:
            print(f"[warn] 加载模型参数失败：{e}")
            
    # 创建保存中间数据的函数
    def save_intermediate_data():
        # 保存指标数据
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics_rows, f, ensure_ascii=False, indent=2)
        
        # 保存概率数据（需要转换numpy数组）
        prob_data = {}
        for scen, rows in prob_collectors.items():
            prob_data[scen] = []
            for row in rows:
                prob_data[scen].append({
                    "model": row["model"],
                    "y": row["y"].tolist(),
                    "p": row["p"].tolist()
                })
        with open(probs_file, "w", encoding="utf-8") as f:
            json.dump(prob_data, f, ensure_ascii=False, indent=2)
        
        # 保存阈值数据
        with open(thresholds_file, "w", encoding="utf-8") as f:
            json.dump(thresholds_from_val, f, ensure_ascii=False, indent=2)
        
        # 保存模型参数
        with open(models_params_file, "w", encoding="utf-8") as f:
            json.dump(model_params_record, f, ensure_ascii=False, indent=2)
        
        print(f"[info] 已保存中间数据至 {data_dir}")

    def eval_and_collect(model_key: str, scen_name: str, X: pd.DataFrame, y: np.ndarray, pipe, thr: Optional[float]) -> None:
        p = _as_proba(pipe, X)
        if thr is None and scen_name == "internal_val":
            try:
                thr_use = _youden_threshold(y, p)
            except Exception:
                thr_use = 0.5
            thresholds_from_val[model_key] = float(thr_use)
            thr = thr_use
        met = compute_9_metrics(y, p, thr if thr is not None else 0.5, compute_ci=True)
        slope, intercept = calib_slope_intercept(y, p)
        row = {"model": model_key, "scenario": scen_name, **met,
               "Calib_Slope": slope, "Calib_Intercept": intercept,
               "Threshold_Used": float(thr if thr is not None else 0.5)}
        metrics_rows.append(row)
        prob_collectors[scen_name].append({"model": model_key, "y": y, "p": p})

    # 模型循环
    for model_key in MODEL_LIST:
        # 断点续跑：若已存在cv结果且开启RESUME，则跳过
        cv_csv = OUT_DIR / f"cv_results_{model_key}.csv"
        if RESUME and cv_csv.exists():
            print(f"[skip] {model_key}: detected existing {cv_csv}")
            continue

        print(f"[model] {model_key}")

        selected_feats = pick_features_for_model(model_key, all_feats)
        sel_num = [c for c in all_num if c in selected_feats]
        sel_bin = [c for c in all_bin if c in selected_feats]
        sel_mul = [c for c in all_mul if c in selected_feats]

        Xtr = align_columns(X_train_all, sel_num, sel_bin, sel_mul)
        Xva = align_columns(X_val_all,   sel_num, sel_bin, sel_mul)
        Xte = align_columns(X_test_all,  sel_num, sel_bin, sel_mul)
        Xext_main = align_columns(ext_main, sel_num, sel_bin, sel_mul) if ext_main is not None else None
        Xext_tran = align_columns(ext_tran, sel_num, sel_bin, sel_mul) if ext_tran is not None else None

        # 只有 LR 用 OHE，其余树模型用 Ordinal（更快更稳）
        use_ohe = (model_key == "lr")
        pre = build_preprocessor(all_num, all_bin, all_mul, selected=selected_feats, use_ohe=use_ohe)

        base_clf = make_model(model_key, y_train)
        if base_clf is None:
            print(f"  [skip] {model_key} 不可用"); continue

        pipe = Pipeline([("pre", pre), ("clf", base_clf)])

        # 网格搜索（Train）
        param_grid = build_param_grid(model_key, GRID_LEVEL)
        if param_grid:
            cv = StratifiedKFold(n_splits=CV_N_SPLITS, shuffle=True, random_state=42)
            gs = GridSearchCV(
                estimator=pipe,
                param_grid=param_grid,
                scoring="roc_auc",
                cv=cv,
                n_jobs=N_JOBS,         # 只开外层并行
                refit=True,
                verbose=1,
                error_score=np.nan
            )
            gs.fit(Xtr, y_train)
            best_pipe = gs.best_estimator_
            # 保存 CV 结果
            cv_df = pd.DataFrame(gs.cv_results_)
            cv_df.to_csv(cv_csv, index=False, encoding="utf-8-sig")
            print("  [save]", cv_csv)
            # 记录
            model_params_record[model_key] = {
                "features": selected_feats,
                "best_params": gs.best_params_,
                "best_cv_auc": float(gs.best_score_)
            }
        else:
            best_pipe = pipe.fit(Xtr, y_train)
            model_params_record[model_key] = {"features": selected_feats, "best_params": {}}

        # 保存“最佳未校准”管线
        joblib.dump(best_pipe, OUT_DIR / "models" / f"{model_key.upper()}_best_pipeline.joblib")

        # 验证集做 Isotonic 校准（若无 val 就跳过）
        if (Xva is not None) and (y_val is not None) and len(y_val) > 0:
            cal_pipe = make_isotonic_calibrated_pipe(best_pipe, Xva, y_val)
            joblib.dump(cal_pipe, OUT_DIR / "models" / f"{model_key.upper()}_calibrated_isotonic.joblib")
        else:
            cal_pipe = best_pipe

        # 阈值：优先用验证集 Youden
        thr_for_model = None
        if (Xva is not None) and (y_val is not None) and len(y_val) > 0:
            pv_val = _as_proba(cal_pipe, Xva)
            try:
                thr_for_model = _youden_threshold(y_val, pv_val)
            except Exception:
                thr_for_model = 0.5

        # 评估
        eval_and_collect(model_key, "internal_train", Xtr, y_train, cal_pipe, thr_for_model)
        if Xva is not None:
            eval_and_collect(model_key, "internal_val",   Xva, y_val,   cal_pipe, thr_for_model)
        if Xte is not None:
            eval_and_collect(model_key, "internal_test",  Xte, y_test,  cal_pipe, thr_for_model)
        if (Xext_main is not None) and (ext_main is not None) and (Y_NAME in ext_main.columns):
            eval_and_collect(model_key, "external_main",  Xext_main, ext_main[Y_NAME].astype(int).values, cal_pipe, thr_for_model)
        if (Xext_tran is not None) and (ext_tran is not None) and (Y_NAME in ext_tran.columns):
            eval_and_collect(model_key, "external_transfer", Xext_tran, ext_tran[Y_NAME].astype(int).values, cal_pipe, thr_for_model)
            
        # 每个模型训练完成后立即保存中间数据，确保断点续跑时数据完整
        save_intermediate_data()
        print(f"[info] 已保存 {model_key} 模型的中间数据")

    # 指标表
    df_metrics = pd.DataFrame(metrics_rows)
    out_metrics = OUT_DIR / "S7_final_metrics.csv"
    df_metrics.to_csv(out_metrics, index=False, encoding="utf-8-sig")
    print("[save]", out_metrics)
    
    # 创建包含置信区间的美观模型比较表
    try:
        for scenario in ["internal_train", "internal_val", "internal_test", "external_main", "external_transfer"]:
            # 筛选特定场景的指标
            df_scenario = df_metrics[df_metrics["scenario"] == scenario].copy()
            if len(df_scenario) > 0:
                # 按模型名排序
                df_scenario.sort_values(by=["model"], inplace=True)
                
                # 重塑数据以创建更美观的表格
                model_comparison = {}
                for _, row in df_scenario.iterrows():
                    model_name = f"{row['model']}_tuned"
                    for metric in ["AUROC", "AUPRC", "Accuracy", "Precision", "Recall", 
                                   "F1_Score", "Brier_Score", "Specificity", "NPV"]:
                        if metric not in model_comparison:
                            model_comparison[metric] = {}
                        model_comparison[metric][model_name] = row[metric]
                
                # 转换为DataFrame并保存
                comparison_df = pd.DataFrame(model_comparison)
                comparison_out = OUT_DIR / f"S7_{scenario}_model_comparison_with_CI.csv"
                comparison_df.to_csv(comparison_out, encoding="utf-8-sig")
                print(f"[save] 已创建包含置信区间的模型比较表: {comparison_out}")
    except Exception as e:
        print(f"[warn] 创建模型比较表失败: {e}")

    # 概率文件
    for scen, rows in prob_collectors.items():
        if not rows: continue
        y0 = rows[0]["y"]
        tab = {"y": y0}
        for r in rows:
            tab[f"p_{r['model']}"] = r["p"]
        pd.DataFrame(tab).to_csv(OUT_DIR / f"S7_probs_{scen}.csv", index=False, encoding="utf-8-sig")

    # 阈值（来自验证集）
    if thresholds_from_val:
        (OUT_DIR / "S7_thresholds_from_val.json").write_text(
            json.dumps(thresholds_from_val, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print("[save]", OUT_DIR / "S7_thresholds_from_val.json")

    # 确保使用所有已保存的数据进行绘图
    # 先尝试从文件加载最新数据，确保所有模型的数据都被包含
    try:
        # 如果有数据文件，尝试加载以确保包含所有模型的数据
        if probs_file.exists():
            with open(probs_file, "r", encoding="utf-8") as f:
                prob_data = json.load(f)
                # 合并数据，确保不丢失任何模型的结果
                for scen, rows in prob_data.items():
                    # 获取已有模型列表
                    existing_models = {r["model"] for r in prob_collectors.get(scen, [])}
                    # 添加缺失的模型数据
                    for row in rows:
                        if row["model"] not in existing_models:
                            prob_collectors.setdefault(scen, []).append({
                                "model": row["model"],
                                "y": np.array(row["y"]),
                                "p": np.array(row["p"])
                            })
            print("[info] 已合并文件中的概率数据用于绘图")
    except Exception as e:
        print(f"[warn] 加载概率数据用于绘图失败：{e}")

    # 绘图（每个场景一张，曲线里包含多个模型）
    for scen in ["internal_train","internal_val","internal_test","external_main","external_transfer"]:
        rows = prob_collectors.get(scen, [])
        if not rows: continue
        preds = { r["model"]: {scen: (r["y"], r["p"]) } for r in rows }
        plot_roc_multi(preds, OUT_DIR / "plots" / f"ROC_{scen}.png",
                       title=f"ROC — {scen.replace('_',' ').title()}")
        plot_calibration_multi(preds, OUT_DIR / "plots" / f"CAL_{scen}.png",
                               title=f"Calibration — {scen.replace('_',' ').title()}")
        plot_dca_multi(preds, OUT_DIR / "plots" / f"DCA_{scen}.png",
                       title=f"Decision Curve — {scen.replace('_',' ').title()}")

    # 模型卡
    mc = {
        "ver_in": str(VER_IN), "ver_out": str(VER_OUT), "y_name": Y_NAME,
        "grid_level": GRID_LEVEL, "cv_n_splits": CV_N_SPLITS,
        "n_jobs": N_JOBS, "use_gpu": USE_GPU, "resume": RESUME,
        "thresholds_from_val": thresholds_from_val,
        "models": model_params_record,
        "notes": "Single-layer parallelism (GridSearchCV only). Optional GPU for XGB/CatBoost/LGBM. "
                 "Isotonic calibration + Youden threshold on Validation (if available). "
                 "Evaluated on Train/Val/Test/External. Trees use OrdinalEncoder; LR uses OneHot."
    }
    (OUT_DIR / "S7_model_card.json").write_text(json.dumps(mc, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[save]", OUT_DIR / "S7_model_card.json")

    print("[done] S7 final evaluation finished.")


if __name__ == "__main__":
    main()
