# -*- coding: utf-8 -*-
r"""
10_feature_count_sweep.py — 固定插补=mean_mode，对所有模型做：
  1) 训练集内交叉验证的“真·RFE”曲线（Accuracy/Kappa vs Top-k）→ 仿 PDF Figure S4
  2) 按 Top-k 扫描（训练/验证/测试/外部），输出 AUROC/AUPRC 等 9 指标、最佳 k、最佳特征清单、折线图

输入（统一读 config.yaml）：
  - project_root, run_id_in, run_id_out
  - 02_processed_data/<VER_IN>/frozen/{charls_mean_mode_Xy.csv, klosa_transfer_Xy.csv}
  - 02_processed_data/<VER_IN>/splits/charls_{train,val,test}_idx.csv

输出（统一到 10_experiments/<VER_OUT>/feature_sweep）：
  - rfe_cv_curve_<model>.csv
  - rfe_curve_<model>.csv
  - S4_RFE_<model}_best_features.csv
  - S4_RFE_<model>_summary.csv
  - plots/<model>_{AUROC|AUPRC}_vs_k.[png|pdf]
  - plots/S4_RFE_trainCV_acc_kappa.[png|pdf]
  - S4_RFE_overview.csv
"""

# —— 必须在导入 matplotlib.pyplot 之前设置无界面后端，避免 Tk 相关报错 ——
import os
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import yaml
import hashlib

# ===================== 读取配置（统一口径） =====================
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

CFG = yaml.safe_load(open(repo_root() / "07_config" / "config.yaml", "r", encoding="utf-8"))

PROJECT_ROOT = Path(CFG["paths"]["project_root"])
VER_IN  = CFG.get("run_id_in",  CFG.get("run_id"))
VER_OUT = CFG.get("run_id_out", CFG.get("out_id", CFG.get("run_id")))
SEED    = int(CFG.get("seed", CFG.get("random_state", 42)))  # 统一随机种子

DATA_DIR   = PROJECT_ROOT / "02_processed_data" / VER_IN
FROZEN_DIR = DATA_DIR / "frozen"
SPLIT_DIR  = DATA_DIR / "splits"
OUT_DIR    = PROJECT_ROOT / "10_experiments" / VER_OUT / "feature_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "plots").mkdir(exist_ok=True)

# 固定插补：mean_mode 带 y 的全量集
MEAN_MODE_XY = FROZEN_DIR / "charls_mean_mode_Xy.csv"
METHOD_NAME  = "mean_mode"

# 外部集（默认 KLOSA transfer；也支持在 YAML paths.external_main_Xy / external_transfer_Xy 覆写）
def _resolve_external(p_like: Optional[str|Path], default: Path) -> Path:
    if not p_like:
        return default
    p = Path(p_like)
    return p if p.is_absolute() else (PROJECT_ROOT / p)

DEFAULT_EXT = FROZEN_DIR / "klosa_transfer_Xy.csv"
ext_paths_cfg = CFG.get("paths", {})
EXT_MAIN_XY = _resolve_external(ext_paths_cfg.get("external_main_Xy", None), DEFAULT_EXT)
EXT_TRAN_XY = _resolve_external(ext_paths_cfg.get("external_transfer_Xy", None), DEFAULT_EXT)

# YAML（用于变量类型）
YAML_PATH = repo_root() / "07_config" / "config.yaml"

# 标签/ID列统一从配置读取
Y_NAME  = (CFG.get("outcome", {}) or {}).get("name", "depression_bin")
ID_COLS = list(CFG.get("id_cols", CFG.get("ids", ["ID", "householdID"])))

# 模型列表：支持 config.eval.models 覆写
_models_cfg = (CFG.get("eval", {}) or {}).get(
    "models",
    ["lr", "rf", "extra_trees", "gb", "adaboost", "lgb", "xgb"]
)
MODEL_LIST = [m for m in _models_cfg if str(m).lower() != "catboost"]

# k 扫描列表（会自动截到 ≤ 特征总数，且包含“全部特征”）
K_LIST = [5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70]

# —— Bootstrap 置信区间设置（只对 AUROC/AUPRC；不重训模型，只对 (y,p) 重采样）——
ENABLE_BOOTSTRAP = True
N_BOOT = 200
CI_ALPHA = 0.05
BOOT_SEED = 1234

# —— 真·RFE（训练集交叉验证，仿 S4）——
USE_TRUE_RFE      = True
RFE_STEP          = 1
RFE_MIN_FEATURES  = 5
CV_FOLDS          = 5


# ---------- sklearn 预处理/模型 ----------
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, precision_score,
    recall_score, f1_score, brier_score_loss, cohen_kappa_score
)
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator

def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # sklearn >=1.4
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)         # sklearn <1.4

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
)

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


def _class_ratio(y: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    pos = y.sum(); neg = len(y) - pos
    return (neg / max(pos, 1))


def make_model(model_key: str, y_train: np.ndarray):
    mk = model_key.lower()
    if mk == "lr":
        return LogisticRegression(solver="liblinear", penalty="l2",
                                  max_iter=1000, class_weight="balanced", random_state=SEED)
    elif mk == "rf":
        return RandomForestClassifier(n_estimators=600, min_samples_leaf=5,
                                     class_weight="balanced", n_jobs=-1, random_state=SEED)
    elif mk == "extra_trees":
        return ExtraTreesClassifier(n_estimators=700, min_samples_leaf=3,
                                    class_weight="balanced", n_jobs=-1, random_state=SEED)
    elif mk == "gb":
        return GradientBoostingClassifier(random_state=SEED)
    elif mk == "adaboost":
        return AdaBoostClassifier(n_estimators=600, learning_rate=0.05,
                                  algorithm="SAMME", random_state=SEED)
    elif mk == "lgb":
        if _HAS_LGBM:
            return LGBMClassifier(n_estimators=700, learning_rate=0.05, max_depth=-1,
                                  subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
                                  objective="binary", random_state=SEED, class_weight="balanced")
        warnings.warn("[WARN] lightgbm 未安装，回退为 GB")
        return GradientBoostingClassifier(random_state=SEED)
    elif mk == "xgb":
        if _HAS_XGB:
            return XGBClassifier(n_estimators=800, learning_rate=0.05, max_depth=5,
                                 subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
                                 objective="binary:logistic", random_state=SEED, n_jobs=-1,
                                 scale_pos_weight=_class_ratio(y_train))
        warnings.warn("[WARN] xgboost 未安装，回退为 GB")
        return GradientBoostingClassifier(random_state=SEED)
    elif mk == "catboost":
        if _HAS_CAT:
            w1 = _class_ratio(y_train)
            return CatBoostClassifier(iterations=800, learning_rate=0.05, depth=6,
                                      loss_function="Logloss", eval_metric="AUC",
                                      class_weights=[1.0, w1], random_seed=SEED, verbose=False)
        warnings.warn("[WARN] catboost 未安装，跳过")
        return None
    else:
        raise ValueError(f"Unsupported model: {model_key}")


# ---------- 数据加载 ----------
def _need(p: Path, hint: str):
    if not p.exists():
        raise FileNotFoundError(f"缺少必要文件：{p}\n请先运行：{hint}")

def load_idx_csv(name: str) -> Optional[np.ndarray]:
    p = SPLIT_DIR / f"charls_{name}_idx.csv"
    if not p.exists(): return None
    s = pd.read_csv(p).iloc[:, 0]
    return s.values.astype(int)

def safe_read_csv(p: Optional[Path]) -> Optional[pd.DataFrame]:
    if p and Path(p).exists():
        return pd.read_csv(p)
    return None


# ---------- YAML（可选） ----------
def read_yaml_variable_types(yaml_path: Path) -> Tuple[List[str], List[str]]:
    """从 YAML 读取 variable_types 的 continuous/categorical。"""
    y = yaml.safe_load(open(yaml_path, "r", encoding="utf-8"))
    num_cols = y.get("preprocess", {}).get("variable_types", {}).get("continuous", []) or []
    cat_cols = y.get("preprocess", {}).get("variable_types", {}).get("categorical", []) or []
    return list(num_cols), list(cat_cols)

def infer_feature_types(df: pd.DataFrame, y_name: str,
                        init_num: Optional[List[str]]=None,
                        init_mul: Optional[List[str]]=None,
                        id_cols: Optional[List[str]]=None) -> Tuple[List[str], List[str], List[str]]:
    """自动推断 + YAML 种子（init_num/init_mul），把明确的 continuous/categorical 作为起点，
       然后实际检查 0/1 列归 bin。"""
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
            (bin_cols if nunq == 2 else mul_cols).append(c)
            continue

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


# ---------- 预处理器（支持子集） ----------
def build_preprocessor(all_num: List[str], all_bin: List[str], all_mul: List[str],
                       selected: Optional[List[str]] = None) -> ColumnTransformer:
    if selected is not None:
        s = set(selected)
        num_cols = [c for c in all_num if c in s]
        bin_cols = [c for c in all_bin if c in s]
        mul_cols = [c for c in all_mul if c in s]
    else:
        num_cols, bin_cols, mul_cols = all_num, all_bin, all_mul

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=False), num_cols),
            ("bin", "passthrough", bin_cols),
            ("mul", make_ohe(), mul_cols),
        ],
        remainder="drop"
    )


# ---------- 概率/指标 + Bootstrap CI ----------
def _as_proba(clf: BaseEstimator, X: pd.DataFrame) -> np.ndarray:
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    if hasattr(clf, "decision_function"):
        z = clf.decision_function(X)
        return 1.0 / (1.0 + np.exp(-z))
    return clf.predict(X).astype(float)

def compute_9_metrics(y_true: np.ndarray, p: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p).clip(1e-8, 1-1e-8)
    yhat = (p >= thr).astype(int)
    tn = np.sum((y == 0) & (yhat == 0))
    fp = np.sum((y == 0) & (yhat == 1))
    fn = np.sum((y == 1) & (yhat == 0))
    tp = np.sum((y == 1) & (yhat == 1))

    def _safe(a, b):
        b = max(float(b), 1e-12)
        return float(a) / b

    # 更稳的 AUC 计算（单一类别时返回 NaN）
    try:
        auroc = roc_auc_score(y, p)
    except Exception:
        auroc = float("nan")
    try:
        auprc = average_precision_score(y, p)
    except Exception:
        auprc = float("nan")

    return {
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

def _seed_for(*parts) -> int:
    """用 md5 生成确定性的 bootstrap 种子（避免内置 hash 的不稳定）"""
    s = "|".join(map(str, parts))
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)

def _bootstrap_ci(y: np.ndarray, p: np.ndarray, which: str,
                  n_boot: int = N_BOOT, alpha: float = CI_ALPHA, seed: int = BOOT_SEED) -> Tuple[float, float]:
    if not ENABLE_BOOTSTRAP:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]; pb = p[idx]
        try:
            if which == "AUROC":
                v = roc_auc_score(yb, pb)
            else:
                v = average_precision_score(yb, pb)
        except Exception:
            v = np.nan
        vals.append(v)
    vals = np.array(vals, dtype=float)
    lo = np.nanpercentile(vals, 100*alpha/2)
    hi = np.nanpercentile(vals, 100*(1 - alpha/2))
    return float(lo), float(hi)


# ---------- 从 pipeline 聚合回“原始特征”的重要度 ----------
def get_feature_importance_per_orig_feature(pipe: Pipeline,
                                            num_cols: List[str], bin_cols: List[str], mul_cols: List[str]) -> Dict[str, float]:
    pre: ColumnTransformer = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]

    start = 0
    mapping: Dict[str, List[int]] = {}

    for c in num_cols:
        mapping[c] = [start]; start += 1
    for c in bin_cols:
        mapping[c] = [start]; start += 1

    # mul 的 one-hot 展开（更安全的取法）
    if len(mul_cols) > 0 and "mul" in pre.named_transformers_:
        ohe = pre.named_transformers_["mul"]
        if hasattr(ohe, "categories_"):
            for c, cats in zip(mul_cols, ohe.categories_):
                idxs = list(range(start, start + len(cats)))
                mapping[c] = idxs
                start += len(cats)

    # 分配重要度
    if hasattr(clf, "coef_"):
        imp = np.abs(np.ravel(clf.coef_))
    elif hasattr(clf, "feature_importances_"):
        imp = np.asarray(clf.feature_importances_, dtype=float)
    else:
        imp = np.zeros(start, dtype=float)

    agg: Dict[str, float] = {}
    for c, idxs in mapping.items():
        if len(idxs) == 0:
            agg[c] = 0.0
        elif len(idxs) == 1:
            agg[c] = float(imp[idxs[0]])
        else:
            agg[c] = float(np.sum(imp[idxs]))
    return agg


# ---------- 真·RFE（训练内 CV），仿 S4 ----------
def rfe_rank_by_cv(df, all_num, all_bin, all_mul, feats_all, y_name, model_key, tr_idx):
    """
    每一轮：
      1) 用当前特征做 Stratified K-fold 交叉验证，记录 Accuracy_CV / Kappa_CV / AUROC_CV
      2) 拟合一次，按“原始特征聚合后的重要度”删掉 RFE_STEP 个最弱特征
    返回：
      ranking: 由强到弱的原始特征排名
      trace:   每一步的 CV 指标与剩余特征数 k
    """
    y_tr = df.loc[tr_idx, y_name].astype(int).values
    remaining = feats_all[:]
    eliminated = []
    trace_rows = []

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

    def _fit_and_prob(sel, tr_ids, va_ids):
        pre = build_preprocessor(all_num, all_bin, all_mul, selected=sel)
        clf = make_model(model_key, df.loc[tr_ids, y_name].values)
        if clf is None:
            return None
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        pipe.fit(df.loc[tr_ids, sel], df.loc[tr_ids, y_name].astype(int).values)
        p = _as_proba(pipe, df.loc[va_ids, sel])
        return p

    while len(remaining) > max(RFE_MIN_FEATURES, RFE_STEP):
        # 1) CV 指标
        accs, kappas, aucs = [], [], []
        for tr_fold, va_fold in skf.split(tr_idx, y_tr):
            tr_ids = tr_idx[tr_fold]; va_ids = tr_idx[va_fold]
            p = _fit_and_prob(remaining, tr_ids, va_ids)
            if p is None:
                break
            yv = df.loc[va_ids, y_name].astype(int).values
            yh = (p >= 0.5).astype(int)
            accs.append(accuracy_score(yv, yh))
            kappas.append(cohen_kappa_score(yv, yh))
            try:
                aucs.append(roc_auc_score(yv, p))
            except Exception:
                aucs.append(np.nan)
        if len(accs) == 0:
            break
        trace_rows.append({
            "k": len(remaining),
            "Accuracy_CV": float(np.mean(accs)),
            "Kappa_CV": float(np.mean(kappas)),
            "AUROC_CV": float(np.nanmean(aucs)),
        })

        # 2) 删除若干最弱原始特征
        pre_full = build_preprocessor(all_num, all_bin, all_mul, selected=remaining)
        clf_full = make_model(model_key, df.loc[tr_idx, y_name].values)
        if clf_full is None:
            break
        pipe_full = Pipeline([("pre", pre_full), ("clf", clf_full)])
        pipe_full.fit(df.loc[tr_idx, remaining], y_tr)
        imp_map = get_feature_importance_per_orig_feature(
            pipe_full,
            [c for c in all_num if c in remaining],
            [c for c in all_bin if c in remaining],
            [c for c in all_mul if c in remaining],
        )
        to_drop = [f for f, _ in sorted(imp_map.items(), key=lambda x: x[1])[:RFE_STEP]]
        eliminated.extend(to_drop)
        remaining = [f for f in remaining if f not in set(to_drop)]

    # 最终把剩余做一次强度排序
    if len(remaining):
        pre_full = build_preprocessor(all_num, all_bin, all_mul, selected=remaining)
        clf_full = make_model(model_key, df.loc[tr_idx, y_name].values)
        pipe_full = Pipeline([("pre", pre_full), ("clf", clf_full)])
        pipe_full.fit(df.loc[tr_idx, remaining], y_tr)
        imp_map = get_feature_importance_per_orig_feature(
            pipe_full,
            [c for c in all_num if c in remaining],
            [c for c in all_bin if c in remaining],
            [c for c in all_mul if c in remaining],
        )
        still = [f for f, _ in sorted(imp_map.items(), key=lambda x: x[1], reverse=True)]
    else:
        still = []
    ranking = still + list(reversed(eliminated))
    trace = pd.DataFrame(trace_rows).sort_values("k")
    return ranking, trace


# ---------- S4 大面板（训练 CV：Accuracy & Kappa vs k） ----------
def plot_S4_panel(models: List[str], out_dir: Path):
    import matplotlib.pyplot as plt
    n = len(models)
    nrows, ncols = n, 2
    fig = plt.figure(figsize=(10, 2.2*n))
    for i, m in enumerate(models):
        p = out_dir / f"rfe_cv_curve_{m}.csv"
        if not p.exists():
            continue
        dfcv = pd.read_csv(p)
        if "k" not in dfcv.columns:
            continue
        xs = dfcv["k"].values
        acc = dfcv["Accuracy_CV"].values
        kap = dfcv["Kappa_CV"].values
        k_best = int(xs[np.nanargmax(acc)])

        ax1 = fig.add_subplot(nrows, ncols, 2*i+1)
        ax1.plot(xs, acc, marker="o", linewidth=1.2)
        ax1.axvline(k_best, linestyle="--", linewidth=1)
        ax1.set_title(m.upper()); ax1.set_ylabel("Accuracy (CV)")
        ax1.set_xlabel("Variables (Top-k)"); ax1.grid(alpha=0.3, linestyle="--")

        ax2 = fig.add_subplot(nrows, ncols, 2*i+2)
        ax2.plot(xs, kap, marker="o", linewidth=1.2)
        ax2.axvline(k_best, linestyle="--", linewidth=1)
        ax2.set_title(m.upper()); ax2.set_ylabel("Kappa (CV)")
        ax2.set_xlabel("Variables (Top-k)"); ax2.grid(alpha=0.3, linestyle="--")

    fig.suptitle("Recursive feature elimination (train CV): Accuracy & Kappa vs Variables", y=0.995, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    (out_dir/"plots").mkdir(exist_ok=True, parents=True)
    fig.savefig(out_dir/"plots/S4_RFE_trainCV_acc_kappa.png", dpi=220)
    fig.savefig(out_dir/"plots/S4_RFE_trainCV_acc_kappa.pdf")
    plt.close(fig)


# ===================== 主流程 =====================
def main():
    # 必要文件检查
    _need(MEAN_MODE_XY, "08_freeze_mean_mode.py")
    _need(SPLIT_DIR / "charls_train_idx.csv", "03_split_charls.py 生成 splits")
    # 可选：val/test 若缺会自动跳过相应场景

    # 加载数据
    df = pd.read_csv(MEAN_MODE_XY)
    if Y_NAME not in df.columns:
        raise KeyError(f"'{Y_NAME}' 不在 {MEAN_MODE_XY}")

    tr_idx = load_idx_csv("train")
    va_idx = load_idx_csv("val")
    te_idx = load_idx_csv("test")

    ext_main = safe_read_csv(EXT_MAIN_XY)
    ext_tran = safe_read_csv(EXT_TRAN_XY)

    for ext, tag in [(ext_main, "external_main"), (ext_tran, "external_transfer")]:
        if ext is not None and Y_NAME not in ext.columns:
            raise KeyError(f"[{tag}] 缺少标签列 {Y_NAME}")

    # 变量类型（优先 YAML）
    try:
        if YAML_PATH.exists():
            init_num, init_mul = read_yaml_variable_types(YAML_PATH)
        else:
            init_num, init_mul = None, None
    except Exception:
        init_num, init_mul = None, None

    all_num, all_bin, all_mul = infer_feature_types(df, Y_NAME, init_num, init_mul, ID_COLS)
    all_feats = all_num + all_bin + all_mul
    if len(all_feats) == 0:
        raise RuntimeError("未识别到可用特征列，请检查 frozen/charls_mean_mode_Xy.csv 是否只含 y 列。")

    max_k = len(all_feats)
    k_list = sorted({k for k in K_LIST if 0 < k < max_k})
    if max_k not in k_list:
        k_list.append(max_k)

    summaries = []

    for model_key in MODEL_LIST:
        print(f"[model] {model_key}")

        # === 先做真·RFE（训练 CV），用于 S4 的曲线 & 最终的特征排序 ===
        if USE_TRUE_RFE:
            ranked_feats, rfe_cv = rfe_rank_by_cv(df, all_num, all_bin, all_mul,
                                                  all_feats, Y_NAME, model_key, tr_idx)
            rfe_cv.to_csv(OUT_DIR / f"rfe_cv_curve_{model_key}.csv", index=False, encoding="utf-8-sig")
        else:
            pre_full = build_preprocessor(all_num, all_bin, all_mul, selected=None)
            clf = make_model(model_key, df[Y_NAME].values[tr_idx])
            if clf is None:
                print(f"  [skip] {model_key} 不可用")
                continue
            pipe = Pipeline([("pre", pre_full), ("clf", clf)])
            pipe.fit(df.loc[tr_idx, all_feats], df.loc[tr_idx, Y_NAME].astype(int).values)
            imp_map = get_feature_importance_per_orig_feature(pipe, all_num, all_bin, all_mul)
            ranked_feats = [f for f, _ in sorted(imp_map.items(), key=lambda x: x[1], reverse=True)]

        # === 按 k 扫描（用于 AUROC/AUPRC 选择最佳 k） ===
        rows = []
        ytr = df.loc[tr_idx, Y_NAME].astype(int).values
        for k in k_list:
            sel = ranked_feats[:k]
            pre_k = build_preprocessor(all_num, all_bin, all_mul, selected=sel)
            clf_k = make_model(model_key, df[Y_NAME].values[tr_idx])
            if clf_k is None:
                continue

            pipe_k = Pipeline([("pre", pre_k), ("clf", clf_k)])
            pipe_k.fit(df.loc[tr_idx, sel], ytr)

            def _eval_on(dfX: pd.DataFrame, split_name: str):
                y_true = dfX[Y_NAME].astype(int).values
                p = _as_proba(pipe_k, dfX[sel])
                met = compute_9_metrics(y_true, p)
                # 确定性的 bootstrap 种子
                au_seed = _seed_for(BOOT_SEED, model_key, split_name, "AUROC", k)
                ap_seed = _seed_for(BOOT_SEED, model_key, split_name, "AUPRC", k)
                au_lo, au_hi = _bootstrap_ci(y_true, p, "AUROC", seed=au_seed)
                ap_lo, ap_hi = _bootstrap_ci(y_true, p, "AUPRC", seed=ap_seed)
                met["AUROC_lo"], met["AUROC_hi"] = au_lo, au_hi
                met["AUPRC_lo"], met["AUPRC_hi"] = ap_lo, ap_hi
                return met

            rows.append({"k": k, "scenario": "internal_train", **_eval_on(df.loc[tr_idx], "train")})
            if va_idx is not None and len(va_idx) > 0:
                rows.append({"k": k, "scenario": "internal_val", **_eval_on(df.loc[va_idx], "val")})
            if te_idx is not None and len(te_idx) > 0:
                rows.append({"k": k, "scenario": "internal_test", **_eval_on(df.loc[te_idx], "test")})
            if ext_main is not None:
                rows.append({"k": k, "scenario": "external_main", **_eval_on(ext_main, "ext_main")})
            if ext_tran is not None:
                rows.append({"k": k, "scenario": "external_transfer", **_eval_on(ext_tran, "ext_tran")})

        curve = pd.DataFrame(rows)
        curve.insert(0, "model", model_key)
        curve.insert(0, "method", METHOD_NAME)
        curve_csv = OUT_DIR / f"rfe_curve_{model_key}.csv"
        curve.to_csv(curve_csv, index=False, encoding="utf-8-sig")
        print(f"  [save] {curve_csv}")

        # === 选最佳 k（优先 internal_val 的 AUROC，AUPRC 打平，再以更小 k 打平；若无 val 用 test） ===
        if "internal_val" in curve["scenario"].unique():
            sub = curve[curve["scenario"] == "internal_val"].copy()
        else:
            sub = curve[curve["scenario"] == "internal_test"].copy()
        if len(sub) == 0:
            print(f"  [warn] {model_key} 无法选 k（缺少 val/test）")
            continue

        sub = sub.sort_values(["AUROC", "AUPRC", "k"], ascending=[False, False, True])
        best_k = int(sub.iloc[0]["k"])
        best_feats = ranked_feats[:best_k]

        pd.DataFrame({"feature": best_feats}).to_csv(
            OUT_DIR / f"S4_RFE_{model_key}_best_features.csv", index=False, encoding="utf-8-sig"
        )
        best_rows = curve[curve["k"] == best_k].copy()
        best_rows.to_csv(OUT_DIR / f"S4_RFE_{model_key}_summary.csv", index=False, encoding="utf-8-sig")
        print(f"  [save] best k={best_k} tables")

        rec = {"model": model_key, "k_best": best_k}
        for scen in ["internal_train","internal_val","internal_test","external_main","external_transfer"]:
            row = best_rows[best_rows["scenario"]==scen]
            if len(row):
                rec[f"{scen}_AUROC"] = float(row["AUROC"].values[0])
                rec[f"{scen}_AUPRC"] = float(row["AUPRC"].values[0])
        summaries.append(rec)

        # === AUROC/AUPRC vs k 曲线（含 CI 阴影 & 竖线 best_k） ===
        import matplotlib.pyplot as plt
        for metric in ["AUROC", "AUPRC"]:
            fig = plt.figure(figsize=(8,4.8)); ax = fig.add_subplot(111)
            for scen in ["internal_train","internal_val","internal_test","external_main","external_transfer"]:
                ss = curve[curve["scenario"]==scen]
                if len(ss)==0: continue
                xs = ss["k"].values
                ys = ss[metric].values
                ax.plot(xs, ys, marker="o", label=scen)
                lo = ss.get(f"{metric}_lo", pd.Series([np.nan]*len(ss))).values
                hi = ss.get(f"{metric}_hi", pd.Series([np.nan]*len(ss))).values
                if np.isfinite(lo).any() and np.isfinite(hi).any():
                    ax.fill_between(xs, lo, hi, alpha=0.15)
            ax.axvline(best_k, linestyle="--", linewidth=1.5)
            ax.set_title(f"{model_key} — {metric} vs k")
            ax.set_xlabel("Top-k features")
            ax.set_ylabel(metric)
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.legend(loc="best")
            png = OUT_DIR / "plots" / f"{model_key}_{metric}_vs_k.png"
            pdf = OUT_DIR / "plots" / f"{model_key}_{metric}_vs_k.pdf"
            fig.tight_layout(); fig.savefig(png, dpi=200); fig.savefig(pdf); plt.close(fig)

    # S4 大面板
    if USE_TRUE_RFE:
        plot_S4_panel(MODEL_LIST, OUT_DIR)

    # 概览表
    if len(summaries):
        # 如果没有 val，就按 test 的列排序，避免 KeyError
        sort_cols = ["internal_val_AUROC","internal_val_AUPRC","k_best"]
        if not all(c in pd.DataFrame(summaries).columns for c in sort_cols[:2]):
            sort_cols = ["internal_test_AUROC","internal_test_AUPRC","k_best"]
        df_sum = pd.DataFrame(summaries).sort_values(sort_cols, ascending=[False, False, True])
        df_sum.to_csv(OUT_DIR / "S4_RFE_overview.csv", index=False, encoding="utf-8-sig")
        print(f"[save] overview -> {OUT_DIR / 'S4_RFE_overview.csv'}")

    print("[done] feature sweep finished.")


if __name__ == "__main__":
    main()
