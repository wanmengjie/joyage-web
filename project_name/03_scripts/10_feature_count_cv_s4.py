# -*- coding: utf-8 -*-
r"""
10_feature_count_cv_s4.py — 仅用训练集做交叉验证的“RFE 式特征数量选择”，生成 Figure S4 风格大图
- 固定插补：mean_mode（frozen/charls_mean_mode_Xy.csv）
- 模型：lr, rf, extra_trees, gb, adaboost, lgb, xgb, catboost（缺库自动跳过）
- 对每个 k：在 TRAIN 内做 StratifiedKFold 交叉验证，聚合 OOF 预测，计算 Accuracy/Kappa 等
- 画图：每个模型两列小图（Accuracy、Kappa），曲线 + 各自最佳 k 的三角标记 + 竖线
- 同时保存每个模型的 k-曲线 CSV

输出目录：10_experiments/<VER_OUT>/feature_sweep_cv_s4/
"""

# --- 固定无界面后端，避免 Tk 报错 ---
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

# ===================== 配置区（改为读 YAML，兼容 run_id/out_id） =====================
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

import yaml
CFG = yaml.safe_load(open(repo_root() / "07_config" / "config.yaml", "r", encoding="utf-8"))

PROJECT_ROOT = Path(CFG["paths"]["project_root"])

# 统一获取输入/输出版本，兼容 run_id / out_id / run_id_in / run_id_out
VER_IN  = CFG.get("run_id_in",  CFG.get("run_id"))
VER_OUT = CFG.get("run_id_out", CFG.get("out_id", CFG.get("run_id")))

# 统一随机种子：优先 seed，回落 random_state
SEED = int(CFG.get("seed", CFG.get("random_state", 42)))

# 固定插补：mean_mode（带 y）
MEAN_MODE_XY = PROJECT_ROOT / "02_processed_data" / VER_IN / "frozen" / "charls_mean_mode_Xy.csv"
METHOD_NAME  = "mean_mode"

# 只用训练划分
TRAIN_IDX = PROJECT_ROOT / "02_processed_data" / VER_IN / "splits" / "charls_train_idx.csv"

# 标签列
Y_NAME = (CFG.get("outcome", {}) or {}).get("name", "depression_bin")

# 模型列表（缺库自动跳过）；若 config.eval.models 存在则优先使用
_models_cfg = (CFG.get("eval", {}) or {}).get(
    "models",
    ["lr", "rf", "extra_trees", "gb", "adaboost", "lgb", "xgb"]
)
MODEL_LIST = [m for m in _models_cfg if str(m).lower() != "catboost"]

# k 扫描（脚本会自动限制到 ≤ 总特征数，并加入“全部特征”）
K_LIST = [5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70]

# 交叉验证
N_SPLITS = 5
CV_SHUFFLE = True
CV_SEED = SEED

# 输出目录
OUT_DIR = PROJECT_ROOT / "10_experiments" / VER_OUT / "feature_sweep_cv_s4"
MAKE_PLOTS = True
# ==================================================


# ---------- sklearn 预处理/模型 ----------
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, precision_score,
    recall_score, f1_score, brier_score_loss, cohen_kappa_score
)
from sklearn.model_selection import StratifiedKFold

def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # sklearn >=1.4
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)         # sklearn <1.4

from sklearn.linear_model import LogisticRegression
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


# ---------- 工具函数 ----------
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

def load_idx_csv(path: Path) -> np.ndarray:
    s = pd.read_csv(path).iloc[:, 0]
    return s.values.astype(int)

def infer_feature_types(df: pd.DataFrame, y_name: str,
                        seed_num: Optional[List[str]]=None, seed_cat: Optional[List[str]]=None,
                        id_cols: Optional[List[str]]=None) -> Tuple[List[str], List[str], List[str]]:
    if id_cols is None: id_cols = []
    exclude = set(id_cols + [y_name])
    feats = [c for c in df.columns if c not in exclude]
    seed_num = set(seed_num or [])
    seed_cat = set(seed_cat or [])
    num_cols, bin_cols, mul_cols = [], [], []
    for c in feats:
        s = df[c]
        if c in seed_num:
            num_cols.append(c); continue
        if c in seed_cat:
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

def ensure_integer_labels_for_cats(df: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, dict]:
    encoders = {}
    out = df.copy()
    for c in cols:
        if c in out.columns and not pd.api.types.is_integer_dtype(out[c]):
            le = LabelEncoder()
            out[c] = le.fit_transform(out[c].astype("category").astype(str))
            encoders[c] = le
    return out, encoders

def build_preprocessor(num_cols: List[str], bin_cols: List[str], mul_cols: List[str],
                       selected: Optional[List[str]]=None) -> ColumnTransformer:
    if selected is not None:
        s = set(selected)
        n = [c for c in num_cols if c in s]
        b = [c for c in bin_cols if c in s]
        m = [c for c in mul_cols if c in s]
    else:
        n, b, m = num_cols, bin_cols, mul_cols
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=False), n),
            ("bin", "passthrough", b),
            ("mul", make_ohe(), m),
        ],
        remainder="drop"
    )

def _as_proba(clf, X):
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    if hasattr(clf, "decision_function"):
        z = clf.decision_function(X)
        return 1 / (1 + np.exp(-z))
    return clf.predict(X).astype(float)

def compute_metrics(y_true: np.ndarray, p: np.ndarray, thr: float=0.5) -> Dict[str, float]:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p).clip(1e-8, 1-1e-8)
    yhat = (p >= thr).astype(int)
    tn = np.sum((y == 0) & (yhat == 0))
    fp = np.sum((y == 0) & (yhat == 1))
    fn = np.sum((y == 1) & (yhat == 0))
    tp = np.sum((y == 1) & (yhat == 1))
    def _safe(a, b): b = max(float(b), 1e-12); return float(a)/b
    return {
        "Accuracy": accuracy_score(y, yhat),
        "Kappa": cohen_kappa_score(y, yhat),
        "AUROC": roc_auc_score(y, p),
        "AUPRC": average_precision_score(y, p),
        "Precision": precision_score(y, yhat, zero_division=0),
        "Recall": recall_score(y, yhat, zero_division=0),
        "F1_Score": f1_score(y, yhat, zero_division=0),
        "Brier_Score": brier_score_loss(y, p),
        "Specificity": _safe(tn, tn + fp),
        "NPV": _safe(tn, tn + fn),
    }

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
    # mul 的 one-hot 展开
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
    agg = {}
    for c, idxs in mapping.items():
        agg[c] = float(np.sum(imp[idxs])) if len(idxs) > 1 else float(imp[idxs[0]])
    return agg
# --------------------------------------------------


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "plots").mkdir(exist_ok=True)

    if not MEAN_MODE_XY.exists():
        raise FileNotFoundError(
            f"未找到 {MEAN_MODE_XY}\n"
            f"请先运行 08_freeze_mean_mode.py 生成 frozen/charls_mean_mode_Xy.csv。"
        )
    if not TRAIN_IDX.exists():
        raise FileNotFoundError(
            f"未找到训练索引 {TRAIN_IDX}\n"
            f"请先运行 03_split_charls.py 生成 splits。"
        )

    # 数据与索引
    df = pd.read_csv(MEAN_MODE_XY)
    if Y_NAME not in df.columns:
        raise KeyError(f"'{Y_NAME}' 不在 {MEAN_MODE_XY}")
    tr_idx = load_idx_csv(TRAIN_IDX)

    # YAML 变量类型（可选）
    seed_num = (CFG.get("preprocess", {}) or {}).get("variable_types", {}).get("continuous", []) or []
    seed_cat = (CFG.get("preprocess", {}) or {}).get("variable_types", {}).get("categorical", []) or []

    # id 列从 YAML 读取（兼容 id_cols / ids）
    id_cols = list(CFG.get("id_cols", CFG.get("ids", ["ID", "householdID"])))

    all_num, all_bin, all_mul = infer_feature_types(df, Y_NAME, seed_num, seed_cat, id_cols)
    all_feats = all_num + all_bin + all_mul
    if len(all_feats) == 0:
        raise RuntimeError("未识别到可用特征列，请检查 frozen/charls_mean_mode_Xy.csv 是否只含 y 列。")

    # k 列表修正
    max_k = len(all_feats)
    k_list = sorted({k for k in K_LIST if 0 < k < max_k})
    if max_k not in k_list:
        k_list.append(max_k)

    # 仅训练集
    X_train_full = df.loc[tr_idx, all_feats].copy()
    y_train_full = df.loc[tr_idx, Y_NAME].astype(int).values
    X_train_full, _ = ensure_integer_labels_for_cats(X_train_full, all_mul)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=CV_SHUFFLE, random_state=CV_SEED)

    # 收集所有模型曲线用于大图
    curves_by_model: Dict[str, pd.DataFrame] = {}

    for model_key in MODEL_LIST:
        print(f"[model] {model_key}")

        # 全特征训练以取排序
        pre_full = build_preprocessor(all_num, all_bin, all_mul, selected=None)
        clf = make_model(model_key, y_train_full)
        if clf is None:
            print(f"  [skip] {model_key} 不可用")
            continue

        pipe_full = Pipeline([("pre", pre_full), ("clf", clf)])
        pipe_full.fit(X_train_full, y_train_full)
        imp_map = get_feature_importance_per_orig_feature(pipe_full, all_num, all_bin, all_mul)
        ranked = sorted(imp_map.items(), key=lambda x: x[1], reverse=True)
        ranked_feats = [f for f, _ in ranked if f in all_feats]
        if not ranked_feats:  # 极端兜底
            var_series = X_train_full.var().sort_values(ascending=False)
            ranked_feats = var_series.index.tolist()

        rows = []
        for k in k_list:
            sel = ranked_feats[:k]
            oof_idx_all, oof_p_all = [], []

            for fold, (tr, va) in enumerate(skf.split(X_train_full, y_train_full), start=1):
                pre_k = build_preprocessor(all_num, all_bin, all_mul, selected=sel)
                clf_k = make_model(model_key, y_train_full[tr])
                if clf_k is None:
                    continue
                pipe_k = Pipeline([("pre", pre_k), ("clf", clf_k)])
                pipe_k.fit(X_train_full.iloc[tr], y_train_full[tr])

                p_va = _as_proba(pipe_k, X_train_full.iloc[va])
                oof_idx_all.append(va)
                oof_p_all.append(p_va)

            oof_idx = np.concatenate(oof_idx_all, axis=0)
            oof_p = np.concatenate(oof_p_all, axis=0)
            y_oof = y_train_full[oof_idx]

            met = compute_metrics(y_oof, oof_p)
            rows.append({"method": METHOD_NAME, "model": model_key, "scenario": "internal_train_cv",
                         "k": k, **met})

        curve = pd.DataFrame(rows).sort_values("k")
        curves_by_model[model_key] = curve

        csv_path = OUT_DIR / f"rfe_curve_{model_key}_cv.csv"
        curve.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"  [save] {csv_path}")

    # ===== 画“Figure S4 风格”的大图：每个模型两列（Accuracy、Kappa）=====
    if MAKE_PLOTS and curves_by_model:
        import matplotlib.pyplot as plt

        models = [m for m in MODEL_LIST if m in curves_by_model]
        n_models = len(models)
        fig_h = 2.1 * n_models  # 每个模型一行
        fig = plt.figure(figsize=(12, max(6, fig_h)))
        gs = fig.add_gridspec(nrows=n_models, ncols=2, hspace=0.35, wspace=0.25)

        for i, mdl in enumerate(models):
            dfm = curves_by_model[mdl]
            sub = dfm[dfm["scenario"] == "internal_train_cv"].sort_values("k")

            # 左：Accuracy
            ax1 = fig.add_subplot(gs[i, 0])
            xs = sub["k"].values; ys = sub["Accuracy"].values
            ax1.plot(xs, ys, lw=1.5)
            # 标记 Accuracy 最佳 k
            idx_best_acc = int(np.nanargmax(ys))
            k_best_acc = int(xs[idx_best_acc]); y_best_acc = float(ys[idx_best_acc])
            ax1.scatter([k_best_acc], [y_best_acc], marker="^", s=40)
            ax1.axvline(k_best_acc, ls="--", lw=1)
            ax1.set_ylabel("Accuracy (CV)")
            ax1.set_title(mdl.upper())
            if i == n_models - 1: ax1.set_xlabel("Variables (Top-k)")
            ax1.grid(True, ls="--", alpha=0.35)

            # 右：Kappa
            ax2 = fig.add_subplot(gs[i, 1])
            ys2 = sub["Kappa"].values
            ax2.plot(xs, ys2, lw=1.5)
            idx_best_kappa = int(np.nanargmax(ys2))
            k_best_kappa = int(xs[idx_best_kappa]); y_best_kappa = float(ys2[idx_best_kappa])
            ax2.scatter([k_best_kappa], [y_best_kappa], marker="^", s=40)
            ax2.axvline(k_best_kappa, ls="--", lw=1)
            ax2.set_ylabel("Kappa (CV)")
            if i == n_models - 1: ax2.set_xlabel("Variables (Top-k)")
            ax2.grid(True, ls="--", alpha=0.35)

        fig.suptitle("Recursive feature elimination (train CV): Accuracy & Kappa vs Variables", y=0.995, fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.98])

        png = OUT_DIR / "plots" / "S4_like_trainCV.png"
        pdf = OUT_DIR / "plots" / "S4_like_trainCV.pdf"
        fig.savefig(png, dpi=220); fig.savefig(pdf)
        plt.close(fig)
        print(f"[save] S4-like figure -> {png}")

    print("[done] train-only CV RFE-style figure finished.")

if __name__ == "__main__":
    main()
