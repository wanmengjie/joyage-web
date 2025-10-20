# -*- coding: utf-8 -*-
r"""
12_feature_importance_rfe.py — S5：基于 S4 已确定的每模型最佳特征，在训练集拟合并导出特征重要度
产物：
  1) feature_importance/<model>_importance.csv    （按重要度降序）
  2) feature_importance/plots/<model>_importance.png / .pdf  （单模型条形图，默认前20）
  3) feature_importance/plots/S5_feature_importance_grid.png / .pdf  （总览）

依赖与假设：
  - 固定插补：mean_mode（charls_mean_mode_Xy.csv）
  - 只用训练集拟合（与 PDF 的 S5 思路一致）
  - 已运行 S4（存在 S4_RFE_<model>_best_features.csv，若无则回退全量）
  - 若已运行 11_tune_models_nestedcv.py，将自动读取 tuning/best_params.json 套用超参
"""

# —— 强制使用无图形后端，避免 Windows Tk 异常 —— #
import os
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json, warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import yaml

# ===================== 统一读取配置 =====================
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

CFG = yaml.safe_load(open(repo_root()/ "07_config"/"config.yaml","r",encoding="utf-8"))

PROJECT_ROOT = Path(CFG["paths"]["project_root"])
VER_IN   = CFG.get("run_id_in",  CFG.get("run_id"))
VER_OUT  = CFG.get("run_id_out", CFG.get("run_id"))
RANDOM_STATE = int(CFG.get("random_state", 42))

# 数据与索引
MEAN_MODE_XY = PROJECT_ROOT / "02_processed_data" / VER_IN / "frozen" / "charls_mean_mode_Xy.csv"
TRAIN_IDX    = PROJECT_ROOT / "02_processed_data" / VER_IN / "splits" / "charls_train_idx.csv"

# S4 产物目录（用于读取每模型的最佳特征列表）
FEATURE_SWEEP_DIR = PROJECT_ROOT / "10_experiments" / VER_OUT / "feature_sweep"

# 若存在，将读取并应用最优超参
TUNING_JSON = PROJECT_ROOT / "10_experiments" / VER_OUT / "tuning" / "best_params.json"

# S5 输出目录
OUT_DIR   = PROJECT_ROOT / "10_experiments" / VER_OUT / "feature_importance"
PLOTS_DIR = OUT_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# 标签名/ID列（统一从配置读取）
Y_NAME   = (CFG.get("outcome", {}) or {}).get("name", "depression_bin")
ID_COLS  = list(CFG.get("id_cols") or []) or ["ID", "householdID"]

# 使用的模型（与前文一致）
MODEL_LIST = ["lr", "rf", "extra_trees", "gb", "adaboost", "lgb", "xgb"]  # 无 catboost

# 单模型图展示的 Top-N
TOP_N = 20
# ===================== 配置区结束 =====================


# ---------- sklearn 预处理与模型 ----------
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline

def make_ohe():
    """兼容 sklearn 版本的 OneHotEncoder"""
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

# try:
#     from catboost import CatBoostClassifier
#     _HAS_CAT = True
# except Exception:
#     _HAS_CAT = False


def _class_ratio(y: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    pos = y.sum(); neg = len(y) - pos
    return (neg / max(pos, 1))


def make_model(model_key: str, y_train: np.ndarray):
    mk = model_key.lower()
    if mk == "lr":
        return LogisticRegression(solver="liblinear", penalty="l2",
                                  max_iter=500, class_weight="balanced", random_state=RANDOM_STATE)
    elif mk == "rf":
        return RandomForestClassifier(n_estimators=600, min_samples_leaf=5,
                                     class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE)
    elif mk == "extra_trees":
        return ExtraTreesClassifier(n_estimators=700, min_samples_leaf=3,
                                    class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE)
    elif mk == "gb":
        return GradientBoostingClassifier(random_state=RANDOM_STATE)
    elif mk == "adaboost":
        return AdaBoostClassifier(n_estimators=600, learning_rate=0.05, algorithm="SAMME",
                                  random_state=RANDOM_STATE)
    elif mk == "lgb":
        if _HAS_LGBM:
            return LGBMClassifier(n_estimators=700, learning_rate=0.05, max_depth=-1,
                                  subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
                                  objective="binary", random_state=RANDOM_STATE, class_weight="balanced")
        warnings.warn("[WARN] lightgbm 未安装，回退为 GB")
        return GradientBoostingClassifier(random_state=RANDOM_STATE)
    elif mk == "xgb":
        if _HAS_XGB:
            return XGBClassifier(n_estimators=800, learning_rate=0.05, max_depth=5,
                                 subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
                                 objective="binary:logistic", random_state=RANDOM_STATE, n_jobs=-1,
                                 scale_pos_weight=_class_ratio(y_train))
        warnings.warn("[WARN] xgboost 未安装，回退为 GB")
        return GradientBoostingClassifier(random_state=RANDOM_STATE)
    # elif mk == "catboost":
    #     if _HAS_CAT:
    #         w1 = _class_ratio(y_train)
    #         return CatBoostClassifier(iterations=800, learning_rate=0.05, depth=6,
    #                                   loss_function="Logloss", eval_metric="AUC",
    #                                   class_weights=[1.0, w1], random_seed=RANDOM_STATE, verbose=False)
    #     warnings.warn("[WARN] catboost 未安装，跳过")
    #     return None
    else:
        raise ValueError(f"Unsupported model: {model_key}")


# ---------- YAML（可选） ----------
def read_yaml_variable_types(yaml_path: Path) -> Tuple[List[str], List[str]]:
    """从 YAML 读取 preprocess.variable_types 的 continuous / categorical"""
    try:
        import yaml as _yaml
    except Exception as e:
        raise RuntimeError("未安装 pyyaml，无法读取 YAML") from e

    if not yaml_path.exists():
        return [], []

    with open(yaml_path, "r", encoding="utf-8") as f:
        y = _yaml.safe_load(f)

    num_cols = y.get("preprocess", {}).get("variable_types", {}).get("continuous", []) or []
    cat_cols = y.get("preprocess", {}).get("variable_types", {}).get("categorical", []) or []
    return list(num_cols), list(cat_cols)


# ---------- 列类型推断 ----------
def infer_feature_types(
    df: pd.DataFrame, y_name: str,
    init_num: Optional[List[str]] = None,
    init_mul: Optional[List[str]] = None,
    id_cols: Optional[List[str]] = None
) -> Tuple[List[str], List[str], List[str]]:
    """自动推断 + YAML 种子：把明确 continuous/categorical 作为起点，再把实际 0/1 的列归到 bin。"""
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


# ---------- 预处理器（支持只对选中特征建构；bin 强制转 float） ----------
def build_preprocessor(all_num: List[str], all_bin: List[str], all_mul: List[str],
                       selected: Optional[List[str]] = None) -> Tuple[ColumnTransformer, List[str], List[str], List[str]]:
    if selected is not None:
        s = set(selected)
        num_cols = [c for c in all_num if c in s]
        bin_cols = [c for c in all_bin if c in s]
        mul_cols = [c for c in all_mul if c in s]
    else:
        num_cols, bin_cols, mul_cols = all_num, all_bin, all_mul

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=False), num_cols),
            ("bin", Pipeline([("astype_float", FunctionTransformer(lambda X: X.astype(float))) ]), bin_cols),
            ("mul", make_ohe(), mul_cols),
        ],
        remainder="drop"
    )
    return pre, num_cols, bin_cols, mul_cols


# ---------- 把“变换后的重要度”聚合回“原始特征” ----------
def get_feature_importance_per_orig_feature(
    pipe: Pipeline, num_cols: List[str], bin_cols: List[str], mul_cols: List[str]
) -> Dict[str, float]:
    """从 pipeline(clf.coef_ 或 feature_importances_) 聚合到原始特征层面"""
    pre: ColumnTransformer = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]

    # 构建索引映射：每个原始特征 -> 变换后列索引列表
    start = 0
    mapping: Dict[str, List[int]] = {}

    # 数值与二元：1:1
    for c in num_cols:
        mapping[c] = [start]; start += 1
    for c in bin_cols:
        mapping[c] = [start]; start += 1

    # 多分类：OneHot 展开
    if len(mul_cols) > 0 and "mul" in pre.named_transformers_:
        ohe: OneHotEncoder = pre.named_transformers_["mul"]
        cats = getattr(ohe, "categories_", None) or []
        for c, cat in zip(mul_cols, cats):
            n = len(cat)
            mapping[c] = list(range(start, start + n))
            start += n

    # 分类器的重要度向量
    if hasattr(clf, "coef_"):
        imp = np.abs(np.ravel(clf.coef_))
    elif hasattr(clf, "feature_importances_"):
        imp = np.asarray(clf.feature_importances_, dtype=float)
    else:
        imp = np.zeros(start, dtype=float)

    # 健壮性：如果长度对不上，按最短截断
    if len(imp) < start:
        for k, idx in mapping.items():
            mapping[k] = [i for i in idx if i < len(imp)]
        start = len(imp)

    # 聚合
    out: Dict[str, float] = {}
    for c, idx in mapping.items():
        if len(idx) == 0:
            out[c] = 0.0
        elif len(idx) == 1:
            out[c] = float(imp[idx[0]])
        else:
            out[c] = float(np.sum(imp[idx]))
    return out


# ---------- 绘图：单模型条形图 ----------
def plot_single_bar(model_key: str, imp_df: pd.DataFrame, png_path: Path, pdf_path: Path, top_n: int = TOP_N):
    import matplotlib.pyplot as plt
    df_top = imp_df.head(top_n).copy()
    fig = plt.figure(figsize=(8, max(4, 0.35*len(df_top))))
    ax = fig.add_subplot(111)
    ax.barh(df_top["feature"][::-1], df_top["importance"][::-1])
    ax.set_title(f"{model_key} — Feature importance (train)\n(mean_mode, top-{len(df_top)})")
    ax.set_xlabel("Importance (aggregated to original feature)")
    ax.set_ylabel("")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    fig.savefig(pdf_path)
    plt.close(fig)


# ---------- 绘图：总览 ----------
def plot_grid(files_by_model: Dict[str, Path], png_path: Path, pdf_path: Path, top_n: int = TOP_N):
    import matplotlib.pyplot as plt
    models = ["lr","rf","extra_trees","gb","adaboost","lgb","xgb"]
    n = len(models)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig = plt.figure(figsize=(16, 2.4*nrow + 0.6))
    for i, mk in enumerate(models):
        ax = fig.add_subplot(nrow, ncol, i+1)
        if mk in files_by_model and Path(files_by_model[mk]).exists():
            df = pd.read_csv(files_by_model[mk]).head(top_n)
            ax.barh(df["feature"][::-1], df["importance"][::-1])
            ax.set_title(mk)
            ax.grid(axis="x", linestyle="--", alpha=0.3)
            ax.tick_params(axis="y", labelsize=8)
        else:
            ax.set_title(mk + " (no data)")
            ax.axis("off")
    fig.suptitle("S5 — Feature importance overview (train, mean_mode)", y=1.0, fontsize=12)
    fig.tight_layout()
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


# ---------- 可选：读取最优超参 ----------
def load_best_params() -> dict:
    if TUNING_JSON.exists():
        try:
            return json.load(open(TUNING_JSON, "r", encoding="utf-8"))
        except Exception:
            pass
    return {}


# ===================== 主流程 =====================
def main():
    # 必要文件检查
    for p in [MEAN_MODE_XY, TRAIN_IDX]:
        if not Path(p).exists():
            raise FileNotFoundError(f"缺少必要文件：{p}。请先完成 03 划分与 08 冻结。")

    df = pd.read_csv(MEAN_MODE_XY)
    if Y_NAME not in df.columns:
        raise KeyError(f"'{Y_NAME}' 不在 {MEAN_MODE_XY}")

    tr_idx = pd.read_csv(TRAIN_IDX).iloc[:,0].astype(int).values
    df_tr  = df.iloc[tr_idx].copy()

    # 变量类型（优先 YAML 种子）
    yaml_num = list((CFG.get("preprocess", {}) or {}).get("variable_types", {}).get("continuous", []) or [])
    yaml_cat = list((CFG.get("preprocess", {}) or {}).get("variable_types", {}).get("categorical", []) or [])

    all_num, all_bin, all_mul = infer_feature_types(df, Y_NAME, init_num=yaml_num, init_mul=yaml_cat, id_cols=ID_COLS)
    all_feats = [c for c in (all_num + all_bin + all_mul) if c in df.columns]

    # 读取最优超参（若存在）
    best_params = load_best_params()

    saved_files: Dict[str, Path] = {}

    for model_key in MODEL_LIST:
        print(f"[model] {model_key}")

        # S4 的最佳特征列表（若缺，则回退为全量）
        s4_path = FEATURE_SWEEP_DIR / f"S4_RFE_{model_key}_best_features.csv"
        if s4_path.exists():
            feats = pd.read_csv(s4_path)["feature"].dropna().astype(str).tolist()
            feats = [c for c in feats if c in all_feats]
            if not feats:
                feats = all_feats[:]
        else:
            print(f"  [info] 未发现 {s4_path.name}，回退使用全量特征")
            feats = all_feats[:]

        # 预处理 + 模型
        pre, n_cols, b_cols, m_cols = build_preprocessor(all_num, all_bin, all_mul, selected=feats)
        clf = make_model(model_key, df_tr[Y_NAME].astype(int).values)
        if clf is None:
            print(f"  [skip] {model_key} 不可用（缺依赖或被显式跳过）")
            continue

        # 应用调参结果（若存在）
        if model_key in best_params and isinstance(best_params[model_key], dict):
            params = best_params[model_key].get("params", {})
            if isinstance(params, dict) and len(params):
                try:
                    clf.set_params(**params)
                    print(f"  [cfg] use tuned params: {params}")
                except Exception as e:
                    print(f"  [warn] tuned params 无法应用：{e}")

        pipe = Pipeline([("pre", pre), ("clf", clf)])
        X_tr = df_tr[feats].copy()
        y_tr = df_tr[Y_NAME].astype(int).values
        pipe.fit(X_tr, y_tr)

        # 聚合到原始特征的重要度
        imp_map = get_feature_importance_per_orig_feature(pipe, n_cols, b_cols, m_cols)
        imp_df = (pd.DataFrame({"feature": list(imp_map.keys()), "importance": list(imp_map.values())})
                    .sort_values("importance", ascending=False)
                    .reset_index(drop=True))

        # 保存 CSV 与单模型图
        csv_path = OUT_DIR / f"{model_key}_importance.csv"
        png_path = PLOTS_DIR / f"{model_key}_importance.png"
        pdf_path = PLOTS_DIR / f"{model_key}_importance.pdf"
        imp_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        plot_single_bar(model_key, imp_df, png_path, pdf_path, top_n=TOP_N)
        saved_files[model_key] = csv_path
        print(f"  [save] {csv_path}")

    # 总览面板
    if saved_files:
        grid_png = PLOTS_DIR / "S5_feature_importance_grid.png"
        grid_pdf = PLOTS_DIR / "S5_feature_importance_grid.pdf"
        plot_grid(saved_files, grid_png, grid_pdf, top_n=TOP_N)
        print(f"[save] overview -> {grid_png}")

    print("[done] feature importance exported.")


if __name__ == "__main__":
    main()
