# -*- coding: utf-8 -*-
"""
28_rf_shap_standalone.py — 使用冻结版全量(Xy)训练随机森林并输出SHAP图

数据:
- 冻结版Xy: 02_processed_data/<VER_IN>/frozen/charls_mean_mode_Xy.csv
- 划分索引(可选): 02_processed_data/<VER_IN>/splits/charls_train_idx.csv / charls_test_idx.csv

输出:
- 10_experiments/<VER_OUT>/figs/
  - RF_SHAP_bar.png/.pdf
  - RF_SHAP_beeswarm.png/.pdf
  - RF_SHAP_waterfall.png/.pdf
  - RF_SHAP_dependence.png/.pdf
"""

import os
os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path
from typing import Optional, Tuple, List
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


CFG = yaml.safe_load(open(repo_root() / "07_config" / "config.yaml", "r", encoding="utf-8"))
PROJECT_ROOT = Path(CFG["paths"]["project_root"])
VER_IN  = CFG.get("run_id_in",  CFG.get("run_id"))
VER_OUT = CFG.get("run_id_out", CFG.get("run_id"))
Y_NAME  = (CFG.get("outcome", {}) or {}).get("name", "depression_bin")

FROZEN_CSV = PROJECT_ROOT / "02_processed_data" / VER_IN / "frozen" / "charls_mean_mode_Xy.csv"
TRAIN_IDX  = PROJECT_ROOT / "02_processed_data" / VER_IN / "splits" / "charls_train_idx.csv"
TEST_IDX   = PROJECT_ROOT / "02_processed_data" / VER_IN / "splits" / "charls_test_idx.csv"

OUT_DIR = PROJECT_ROOT / "10_experiments" / VER_OUT / "figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def infer_types(df: pd.DataFrame, y_name: Optional[str]) -> Tuple[List[str], List[str]]:
    feats = [c for c in df.columns if (y_name is None or c != y_name)]
    num, cat = [], []
    for c in feats:
        s = df[c]
        if np.issubdtype(s.dtype, np.number) and pd.Series(s).dropna().nunique() > 5:
            num.append(c)
        else:
            cat.append(c)
    return num, cat


def robust_tree_shap(clf, Xt):
    import shap
    Xt = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)
    try:
        explainer = shap.TreeExplainer(clf, feature_perturbation="interventional", model_output="probability")
    except Exception:
        try:
            explainer = shap.TreeExplainer(clf, model_output="raw")
        except Exception:
            explainer = shap.TreeExplainer(clf)

    # 新接口优先
    try:
        exp = explainer(Xt, check_additivity=False)
        vals = getattr(exp, "values", None)
        base = getattr(exp, "base_values", None)
        if vals is not None:
            if vals.ndim == 2:
                return explainer, vals, float(np.mean(base)) if np.ndim(base) else float(base)
            if vals.ndim == 3:
                return explainer, vals[:, 1, :], float(np.mean(base[:, 1]))
    except Exception:
        pass

    # 旧接口
    sv = explainer.shap_values(Xt)
    if isinstance(sv, list):
        shap_vals = np.asarray(sv[1]) if len(sv) > 1 else np.asarray(sv[0])
        ev = explainer.expected_value
        base = ev[1] if isinstance(ev, (list, np.ndarray)) and len(ev) > 1 else ev
    else:
        shap_vals = np.asarray(sv)
        base = explainer.expected_value
    base = float(np.mean(base)) if np.ndim(base) else float(base)
    return explainer, shap_vals, base


def main():
    if not FROZEN_CSV.exists():
        raise FileNotFoundError(f"找不到数据: {FROZEN_CSV}")
    df = pd.read_csv(FROZEN_CSV)
    if Y_NAME not in df.columns:
        raise KeyError(f"'{Y_NAME}' 不在 {FROZEN_CSV}")

    # 丢弃潜在泄露列（如 cesd）
    drop_cols = list(((CFG.get("preprocess", {}) or {}).get("drop_cols", []) or []))
    if drop_cols:
        keep = [c for c in df.columns if c not in drop_cols]
        if len(keep) < len(df.columns):
            df = df[keep].copy()

    # 切分: 优先用已有索引
    if TRAIN_IDX.exists() and TEST_IDX.exists():
        tr = pd.read_csv(TRAIN_IDX).iloc[:, 0].astype(int).values
        te = pd.read_csv(TEST_IDX).iloc[:, 0].astype(int).values
        df_tr = df.iloc[tr].reset_index(drop=True)
        df_te = df.iloc[te].reset_index(drop=True)
    else:
        from sklearn.model_selection import train_test_split
        df_tr, df_te = train_test_split(df, test_size=0.2, stratify=df[Y_NAME], random_state=42)

    Xtr = df_tr.drop(columns=[Y_NAME])
    y_tr = df_tr[Y_NAME].astype(int).values
    Xte = df_te.drop(columns=[Y_NAME])
    y_te = df_te[Y_NAME].astype(int).values

    # 预处理 - 树模型直接使用原始特征，不做标准化和One-Hot
    num, cat = infer_types(pd.concat([Xtr, Xte], axis=0), y_name=None)
    pre = ColumnTransformer([
        ("num", FunctionTransformer(lambda X: X.astype(float)), num),
        ("cat", FunctionTransformer(lambda X: X.astype(float)), cat)
    ], remainder="drop")

    rf = RandomForestClassifier(
        n_estimators=600,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=min(4, max((os.cpu_count() or 4) - 1, 1))
    )
    pipe = Pipeline([("pre", pre), ("clf", rf)])
    pipe.fit(Xtr, y_tr)

    # 测试AUC
    try:
        Xte_t = pipe.named_steps["pre"].transform(Xte)
        p_te = pipe.named_steps["clf"].predict_proba(Xte_t)[:, 1]
        uniq = np.unique(y_te)
        if len(uniq) > 1:
            auc = roc_auc_score(y_te, p_te)
            print(f"[AUC test] RF = {auc:.3f}")
        else:
            print("[AUC test] 单一类别，跳过AUC计算")
    except Exception as ex:
        print(f"[AUC test] 计算失败: {ex}")

    # SHAP
    Xtr_t = pipe.named_steps["pre"].transform(Xtr)
    feat_names = pipe.named_steps["pre"].get_feature_names_out()
    clf = pipe.named_steps["clf"]
    explainer, shap_values, base = robust_tree_shap(clf, Xtr_t)
    Xtr_td = Xtr_t.toarray() if hasattr(Xtr_t, "toarray") else np.asarray(Xtr_t)

    # 裁剪对齐（防御性）
    d = int(min(shap_values.shape[1], Xtr_td.shape[1], len(feat_names)))
    shap_values = shap_values[:, :d]
    Xtr_td = Xtr_td[:, :d]
    try:
        feat_names = list(feat_names)[:d]
    except Exception:
        feat_names = [f"f{i}" for i in range(d)]

    import shap

    # A: bar
    plt.figure(figsize=(9, 6), dpi=240)
    shap.summary_plot(shap_values, features=Xtr_td, feature_names=feat_names,
                      plot_type="bar", show=False, max_display=20)
    plt.title("RF SHAP — mean(|SHAP|)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "RF_SHAP_bar.png", dpi=300)
    plt.savefig(OUT_DIR / "RF_SHAP_bar.pdf")
    plt.close()

    # B: beeswarm
    plt.figure(figsize=(9, 6), dpi=240)
    shap.summary_plot(shap_values, features=Xtr_td, feature_names=feat_names,
                      show=False, max_display=20)
    plt.title("RF SHAP — beeswarm (train)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "RF_SHAP_beeswarm.png", dpi=300)
    plt.savefig(OUT_DIR / "RF_SHAP_beeswarm.pdf")
    plt.close()

    # C: waterfall（取训练集中预测概率最高样本）
    p_tr = pipe.named_steps["clf"].predict_proba(Xtr_t)[:, 1]
    idx_top = int(np.argmax(p_tr))
    try:
        e = shap.Explanation(values=shap_values[idx_top],
                             base_values=base,
                             data=Xtr_td[idx_top],
                             feature_names=feat_names)
        plt.figure(figsize=(8, 6), dpi=240)
        shap.plots.waterfall(e, max_display=20, show=False)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "RF_SHAP_waterfall.png", dpi=300)
        plt.savefig(OUT_DIR / "RF_SHAP_waterfall.pdf")
        plt.close()
    except Exception as ex:
        print("[WARN] waterfall 绘制失败:", ex)

    # D: dependence（Top-9）
    try:
        mean_abs = np.mean(np.abs(shap_values), axis=0)
        order = np.argsort(mean_abs)[::-1][:9]
        rows, cols = 3, 3
        fig, axes = plt.subplots(rows, cols, figsize=(10, 8), dpi=240)
        for ax, j in zip(axes.ravel(), order):
            shap.dependence_plot(j, shap_values, Xtr_td, feature_names=feat_names, show=False, ax=ax)
        fig.suptitle("RF SHAP — dependence (top features)")
        fig.tight_layout()
        plt.savefig(OUT_DIR / "RF_SHAP_dependence.png", dpi=300)
        plt.savefig(OUT_DIR / "RF_SHAP_dependence.pdf")
        plt.close(fig)
    except Exception as ex:
        print("[WARN] dependence 绘制失败:", ex)

    print("[save]", OUT_DIR)


if __name__ == "__main__":
    main()



