# -*- coding: utf-8 -*-
"""
26.py — 读取 25步缓存矩阵，训练 RF，计算并绘制 SHAP（聚合到原始变量）
修复：
- 兼容 shap 返回 3D (n,f,2) 的情形，统一转成二维 (n,f)
- build_agg_matrices 内部再次兜底，确保二维
- 风格统一为 SHAP 官方配色；依赖图 3×3（Top-9），分类横轴取整数刻度
"""

from pathlib import Path
import os, json, warnings
import joblib
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# ----------------- 路径/配置 -----------------
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

CFG = yaml.safe_load(open(repo_root()/ "07_config"/"config.yaml","r",encoding="utf-8"))
PROJECT_ROOT = Path(CFG["paths"]["project_root"])
VER_IN  = CFG.get("run_id_in", CFG.get("run_id"))
VER_OUT = CFG.get("run_id_out", CFG.get("run_id"))
Y_NAME  = (CFG.get("outcome", {}) or {}).get("name", "depression_bin")

CACHE_DIR = PROJECT_ROOT / "10_experiments" / VER_OUT / "cache" / "shap_rf"
FIG_DIR   = PROJECT_ROOT / "10_experiments" / VER_OUT / "figs"
TUNE_JSON = PROJECT_ROOT / "10_experiments" / VER_OUT / "tuning" / "best_params.json"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 原始数据与测试索引（用于测试集AUC验证）
MEAN_MODE_XY = PROJECT_ROOT / "02_processed_data" / VER_IN / "frozen" / "charls_mean_mode_Xy.csv"
TEST_IDX     = PROJECT_ROOT / "02_processed_data" / VER_IN / "splits" / "charls_test_idx.csv"

# 为了反序列化 25 步保存的 pre.joblib 中的 FunctionTransformer，
# 需要在 __main__ 下提供同名函数 _astype_float
def _astype_float(X):
    try:
        return X.astype(float)
    except Exception:
        return np.asarray(X, dtype=float)

# ----------------- 小工具 -----------------
def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))

def get_rf_params_from_tuning():
    if TUNE_JSON.exists():
        try:
            bp = load_json(TUNE_JSON)
            rf = (bp.get("rf") or {}).get("params") or {}
            if rf.get("max_features", None) == "auto":
                rf["max_features"] = "sqrt"
            # 合理的并行&随机种子
            rf.setdefault("class_weight", "balanced")
            rf.setdefault("n_jobs", min(4, max((os.cpu_count() or 4) - 1, 1)))
            rf.setdefault("random_state", int(CFG.get("random_state", 42)))
            return rf
        except Exception:
            pass
    return dict(
        n_estimators=600, max_depth=None, min_samples_leaf=1,
        class_weight="balanced", n_jobs=min(4, max((os.cpu_count() or 4)-1, 1)),
        random_state=int(CFG.get("random_state", 42)), max_features="sqrt"
    )

def robust_tree_explainer(clf, X_bg):
    import shap
    try:
        masker = shap.maskers.Independent(X_bg)
        return shap.TreeExplainer(clf, data=masker, model_output="probability")
    except Exception:
        return shap.TreeExplainer(clf)

def extract_class1_matrix(Sv):
    """把 shap 值统一成 (n_samples, n_features) 的 ndarray（取二分类第1类）。"""
    import numpy as np
    try:
        import shap
        from shap import Explanation
    except Exception:
        shap = None
        Explanation = tuple()

    # list: [class0, class1]
    if isinstance(Sv, list):
        S = Sv[1] if len(Sv) >= 2 else Sv[0]
        if hasattr(S, "values"):  # shap.Explanation
            S = S.values
        arr = np.asarray(S)
    elif shap is not None and isinstance(Sv, Explanation):
        arr = np.asarray(Sv.values)
    else:
        arr = np.asarray(Sv)

    # 若 3D，取最后一维 index=1（否则取最后一个）
    if arr.ndim == 3:
        arr = arr[:, :, 1] if arr.shape[2] >= 2 else arr[:, :, -1]
    return arr  # (n,f)

def flatten_1d(x, to_len=None):
    arr = np.asarray(x).ravel()
    if to_len is not None:
        arr = arr[:min(len(arr), to_len)]
    return arr

def kind_and_root(feat_name: str):
    """返回(类型, 原始变量名)，支持 bin__/mul__/num__ 命名约定。"""
    if "__" in feat_name:
        head, tail = feat_name.split("__", 1)
        if head in ("bin", "mul", "num"):
            if head == "mul" and "=" in tail:
                root = tail.split("=")[0]
            else:
                root = tail
            return head, root
    # 回退：当作数值
    return "num", feat_name

def build_agg_matrices(S_full, X_full, feat_names):
    """把 one-hot 列按原始变量聚合：返回 S_agg(n,k), X_agg(n,k), 原名列表, 类型列表."""
    S_full = np.asarray(S_full)
    # 兜底：若传进来是 3D，则转为二维
    if S_full.ndim == 3:
        S_full = S_full[:, :, 1] if S_full.shape[2] >= 2 else S_full[:, :, -1]
    n, f = S_full.shape

    feat_names = [str(x) for x in flatten_1d(feat_names, to_len=f)]
    X_full = np.asarray(X_full)  # (n,f)

    groups = {}  # root -> list of indices
    kinds  = {}  # root -> kind
    for j, name in enumerate(feat_names):
        k, root = kind_and_root(name)
        groups.setdefault(root, []).append(j)
        kinds[root] = k

    roots = list(groups.keys())
    S_list, X_list, typ_list = [], [], []
    for r in roots:
        idx = groups[r]
        if kinds[r] == "num":
            # 数值：直接取该列（允许多列时取第一列）
            S_list.append(S_full[:, idx[0]])
            X_list.append(X_full[:, idx[0]])
        else:
            # 分类：把该原始变量所有 one-hot 的 |SHAP| 相加
            S_list.append(np.sum(np.abs(S_full[:, idx]), axis=1))
            # 横轴用“原始取值编码”：one-hot 列名里的枚举值（粗略用最大 one-hot 列位置代表）
            X_list.append(np.argmax(X_full[:, idx], axis=1))
        typ_list.append(kinds[r])

    S_agg = np.vstack(S_list).T   # (n,k)
    X_agg = np.vstack(X_list).T
    return S_agg, X_agg, roots, typ_list

# ----------------- 主流程 -----------------
def main():
    # 读缓存矩阵
    Xt = np.load(CACHE_DIR / "X_train.npy")
    yt = np.load(CACHE_DIR / "y_train.npy")
    Xb = np.load(CACHE_DIR / "X_bg.npy")
    try:
        feat_names = load_json(CACHE_DIR / "feature_names.json")
    except Exception:
        feat_names = [f"f{i}" for i in range(Xt.shape[1])]
    feat_names = [str(x) for x in flatten_1d(feat_names)]
    if len(feat_names) != Xt.shape[1]:
        feat_names = [f"f{i}" for i in range(Xt.shape[1])]

    print(f"[cfg ] VER_OUT={VER_OUT}")
    print(f"[data] Xt={Xt.shape}, Xb={Xb.shape}, y={yt.shape}, features={len(feat_names)}")

    # 训练 RF（用 11 步最佳参数，或回退）
    rf_params = get_rf_params_from_tuning()
    clf = RandomForestClassifier(**rf_params)
    clf.fit(Xt, yt)

    # 训练集 AUC（检查）
    try:
        p = clf.predict_proba(Xt)[:,1]
        auc = roc_auc_score(yt, p)
        print(f"[AUC train] RF = {auc:.3f}")
    except Exception:
        pass

    # 测试集 AUC（与25步预处理完全一致）
    try:
        if MEAN_MODE_XY.exists() and TEST_IDX.exists():
            df_all = pd.read_csv(MEAN_MODE_XY)
            if Y_NAME in df_all.columns:
                drop_cols = list(((CFG.get("preprocess", {}) or {}).get("drop_cols", []) or []))
                if drop_cols:
                    keep = [c for c in df_all.columns if c not in drop_cols]
                    df_all = df_all[keep].copy()
                te_idx = pd.read_csv(TEST_IDX).iloc[:,0].astype(int).values
                X_raw = df_all.drop(columns=[Y_NAME]).iloc[te_idx].copy()
                y_true = pd.to_numeric(pd.read_csv(MEAN_MODE_XY).iloc[te_idx][Y_NAME], errors="coerce").fillna(0).astype(int).values
                pre = joblib.load(CACHE_DIR / "pre.joblib")
                X_te = pre.transform(X_raw)
                p_te = clf.predict_proba(X_te)[:,1]
                auc_te = roc_auc_score(y_true, p_te)
                print(f"[AUC test] RF (pre from step25) = {auc_te:.3f}")
    except Exception as ex:
        print(f"[WARN] 测试集AUC计算失败：{ex}")

    # SHAP
    import shap
    explainer = robust_tree_explainer(clf, Xb)
    raw_Sv = explainer(Xt, check_additivity=False) if hasattr(explainer, "__call__") else explainer.shap_values(Xt, check_additivity=False)
    Sv = extract_class1_matrix(raw_Sv)  # -> (n,f)

    # === 聚合到原始变量 ===
    S_agg, X_agg, roots, kinds = build_agg_matrices(Sv, Xt, feat_names)  # (n,k)

    # ---- (1) 条形图：mean(|SHAP|) Top-20 ----
    imp = np.mean(np.abs(S_agg), axis=0)
    order = np.argsort(imp)[::-1]
    top_idx = order[:20]
    plt.figure(figsize=(12, 7))
    shap.summary_plot(
        S_agg[:, top_idx], features=X_agg[:, top_idx], feature_names=[roots[i] for i in top_idx],
        plot_type="bar", show=False, max_display=len(top_idx)
    )
    out_bar = FIG_DIR / "Fig6_shap_bar_agg.png"
    plt.tight_layout(); plt.savefig(out_bar, dpi=300, bbox_inches="tight"); plt.close()

    # ---- (2) 蜂群图：同样的 Top-20 ----
    plt.figure(figsize=(9, 11))
    shap.summary_plot(
        S_agg[:, top_idx], features=X_agg[:, top_idx], feature_names=[roots[i] for i in top_idx],
        show=False, max_display=len(top_idx)
    )
    out_bee = FIG_DIR / "Fig6_shap_beeswarm_agg.png"
    plt.tight_layout(); plt.savefig(out_bee, dpi=300, bbox_inches="tight"); plt.close()

    # ---- (3) 依赖图：3×3（Top-9），分类取整数刻度 ----
    top9 = order[:9]
    nrow, ncol = 3, 3
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 12))
    axes = axes.ravel()
    for ax, i in zip(axes, top9):
        s = S_agg[:, i]
        x = X_agg[:, i]
        ax.scatter(x, s, s=6, alpha=0.6)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_title(roots[i])
        # 分类特征：整数刻度
        if kinds[i] in ("bin", "mul"):
            uniq = np.unique(x.astype(int))
            ax.set_xticks(uniq)
        ax.grid(False)
    for j in range(len(top9), len(axes)):
        fig.delaxes(axes[j])
    out_grid = FIG_DIR / "Fig6_shap_dependence_grid.png"
    plt.tight_layout(); plt.savefig(out_grid, dpi=300, bbox_inches="tight"); plt.close()

    print("[save]", out_bar)
    print("[save]", out_bee)
    print("[save]", out_grid)
    print("[done] SHAP figures ready (aggregated).")

if __name__ == "__main__":
    main()
