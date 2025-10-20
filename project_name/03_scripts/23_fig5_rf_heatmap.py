# -*- coding: utf-8 -*-
# 23_fig5_rf_heatmap_pretty.py — RF heatmap styled like the paper's Fig.5
# 上：样本×变量热图（按组内 RF 概率排序）；
# 下：每个变量的迷你色条（连续变量显示数值范围；类别变量显示端点标签）。
# 依赖 S7 概率文件（train/test）与 CHARLS mean_mode_Xy + 划分索引。

import os; os.environ["MPLBACKEND"] = "Agg"
import matplotlib; matplotlib.use("Agg")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.colors import LinearSegmentedColormap
import yaml

# ---------------- 配色 ----------------
def morandi_cmap():
    colors = ["#f3f1ed","#e2dbcf","#cdbfb1","#b7a99a","#a0b2ad","#8ea7a1","#6f8b88"]
    return LinearSegmentedColormap.from_list("morandi", colors, N=256)

CMAP = morandi_cmap()

# ============ 读取配置（自动路径） ============
def repo_root() -> Path:
    # 脚本位于 03_scripts/ 下；父级为项目根
    return Path(__file__).resolve().parents[1]

CFG = yaml.safe_load(open(repo_root() / "07_config" / "config.yaml", "r", encoding="utf-8"))
PROJECT_ROOT = Path(CFG["paths"]["project_root"])
VER_IN  = CFG.get("run_id_in",  CFG.get("run_id", "v2025-10-01"))
VER_OUT = CFG.get("run_id_out", CFG.get("run_id", "v2025-10-03"))
Y_NAME  = (CFG.get("outcome", {}) or {}).get("name", "depression_bin")

MEAN_MODE_XY = PROJECT_ROOT / "02_processed_data" / VER_IN  / "frozen" / "charls_mean_mode_Xy.csv"
TRAIN_IDX    = PROJECT_ROOT / "02_processed_data" / VER_IN  / "splits" / "charls_train_idx.csv"
TEST_IDX     = PROJECT_ROOT / "02_processed_data" / VER_IN  / "splits" / "charls_test_idx.csv"

IMP_CSV  = PROJECT_ROOT / "10_experiments" / VER_OUT / "feature_importance" / "rf_importance.csv"
RFE_BEST = PROJECT_ROOT / "10_experiments" / VER_OUT / "feature_sweep" / "S4_RFE_rf_best_features.csv"

PROB_TR = PROJECT_ROOT / "10_experiments" / VER_OUT / "final_eval_s7" / "S7_probs_internal_train.csv"
PROB_TE = PROJECT_ROOT / "10_experiments" / VER_OUT / "final_eval_s7" / "S7_probs_internal_test.csv"

OUT_PNG_DIR = PROJECT_ROOT / "10_experiments" / VER_OUT / "figs"
OUT_PNG_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG = OUT_PNG_DIR / "Fig5_rf_heatmap_pretty.png"

TOPK = 10

# ---------------- 小工具 ----------------
def _load_idx(p: Path) -> np.ndarray:
    if not p.exists():
        raise FileNotFoundError(f"缺少索引：{p}")
    return pd.read_csv(p).iloc[:, 0].astype(int).values

def _minmax01(a: np.ndarray):
    a = a.astype(float)
    mn = np.nanmin(a) if np.isfinite(np.nanmin(a)) else 0.0
    mx = np.nanmax(a) if np.isfinite(np.nanmax(a)) else 1.0
    rng = mx - mn
    if not np.isfinite(rng) or rng <= 0:
        return np.zeros_like(a, dtype=float), float(mn), float(mx)
    return (a - mn) / rng, float(mn), float(mx)

def _fmt_tick(v: float) -> str:
    if abs(v) >= 100:   return f"{v:.0f}"
    if abs(v) >= 10:    return f"{v:.0f}"
    if abs(v) >= 1:     return f"{v:.1f}"
    return f"{v:.2f}"

PRIORITY_MODELS = [
    "rf", "xgb", "catboost", "lgb", "gb", "extra_trees", "adaboost", "lr"
]

def _common_prob_col(dfp_tr: pd.DataFrame, dfp_te: pd.DataFrame) -> tuple[str, str] | None:
    """在 train/test 概率表中选择共同可用的模型列，返回(col_tr, model_key)。
    优先顺序由 PRIORITY_MODELS 决定。列名采用严格形式 p_<model_key>。
    """
    cols_tr = [c for c in dfp_tr.columns if c.startswith("p_")]
    cols_te = [c for c in dfp_te.columns if c.startswith("p_")]
    keys_tr = {c[2:].lower() for c in cols_tr}
    keys_te = {c[2:].lower() for c in cols_te}
    common = list(keys_tr & keys_te)
    if not common:
        return None
    for k in PRIORITY_MODELS:
        if k in common:
            return (f"p_{k}", k)
    # 兜底：按字母序选一个
    k0 = sorted(common)[0]
    return (f"p_{k0}", k0)

def _read_top_features() -> list[str]:
    # 1) S5 importance（优先，且有 importance 排序）
    if IMP_CSV.exists():
        df = pd.read_csv(IMP_CSV)
        if "feature" in df.columns and "importance" in df.columns:
            feats = (df[["feature","importance"]]
                     .sort_values("importance", ascending=False)["feature"]
                     .astype(str).tolist())
            return feats
        # fallback: 第一列当作特征名
        c0 = df.columns[0]
        if c0:
            return df[c0].astype(str).tolist()
    # 2) S4 RFE best list
    if RFE_BEST.exists():
        df = pd.read_csv(RFE_BEST)
        col = "feature" if "feature" in df.columns else df.columns[0]
        return df[col].astype(str).tolist()
    return []

def _find_y_col(df: pd.DataFrame) -> str:
    for cand in [Y_NAME, "y"]:
        if cand in df.columns:
            return cand
    raise KeyError("概率文件中找不到标签列（尝试 config.outcome.name 与 'y'）")

# ---------------- 主流程 ----------------
def main():
    # 读数据
    if not MEAN_MODE_XY.exists():
        raise FileNotFoundError(f"缺少数据：{MEAN_MODE_XY}")
    df = pd.read_csv(MEAN_MODE_XY)
    if Y_NAME not in df.columns:
        raise KeyError(f"'{Y_NAME}' 不在 {MEAN_MODE_XY}")

    tr = _load_idx(TRAIN_IDX)
    te = _load_idx(TEST_IDX)

    df_tr = df.loc[tr].copy(); df_tr["group"] = "Train"
    df_te = df.loc[te].copy(); df_te["group"] = "Test"
    dfc = pd.concat([df_tr, df_te], axis=0, ignore_index=True)

    y = dfc[Y_NAME].astype(int).to_numpy()

    # 选择 RF Top 特征
    feats_all = _read_top_features()
    if not feats_all:
        # 退回为除 y 与 group 外的全部特征
        feats_all = [c for c in df.columns if c not in (Y_NAME,)]
        print("[WARN] 找不到 S5/S4 特征列表，回退为全特征（已去掉 y）")
    feats = [c for c in feats_all if c in dfc.columns and c != Y_NAME][:TOPK]
    if not feats:
        raise RuntimeError("无可用特征用于热图。")

    # RF 概率（与 S7 对齐）
    if not PROB_TR.exists() or not PROB_TE.exists():
        raise FileNotFoundError("需要 S7_probs_internal_train.csv 与 S7_probs_internal_test.csv")

    dfp_tr = pd.read_csv(PROB_TR)
    dfp_te = pd.read_csv(PROB_TE)

    # y 列名兼容
    ycol_tr = _find_y_col(dfp_tr)
    ycol_te = _find_y_col(dfp_te)

    # 概率列识别（选 train/test 共同可用的一个模型，优先 RF，不在则选 XGB/CATBOOST/LGB 等）
    pick = _common_prob_col(dfp_tr, dfp_te)
    if pick is None:
        raise RuntimeError("在 S7 概率文件中未找到 train/test 共同的模型概率列（p_<model>）。")
    col_name, model_key = pick
    col_tr = col_te = col_name

    # 长度校验
    if len(dfp_tr) != len(df_tr) or len(dfp_te) != len(df_te):
        raise RuntimeError(
            f"概率文件行数与切分后的样本数不一致：\n"
            f"  train probs={len(dfp_tr)} vs train rows={len(df_tr)}\n"
            f"  test  probs={len(dfp_te)} vs test  rows={len(df_te)}"
        )

    p_tr = dfp_tr[col_tr].to_numpy(dtype=float)
    p_te = dfp_te[col_te].to_numpy(dtype=float)

    # 组内排序（升序更平滑；如需降序可改为 np.argsort(-p_tr) / -p_te）
    order_tr = np.argsort(p_tr)
    order_te = np.argsort(p_te)
    offset = len(df_tr)
    col_order = np.r_[order_tr, order_te + offset]  # 先 Train，再 Test

    # --------- 构建热图矩阵 + 每行元信息 ---------
    mats = []
    row_meta = []   # [{name, kind, vmin, vmax, cats(optional)}]

    for f in feats:
        s = dfc[f]
        # 连续：>5 个唯一值
        if pd.api.types.is_numeric_dtype(s) and s.dropna().nunique() > 5:
            v01, vmin, vmax = _minmax01(s.to_numpy())
            mats.append(v01)
            row_meta.append({"name": f, "kind": "cont", "vmin": vmin, "vmax": vmax})
        else:
            # 类别：用类别码（0..k-1），缺失用众数填充
            cats = pd.Categorical(s.astype(str))
            codes = cats.codes.astype(float)  # -1 为缺失
            if np.all(codes < 0):
                codes[:] = 0
            else:
                mask = codes >= 0
                vals, cnts = np.unique(codes[mask], return_counts=True)
                fillv = vals[np.argmax(cnts)] if len(vals) else 0
                codes = np.where(codes < 0, fillv, codes)
            v01, _, _ = _minmax01(codes)
            mats.append(v01)
            row_meta.append({"name": f, "kind": "cat", "vmin": 0.0, "vmax": float(max(1, len(cats.categories)-1)),
                             "cats": cats.categories.tolist()})

    # 追加三行：<MODEL>_pred、Outcome、Group(Test=1)
    pr = np.r_[p_tr, p_te]
    mats.append(pr.astype(float))
    row_meta.append({"name":f"{model_key.upper()} predicted value", "kind":"cont", "vmin":0.0, "vmax":1.0})

    y_all = np.r_[dfp_tr[ycol_tr].values.astype(int), dfp_te[ycol_te].values.astype(int)]
    mats.append(y_all.astype(float))
    row_meta.append({"name":"Outcome", "kind":"cat", "vmin":0.0, "vmax":1.0, "cats":["0","1"]})

    mats.append((dfc["group"]=="Test").to_numpy().astype(float))
    row_meta.append({"name":"Group", "kind":"cat", "vmin":0.0, "vmax":1.0, "cats":["Training set","Test set"]})

    M = np.vstack(mats)[:, col_order]
    row_labels = [m["name"] for m in row_meta]

    # --------- 画图 ---------
    fig = plt.figure(figsize=(14, 6.8), dpi=260)
    gs = GridSpec(2, 1, height_ratios=[3.2, 1.9], hspace=0.18, figure=fig)

    # 主热图
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(M, aspect="auto", interpolation="nearest", cmap=CMAP, vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xticks([])
    ax.set_xlabel("Samples (Train sorted by RF prob, then Test)")

    # 分隔线：特征区 / 元信息区
    ax.hlines([len(feats)-0.5], xmin=-0.5, xmax=M.shape[1]-0.5, colors="k", lw=0.6, alpha=0.5)
    # 分隔线：Train / Test
    ax.vlines([len(order_tr)-0.5], ymin=-0.5, ymax=M.shape[0]-0.5, colors="k", lw=0.6, alpha=0.6)

    # 颜色条（整体）
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.015)
    cbar.set_label("Scaled value (row-wise)")

    # 下面的迷你色条区
    gs_legend = GridSpecFromSubplotSpec(2, 6, subplot_spec=gs[1, 0], hspace=0.55, wspace=0.6)
    n_leg = min(len(row_meta), 12)

    def _add_cont_bar(axb, name, vmin, vmax):
        grad = np.linspace(0, 1, 256)[None, :]
        axb.imshow(grad, aspect="auto", cmap=CMAP, vmin=0, vmax=1)
        axb.set_yticks([])
        axb.set_xticks([0, grad.shape[1]//2, grad.shape[1]-1])
        axb.set_xticklabels([_fmt_tick(vmin), _fmt_tick((vmin+vmax)/2), _fmt_tick(vmax)], fontsize=8)
        axb.set_title(name, fontsize=9, pad=2)

    def _add_cat_bar(axb, name, cats):
        cats = list(map(str, cats)) if cats else ["0","1"]
        k = max(2, len(cats))
        row = np.arange(k)[None, :]
        axb.imshow(row, aspect="auto", cmap=CMAP, vmin=0, vmax=max(1, k-1))
        axb.set_yticks([])
        if k <= 2:
            axb.set_xticks([0, 1]); labels = cats if len(cats) == 2 else ["0","1"]
        else:
            axb.set_xticks([0, k-1]); labels = [cats[0], cats[-1]]
        axb.set_xticklabels(labels, fontsize=8)
        axb.set_title(name, fontsize=9, pad=2)

    for i in range(n_leg):
        r = row_meta[i]
        axb = fig.add_subplot(gs_legend[i//6, i%6])
        if r["kind"] == "cont":
            _add_cont_bar(axb, r["name"], r["vmin"], r["vmax"])
        else:
            _add_cat_bar(axb, r["name"], r.get("cats", ["0","1"]))

    fig.suptitle("RF variable heatmap — values and legends", y=0.995, fontsize=14)
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300)
    fig.savefig(OUT_PNG.with_suffix(".pdf"))
    print("[save]", OUT_PNG)

if __name__ == "__main__":
    main()
