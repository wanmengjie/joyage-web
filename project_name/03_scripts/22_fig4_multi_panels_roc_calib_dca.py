# -*- coding: utf-8 -*-
# 22_fig4_multi_panels_roc_calib_dca.py — Multi-panel (ROC/Calibration/Metrics/DCA) + 底部共享图例
# 使用 Validation 的固定阈值（若可用），与 S7 输出保持一致；否则回退各自数据的 Youden 阈值

import os; os.environ["MPLBACKEND"] = "Agg"
import matplotlib; matplotlib.use("Agg")

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import (
    roc_curve, roc_auc_score, brier_score_loss,
    accuracy_score, precision_score, recall_score, f1_score
)
import yaml

# ================== 路径配置（从 config 读取） ==================
def repo_root() -> Path:
    # 脚本位于 03_scripts/ 下；父级为项目根
    return Path(__file__).resolve().parents[1]

CFG = yaml.safe_load(open(repo_root() / "07_config" / "config.yaml", "r", encoding="utf-8"))
PROJECT_ROOT = Path(CFG["paths"]["project_root"])
VER_OUT      = CFG.get("run_id_out", CFG.get("run_id"))
Y_NAME_CFG   = (CFG.get("outcome", {}) or {}).get("name", None)

OUT_DIR      = PROJECT_ROOT / "10_experiments" / VER_OUT / "figs"
P_TR   = PROJECT_ROOT / "10_experiments" / VER_OUT / "final_eval_s7" / "S7_probs_internal_train.csv"
P_TE   = PROJECT_ROOT / "10_experiments" / VER_OUT / "final_eval_s7" / "S7_probs_internal_test.csv"
P_VAL  = PROJECT_ROOT / "10_experiments" / VER_OUT / "final_eval_s7" / "S7_probs_internal_val.csv"
THR_JS = PROJECT_ROOT / "10_experiments" / VER_OUT / "final_eval_s7" / "S7_thresholds_from_val.json"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ================== 小工具 ==================
def _find_y_col(df: pd.DataFrame) -> str:
    for cand in [Y_NAME_CFG, "y"]:
        if cand and cand in df.columns:
            return cand
    raise KeyError("概率文件找不到标签列（尝试 config.outcome.name 与 'y'）")

def _youden_thr(y, p):
    fpr, tpr, th = roc_curve(y, p)
    j = tpr - fpr
    return float(th[int(np.argmax(j))])

def _metrics(y, p, thr=0.5):
    y = np.asarray(y).astype(int); p = np.asarray(p, dtype=float)
    yhat = (p >= float(thr)).astype(int)
    tn = ((y == 0) & (yhat == 0)).sum(); fp = ((y == 0) & (yhat == 1)).sum()
    fn = ((y == 1) & (yhat == 0)).sum(); tp = ((y == 1) & (yhat == 1)).sum()
    spec = tn / max(tn + fp, 1); npv = tn / max(tn + fn, 1)
    return dict(
        Accuracy=accuracy_score(y, yhat),
        F1=f1_score(y, yhat, zero_division=0),
        NPV=npv,
        Precision=precision_score(y, yhat, zero_division=0),
        Sensitivity=recall_score(y, yhat, zero_division=0),
        Specificity=spec,
        AUROC=roc_auc_score(y, p),
        Brier=brier_score_loss(y, p),
    )

def _calib_curve(y, p, n_bins=10, min_per_bin=8):
    y = np.asarray(y).astype(int); p = np.asarray(p, dtype=float)
    qs = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    qs[0] = 0.0; qs[-1] = 1.0
    out = []
    for i in range(n_bins):
        m = (p >= qs[i]) & (p <= qs[i + 1])
        if m.sum() < min_per_bin:
            continue
        out.append((p[m].mean(), y[m].mean()))
    return np.array(out) if out else np.empty((0, 2))

def _dca(y, p, ts=np.linspace(0.01, 0.99, 99)):
    y = np.asarray(y).astype(int); p = np.asarray(p, dtype=float)
    N = len(y); out = []
    for t in ts:
        yhat = (p >= t).astype(int)
        TP = ((yhat == 1) & (y == 1)).sum()
        FP = ((yhat == 1) & (y == 0)).sum()
        nb = (TP / N) - (FP / N) * (t / (1 - t))
        out.append(nb)
    return ts, np.asarray(out, dtype=float)

# 名称与配色（固定顺序）
def _pretty_name(col: str) -> str:
    key = col.replace("p_", "").upper()
    mapping = {
        "LR": "LR", "RF": "RF", "EXTRA_TREES": "EXTRA_TREES", "GB": "GB",
        "ADABOOST": "ADABOOST", "LGB": "LGB", "XGB": "XGB", "CATBOOST": "CATBOOST"
    }
    return mapping.get(key, key)

TAB10 = plt.get_cmap("tab10")
ORDER = ["LR", "RF", "EXTRA_TREES", "GB", "ADABOOST", "LGB", "XGB"]
COLOR_MAP = {name: TAB10(i % 10) for i, name in enumerate(ORDER)}

def _load_thr_map():
    """优先读 json；若无则用 Validation 概率现算；再无则返回空 dict。key 为小写模型名。"""
    thr_map = {}
    if THR_JS.exists():
        try:
            data = json.loads(THR_JS.read_text(encoding="utf-8")) or {}
            for k, v in data.items():
                thr_map[str(k).lower()] = float(v)
            return thr_map
        except Exception:
            pass

    if P_VAL.exists():
        df = pd.read_csv(P_VAL)
        try:
            y_col = _find_y_col(df)
            yv = df[y_col].values.astype(int)
            for c in df.columns:
                if c.startswith("p_"):
                    mk = c.replace("p_", "").lower()
                    thr_map[mk] = _youden_thr(yv, df[c].values.astype(float))
        except Exception:
            pass
    return thr_map

# ================== 主作图 ==================
def _panel(prob, title, out_path: Path, csv_out: Path, thr_map: dict):
    # 读入与模型交集
    y_col_tr = _find_y_col(prob["train"])
    y_col_te = _find_y_col(prob["test"])
    y_tr = prob["train"][y_col_tr].values.astype(int)
    y_te = prob["test"][y_col_te].values.astype(int)

    cols_tr = [c for c in prob["train"].columns if c.startswith("p_")]
    cols_te = [c for c in prob["test"].columns if c.startswith("p_")]
    inter = sorted(set(cols_tr).intersection(cols_te))
    # 固定顺序（按 ORDER）
    model_cols = sorted(
        inter,
        key=lambda c: ORDER.index(_pretty_name(c)) if _pretty_name(c) in ORDER else 999
    )
    if not model_cols:
        raise RuntimeError("train/test 之间无共同模型概率列。")

    # 画布：2 行 4 列
    fig = plt.figure(figsize=(16, 9), dpi=300)
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], width_ratios=[1, 1, 1, 1])

    # A, B: ROC
    ax_roc_tr = fig.add_subplot(gs[0, 0])
    ax_roc_te = fig.add_subplot(gs[0, 1])
    for col in model_cols:
        name = _pretty_name(col); color = COLOR_MAP.get(name, None)
        # train
        p = prob["train"][col].values.astype(float)
        fpr, tpr, _ = roc_curve(y_tr, p)
        ax_roc_tr.plot(fpr, tpr, lw=2, color=color)
        # test
        p = prob["test"][col].values.astype(float)
        fpr, tpr, _ = roc_curve(y_te, p)
        ax_roc_te.plot(fpr, tpr, lw=2, color=color)

    for ax, ttl in [(ax_roc_tr, "ROC (Train)"), (ax_roc_te, "ROC (Test)")]:
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
        ax.set_xlabel("1 - Specificity"); ax.set_ylabel("Sensitivity")
        ax.set_title(ttl); ax.grid(ls="--", alpha=.3); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # C, D: Calibration
    ax_cal_tr = fig.add_subplot(gs[0, 2])
    ax_cal_te = fig.add_subplot(gs[0, 3])
    for col in model_cols:
        name = _pretty_name(col); color = COLOR_MAP.get(name, None)
        cur = _calib_curve(y_tr, prob["train"][col].values.astype(float))
        if cur.size: ax_cal_tr.plot(cur[:, 0], cur[:, 1], "-o", ms=3, lw=1.8, color=color)
        cur = _calib_curve(y_te, prob["test"][col].values.astype(float))
        if cur.size: ax_cal_te.plot(cur[:, 0], cur[:, 1], "-o", ms=3, lw=1.8, color=color)

    for ax, ttl in [(ax_cal_tr, "Calibration (Train)"), (ax_cal_te, "Calibration (Test)")]:
        ax.plot([0, 1], [0, 1], "--", c="gray", lw=1)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Observed")
        ax.set_title(ttl); ax.grid(ls="--", alpha=.3)

    # E, F: Metrics 折线（阈值优先来自 thr_map）
    pick = ["Accuracy", "F1", "NPV", "Precision", "Sensitivity", "Specificity"]
    ax_met_tr = fig.add_subplot(gs[1, 0])
    ax_met_te = fig.add_subplot(gs[1, 1])

    # 同步导出 CSV 的记录
    recs = []

    # 如能拿到 Validation 概率，用其兜底算阈值（优先度仅次于 JSON）
    val_df = pd.read_csv(P_VAL) if P_VAL.exists() else None
    if val_df is not None:
        try:
            y_val = val_df[_find_y_col(val_df)].values.astype(int)
        except Exception:
            y_val, val_df = None, None

    for col in model_cols:
        name = _pretty_name(col); color = COLOR_MAP.get(name, None)
        key_lc = col.replace("p_", "").lower()

        # —— 阈值选择顺序：JSON -> Validation(现算) -> 当前 split(兜底) —— #
        thr_fixed = thr_map.get(key_lc, None)
        if thr_fixed is None and val_df is not None and f"p_{key_lc}" in val_df.columns:
            try:
                thr_fixed = _youden_thr(y_val, val_df[f"p_{key_lc}"].values.astype(float))
            except Exception:
                thr_fixed = None

        # train
        p_tr = prob["train"][col].values.astype(float)
        thr_use_tr = thr_fixed if thr_fixed is not None else _youden_thr(y_tr, p_tr)
        met_tr = _metrics(y_tr, p_tr, thr_use_tr)
        ax_met_tr.plot(range(len(pick)), [met_tr[k] for k in pick], "-o", ms=3, lw=1.8, color=color)

        # test
        p_te = prob["test"][col].values.astype(float)
        thr_use_te = thr_fixed if thr_fixed is not None else _youden_thr(y_te, p_te)
        met_te = _metrics(y_te, p_te, thr_use_te)
        ax_met_te.plot(range(len(pick)), [met_te[k] for k in pick], "-o", ms=3, lw=1.8, color=color)

        # 记录
        recs.append({"Model": name, "Split": "Train", "Threshold": thr_use_tr, **met_tr})
        recs.append({"Model": name, "Split": "Test",  "Threshold": thr_use_te, **met_te})

    for ax, ttl in [(ax_met_tr, "Metrics (Train)"), (ax_met_te, "Metrics (Test)")]:
        ax.set_xticks(range(len(pick))); ax.set_xticklabels(pick, rotation=20)
        ax.set_ylim(0, 1.0); ax.set_title(ttl); ax.grid(ls="--", alpha=.3)

    # G, H: DCA
    ax_dca_tr = fig.add_subplot(gs[1, 2])
    ax_dca_te = fig.add_subplot(gs[1, 3])
    for (ax, y, key, ttl) in [(ax_dca_tr, y_tr, "train", "DCA (Train)"),
                              (ax_dca_te, y_te, "test",  "DCA (Test)")]:
        ts = np.linspace(0.01, 0.99, 99)
        prev = float(np.mean(y))
        ax.plot(ts, np.zeros_like(ts), "--", c="gray", lw=1, label="Treat none")
        ax.plot(ts, prev - (1 - prev) * ts / (1 - ts), ":", c="black", lw=1, label="Treat all")
        for col in model_cols:
            name = _pretty_name(col); color = COLOR_MAP.get(name, None)
            _, nb = _dca(y, prob[key][col].values.astype(float), ts)
            ax.plot(ts, nb, lw=1.8, color=color)
        ax.set_xlabel("Threshold"); ax.set_ylabel("Net benefit")
        ax.set_title(ttl); ax.grid(ls="--", alpha=.3)

    # —— 底部共享图例 —— #
    handles, labels = [], []
    for col in model_cols:
        name = _pretty_name(col)
        if name not in labels:
            handles.append(Line2D([0], [0], color=COLOR_MAP.get(name, "gray"), lw=3))
            labels.append(name)

    fig.legend(handles, labels, loc="lower center",
               ncol=min(4, len(labels)), frameon=False, columnspacing=1.2,
               handlelength=2.8, fontsize=11)

    fig.suptitle(title, y=0.98, fontsize=16)
    fig.tight_layout(rect=[0.03, 0.10, 0.97, 0.95])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print("[save]", out_path)

    # 导出度量 CSV
    df_out = pd.DataFrame(recs)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(csv_out, index=False, encoding="utf-8-sig")
    print("[save]", csv_out)

def main():
    if not P_TR.exists() or not P_TE.exists():
        raise FileNotFoundError(f"需要概率文件：\n- {P_TR}\n- {P_TE}")
    prob = {"train": pd.read_csv(P_TR), "test": pd.read_csv(P_TE)}
    # 验证 y 列
    _ = _find_y_col(prob["train"]); _ = _find_y_col(prob["test"])

    thr_map = _load_thr_map()
    title = "Models performance — ROC / Calibration / Metrics / DCA (fixed thresholds if available)"
    _panel(prob, title,
           OUT_DIR / "Fig4_multi_panels_bottom_legend.png",
           OUT_DIR / "Fig4_multi_panels_metrics.csv",
           thr_map)

if __name__ == "__main__":
    main()
