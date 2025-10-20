# -*- coding: utf-8 -*-
r"""
13_sensitivity_panels.py — S6 风格的敏感性面板（多指标 × 多数据集 × 多模型）

要点：
- 统一从 07_config/config.yaml 读取 project_root / run_id_out / eval.models
- 直接读取 S4 产物：10_experiments/<VER_OUT>/feature_sweep/S4_RFE_<model>_summary.csv
- 仅锁定指定 method（默认 mean_mode），缺表/缺列自动跳过
- 输出：
  1) 10_experiments/<VER_OUT>/feature_sweep/S6_panel_data.csv
  2) 10_experiments/<VER_OUT>/feature_sweep/plots/S6_sensitivity_panels.[png|pdf]
"""

import os
os.environ["MPLBACKEND"] = "Agg"  # 非交互环境也能出图

from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

# ================== 统一配置读取 ==================
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

CFG = yaml.safe_load(open(repo_root() / "07_config" / "config.yaml", "r", encoding="utf-8"))

PROJECT_ROOT = Path(CFG["paths"]["project_root"])
VER_OUT      = CFG.get("run_id_out", CFG.get("run_id"))
# method 可放到 cfg["imputation"]["target_method"]，未配置则默认 mean_mode
METHOD       = (CFG.get("imputation", {}) or {}).get("target_method", "mean_mode")

# 模型集合：优先 cfg.eval.models，否则用默认
_models_cfg  = list((CFG.get("eval", {}) or {}).get("models", []))
if not _models_cfg:
    _models_cfg = ["lr", "rf", "extra_trees", "gb", "adaboost", "lgb", "xgb"]
MODEL_ORDER = [m for m in _models_cfg if str(m).lower() != "catboost"]

IN_DIR  = PROJECT_ROOT / "10_experiments" / VER_OUT / "feature_sweep"
OUT_DIR = IN_DIR  # S6 输出与 S4 同目录层
(OUT_DIR / "plots").mkdir(parents=True, exist_ok=True)

# 目标九个指标（列名别名兼容）
METRIC_ALIAS: Dict[str, str] = {
    "AUROC": "AUROC",
    "AUPRC": "AUPRC",
    "Accuracy": "Accuracy",
    "Precision": "Precision",
    "Recall": "Recall",             # = Sensitivity
    "Sensitivity": "Recall",
    "F1_Score": "F1_Score",
    "Brier": "Brier_Score",
    "Brier_Score": "Brier_Score",
    "Specificity": "Specificity",
    "NPV": "NPV",
}
# 面板显示顺序（存在即画）
PANEL_ORDER = ["Accuracy", "AUROC", "Brier_Score",
               "F1_Score", "NPV", "Precision",
               "Recall", "Specificity", "AUPRC"]

DISPLAY_NAME = {
    "lr": "LR", "rf": "RF", "extra_trees": "EXTRA_TREES", "gb": "GB",
    "adaboost": "ADABOOST", "lgb": "LGB", "xgb": "XGB", "catboost": "CATBOOST"
}
SCEN_DISPLAY = {
    "internal_train": "Train",
    "internal_val": "Val",
    "internal_test": "Test",
    "external_main": "External (main)",
    "external_transfer": "External (transfer)",
}
# =================================================


def _read_summary_for_model(model_key: str) -> pd.DataFrame:
    """读取 S4 的 summary：S4_RFE_<model>_summary.csv，并做 method/列名标准化。"""
    p = IN_DIR / f"S4_RFE_{model_key}_summary.csv"
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(p)
    except Exception:
        return pd.DataFrame()

    # method / model 兜底
    if "method" not in df.columns:
        df.insert(0, "method", METHOD)
    if "model" not in df.columns:
        df.insert(1, "model", model_key)
    df["model"] = df["model"].astype(str).str.lower()

    # 只保留目标 method
    if "method" in df.columns:
        df = df[df["method"].astype(str).str.lower() == METHOD.lower()].copy()
    else:
        # 没有 method 列则默认视作目标 method
        df["method"] = METHOD
    if df.empty:
        return df

    # 指标列标准化
    col_map = {c: METRIC_ALIAS[c] for c in df.columns if c in METRIC_ALIAS}
    df = df.rename(columns=col_map)

    # 必要列存在性检查
    need_cols = {"method", "model", "scenario"}
    if not need_cols.issubset(set(df.columns)):
        return pd.DataFrame()

    # 丢掉 CI 列（*_lo, *_hi）
    df = df[[c for c in df.columns if not (str(c).endswith("_lo") or str(c).endswith("_hi"))]]
    return df


def load_all_from_s4() -> pd.DataFrame:
    """整合所有模型的 S4 summary 表"""
    parts: List[pd.DataFrame] = []
    for mk in MODEL_ORDER:
        sub = _read_summary_for_model(mk)
        if not sub.empty:
            parts.append(sub)

    if not parts:
        raise FileNotFoundError(
            f"未发现任何 S4_RFE_<model>_summary.csv 于：{IN_DIR}\n"
            f"请先运行 10_feature_count_sweep.py 生成 S4 产物。"
        )

    df = pd.concat(parts, ignore_index=True)

    # 仅保留我们关心的模型 & 顺序
    df["model"] = df["model"].astype(str).str.lower()
    df = df[df["model"].isin(MODEL_ORDER)].copy()
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)

    # 场景顺序（按存在的来）
    scen_order_all = ["internal_train","internal_val","internal_test","external_main","external_transfer"]
    scen_have = [s for s in scen_order_all if s in df["scenario"].astype(str).unique().tolist()]
    df["scenario"] = pd.Categorical(df["scenario"], categories=scen_have, ordered=True)

    # 只留 9 指标 + 必要列
    metric_cols = [m for m in set(METRIC_ALIAS.values()) if m in df.columns]
    keep = ["method", "model", "scenario"] + metric_cols
    df = df[keep].sort_values(["model","scenario"]).reset_index(drop=True)
    return df


def longify(df: pd.DataFrame) -> pd.DataFrame:
    """宽表 -> 长表"""
    metric_cols = [m for m in PANEL_ORDER if m in df.columns]
    long = df.melt(
        id_vars=["method", "model", "scenario"],
        value_vars=metric_cols,
        var_name="metric", value_name="value"
    )
    # 分类顺序
    if not isinstance(long["model"].dtype, pd.CategoricalDtype):
        long["model"] = pd.Categorical(long["model"], categories=MODEL_ORDER, ordered=True)
    long["scenario"] = long["scenario"].cat.remove_unused_categories()
    long["metric"] = pd.Categorical(long["metric"],
                                    categories=[m for m in PANEL_ORDER if m in metric_cols],
                                    ordered=True)
    return long


def plot_panels(long_df: pd.DataFrame, out_dir: Path):
    # 可用指标
    metrics = [m for m in PANEL_ORDER if m in long_df["metric"].unique()]
    if not metrics:
        print("[warn] 没有可用指标列，跳过绘图")
        return

    # 计算行列（最多 3×3；若不足则隐藏多余子图）
    n_panels = len(metrics)
    n_rows, n_cols = 3, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 12))
    axes = axes.flatten()

    # 颜色：按场景
    scen_list = list(long_df["scenario"].cat.categories)
    cmap = plt.get_cmap("tab10")
    colors = {scen: cmap(i % 10) for i, scen in enumerate(scen_list)}

    for i, metric in enumerate(metrics):
        ax = axes[i]
        sub = long_df[long_df["metric"] == metric].copy()
        # pivot: index=model, columns=scenario
        tab = sub.pivot_table(index="model", columns="scenario", values="value", aggfunc="mean")
        tab = tab.reindex(index=[m for m in MODEL_ORDER if m in tab.index], columns=scen_list)

        x = np.arange(len(tab.index))
        w = 0.8 / max(1, len(scen_list))

        for j, scen in enumerate(scen_list):
            y = tab[scen].values if scen in tab.columns else np.zeros(len(tab.index))
            ax.bar(x + j*w - 0.4 + w/2, y, width=w, label=SCEN_DISPLAY.get(scen, scen), color=colors[scen])

        ax.set_title(metric.replace("_", " "))
        ax.set_xticks(x)
        ax.set_xticklabels([DISPLAY_NAME.get(m, str(m).upper()) for m in tab.index], rotation=0)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        # y 轴范围：Brier(越小越好)通常在 [0,0.25]；其他在 [0,1]
        if metric == "Brier_Score":
            ymax = np.nanmax(tab.values) if tab.size else 0.0
            ymax = 0.0 if not np.isfinite(ymax) else float(ymax)
            ax.set_ylim(0, max(0.25, ymax * 1.1 if ymax > 0 else 0.25))
        else:
            ax.set_ylim(0, 1.0)

        if i == 0:
            ax.legend(loc="best", fontsize=9)

    # 多出来的空子图隐藏
    for k in range(n_panels, n_rows*n_cols):
        axes[k].axis("off")

    fig.suptitle("S6 — Sensitivity analysis (method: {}, models × datasets × metrics)".format(METHOD),
                 fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    png = out_dir / "plots" / "S6_sensitivity_panels.png"
    pdf = out_dir / "plots" / "S6_sensitivity_panels.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"[save] {png}")
    print(f"[save] {pdf}")


def main():
    if not IN_DIR.exists():
        raise FileNotFoundError(f"S4 目录不存在：{IN_DIR}")

    df = load_all_from_s4()
    long_df = longify(df)

    # 保存整合数据
    out_csv = OUT_DIR / "S6_panel_data.csv"
    long_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[save] {out_csv}")

    # 作图
    plot_panels(long_df, OUT_DIR)
    print("[done] S6-style sensitivity panels finished.")


if __name__ == "__main__":
    main()
