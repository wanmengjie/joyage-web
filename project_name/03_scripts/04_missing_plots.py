# -*- coding: utf-8 -*-
"""
04_missing_plots.py — 缺失可视化（Figure S2 风格：Morandi 静谧绿灰，A/B 面板 + Top-K 条形图 + 可选热图）
运行：python 04_missing_plots.py

读入：02_processed_data/<VER>/splits/{charls_train.csv, charls_val.csv, charls_test.csv}
配置：
  ids: ["ID","householdID"]               # 需要排除的 ID 列
  outcome: {source: "...", name: "..."}   # 同样排除
  preprocess.missing_plots:
      top_k: 30
      max_rows: 5000
      xtick_step: 1
      angle: 60
      draw_heatmap: true
产物：
  02_processed_data/<VER>/diagnostics/
    ├─ missing_rate_train.csv / val.csv / test.csv
    ├─ missing_bars_train.png / val.png / test.png
    ├─ missing_heatmap_train.png / val.png / test.png   # 若启用
    └─ FigS2_missing_value_overview.png                 # A/B 面板（train vs test；若无 test 则仅 A=train）
"""

from __future__ import annotations
import yaml, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from pathlib import Path

# ---------------- 项目与配置 ----------------
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

CFG = yaml.safe_load(open(repo_root() / "07_config" / "config.yaml", "r", encoding="utf-8"))

PROJECT_ROOT = Path(CFG["paths"]["project_root"])
VER_IN  = CFG.get("run_id_in", CFG.get("run_id"))
RANDOM_STATE = int(CFG.get("random_state", CFG.get("seed", 42)))

splits_dir = PROJECT_ROOT / "02_processed_data" / VER_IN / "splits"
diag_dir   = PROJECT_ROOT / "02_processed_data" / VER_IN / "diagnostics"
diag_dir.mkdir(parents=True, exist_ok=True)

LABEL = (CFG.get("outcome", {}) or {}).get("name")
ID_COLS = CFG.get("ids", ["ID", "householdID"])

mp_cfg = (CFG.get("preprocess", {}) or {}).get("missing_plots", {}) or {}
TOP_K = int(mp_cfg.get("top_k", 30))
MAX_ROWS = int(mp_cfg.get("max_rows", 5000))
ANGLE = int(mp_cfg.get("angle", 60))
DRAW_HEAT = bool(mp_cfg.get("draw_heatmap", True))

# ---------------- 读取并过滤列（排除 id / 标签） ----------------
def load_split(name: str) -> pd.DataFrame | None:
    p = splits_dir / f"charls_{name}.csv"
    if not p.exists(): return None
    df = pd.read_csv(p)
    drop_cols = set(ID_COLS)
    src = (CFG.get("outcome", {}) or {}).get("source")
    if src and src in df.columns: drop_cols.add(src)
    if LABEL and LABEL in df.columns: drop_cols.add(LABEL)
    keep = [c for c in df.columns if c not in drop_cols]
    return df[keep].copy()

df_train = load_split("train")
df_val   = load_split("val")
df_test  = load_split("test")

# ---------------- 配色：Morandi 静谧绿灰 ----------------
def get_missing_cmap():
    present, missing = "#EDE9E4", "#5B6D5B"  # 浅米灰/墨绿
    return ListedColormap([present, missing])

# ---------------- 绘制函数 ----------------
def plot_missing_matrix_panel(ax, df: pd.DataFrame, panel_tag: str):
    df2 = df.sample(n=MAX_ROWS, random_state=RANDOM_STATE) if len(df) > MAX_ROWS else df
    cols = df2.columns.tolist()
    mask = df2[cols].isna().astype(np.uint8).values
    cmap = get_missing_cmap()
    present_color, missing_color = cmap.colors[0], cmap.colors[1]

    im = ax.imshow(mask, aspect="auto", interpolation="nearest", cmap=cmap)
    ax.set_ylabel("Observations"); ax.set_xlabel("")
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", which="both", labelsize=8)
    ax.tick_params(axis="y", which="both", labelsize=8)

    step_auto = max(1, len(cols)//22)
    step = max(1, int(mp_cfg.get("xtick_step", step_auto)))
    ticks = list(range(0, len(cols), step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([cols[i] for i in ticks], rotation=ANGLE, ha="left")

    for spine in ax.spines.values(): spine.set_visible(False)

    total = mask.size; miss = int(mask.sum()); pres = total - miss
    miss_pct = miss/total*100.0; pres_pct = pres/total*100.0
    handles = [
        Patch(facecolor=missing_color, edgecolor="none", label=f"Missing ({miss_pct:.2f}%)"),
        Patch(facecolor=present_color, edgecolor="none", label=f"Present ({pres_pct:.2f}%)"),
    ]
    ax.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=8,
              bbox_to_anchor=(0.5, -0.06))
    ax.text(-0.02, 1.06, panel_tag, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="bottom", ha="left")

def save_missing_rate_and_bars(df: pd.DataFrame, split: str):
    cols = df.columns.tolist()
    miss = df[cols].isna().mean().sort_values(ascending=False)
    miss.to_csv(diag_dir/f"missing_rate_{split}.csv", header=["MissingRate"])

    top = miss.head(TOP_K)
    plt.figure(figsize=(max(8, 0.24*len(top)), 5), dpi=300)
    ax = plt.gca()
    ax.bar(range(len(top)), top.values)
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(top.index, rotation=ANGLE, ha="right", fontsize=8)
    ax.set_ylabel("Missing rate")
    ax.set_title(f"Top-{len(top)} Missing rate — {split}")
    plt.tight_layout()
    plt.savefig(diag_dir/f"missing_bars_{split}.png"); plt.close()

def save_heatmap(df: pd.DataFrame, split: str):
    # 简单的“是否缺失”的变量间相关性（phi-like），用于聚类热图（不使用 seaborn）
    mask = df.isna().astype(int)
    if mask.shape[1] < 2: return
    cor = np.corrcoef(mask.T)  # 列间相关
    plt.figure(figsize=(7, 6), dpi=300)
    plt.imshow(cor, aspect="auto", interpolation="nearest", cmap="viridis")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(f"Missingness correlation (variables) — {split}")
    plt.xticks([]); plt.yticks([])  # 避免太拥挤；如需标签可自行开启
    plt.tight_layout()
    plt.savefig(diag_dir/f"missing_heatmap_{split}.png"); plt.close()

# ---------------- 逐 split 导出诊断 ----------------
any_for_panel = False
for sp, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
    if df is None: continue
    any_for_panel = True
    save_missing_rate_and_bars(df, sp)
    if DRAW_HEAT: save_heatmap(df, sp)

# ---------------- A/B 面板（train vs test；若无 test 则只画 train） ----------------
if df_train is not None and df_test is not None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), dpi=300, gridspec_kw={"hspace": 0.35})
    plot_missing_matrix_panel(axes[0], df_train, "A")
    plot_missing_matrix_panel(axes[1], df_test,  "B")
    axes[1].set_xlabel("")
    fig.suptitle("Figure S2. Variable missing value overview", y=0.995, fontsize=12)
    plt.savefig(diag_dir / "FigS2_missing_value_overview.png", bbox_inches="tight"); plt.close()
elif df_train is not None:
    fig, ax = plt.subplots(1, 1, figsize=(12, 6), dpi=300)
    plot_missing_matrix_panel(ax, df_train, "A")
    fig.suptitle("Figure S2. Variable missing value overview (train only)", y=0.995, fontsize=12)
    plt.savefig(diag_dir / "FigS2_missing_value_overview.png", bbox_inches="tight"); plt.close()

print(f"[ok] outputs -> {diag_dir}")
