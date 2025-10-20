# -*- coding: utf-8 -*-
r"""
20_fig2_rcs.py — Restricted Cubic Spline (natural spline) on CHARLS continuous variables

- 训练集拟合：GLM(Binomial, logit)
- P_overall：样条 vs 截距-only（LR 检验）
- P_nonlin ：样条 vs 线性（LR 检验）
- 对自变量做分位裁剪以稳健绘图；每变量导出曲线 CSV
- 输出到：10_experiments/<VER_OUT>/figs/
"""

import os; os.environ["MPLBACKEND"] = "Agg"
import matplotlib; matplotlib.use("Agg")

from pathlib import Path
from typing import List, Optional, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from patsy import dmatrix
from scipy.stats import chi2
import yaml

# ---------------- 配置与路径（读取 config.yaml） ----------------
def repo_root() -> Path:
    # 脚本位于 03_scripts/，项目根在上两级
    return Path(__file__).resolve().parents[1]

CFG = yaml.safe_load(open(repo_root() / "07_config" / "config.yaml", "r", encoding="utf-8"))

PROJECT_ROOT = Path(CFG["paths"]["project_root"])
VER_IN  = CFG.get("run_id_in",  CFG.get("run_id"))
VER_OUT = CFG.get("run_id_out", CFG.get("run_id"))
Y_NAME  = (CFG.get("outcome", {}) or {}).get("name", "depression_bin")

MEAN_MODE_XY = PROJECT_ROOT / "02_processed_data" / VER_IN  / "frozen" / "charls_mean_mode_Xy.csv"
TRAIN_IDX    = PROJECT_ROOT / "02_processed_data" / VER_IN  / "splits" / "charls_train_idx.csv"

# ---------------- 连续变量（若无效将自动挑选） ----------------
TARGETS: List[str] = [
    "agey","child","hhres",
    "comparable_hexp","comparable_exp","comparable_frec",
    "comparable_itearn","comparable_ipubpen","comparable_tgiv",
]

VAR_LABELS: Dict[str,str] = {
    "agey": "Age (years)",
    "child": "Children in household",
    "hhres": "Household residents",
    "comparable_hexp": "Health exp. (comparable)",
    "comparable_exp": "Total exp. (comparable)",
    "comparable_frec": "Food exp. (comparable)",
    "comparable_itearn": "Labor income (comparable)",
    "comparable_ipubpen": "Public pension (comparable)",
    "comparable_tgiv": "Transfer-in (comparable)",
}

# ---------------- RCS / 选择参数 ----------------
DF_SPLINE = 4           # 样条自由度（常用 3~5）
N_GRID    = 200         # 曲线评估点
AUTO_TOPK = 6           # 自动挑选数量（当 TARGETS 不可用时）
CLIP_Q    = (1, 99)     # 分位裁剪百分位
MIN_N     = 30          # 变量有效样本下限
MIN_UNIQ  = 6           # 变量有效唯一下限

# ---------------- IO & 小工具 ----------------
def _safe_read_idx(p: Path) -> Optional[np.ndarray]:
    if p.exists():
        try:
            return pd.read_csv(p).iloc[:, 0].astype(int).values
        except Exception:
            pass
    return None

def _coerce_numeric(s: pd.Series) -> pd.Series:
    if np.issubdtype(s.dtype, np.number):
        return s
    return pd.to_numeric(s, errors="coerce")

def resolve_targets(df: pd.DataFrame, wants: List[str]) -> List[str]:
    keeps = []
    for v in wants:
        if v in df.columns:
            x = _coerce_numeric(df[v])
            if x.notna().sum() >= MIN_N and x.dropna().nunique() >= MIN_UNIQ:
                keeps.append(v)
    return keeps

def pick_auto_continuous(df_tr: pd.DataFrame, y: np.ndarray, exclude: List[str], k: int) -> List[str]:
    feats = [c for c in df_tr.columns if c not in exclude]
    scores = []
    for c in feats:
        s = _coerce_numeric(df_tr[c])
        if s.notna().sum() < MIN_N or s.dropna().nunique() < MIN_UNIQ:
            continue
        try:
            corr = pd.Series(s).corr(pd.Series(y), method="spearman")
        except Exception:
            corr = 0.0
        var = np.nanvar(s.values)
        scores.append((c, abs(corr) + 0.01*np.log1p(var)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in scores[:k]]

# ---------------- 统计函数 ----------------
def _lr_pvalue(llf_big: float, llf_small: float, df_diff: int) -> float:
    lr = 2.0 * (llf_big - llf_small)
    df_diff = max(int(df_diff), 1)
    try:
        return float(1 - chi2.cdf(lr, df=df_diff))
    except Exception:
        return np.nan

def fit_rcs_glm(y: np.ndarray, x: np.ndarray, df: int = DF_SPLINE) -> Tuple[Optional[sm.GLM], float, float]:
    """
    返回：样条模型 m_spline，以及
    p_overall: 样条 vs 截距-only（LR检验）
    p_nonlin: 样条 vs 线性（LR检验）
    任一模型拟合异常则抛回 None, nan, nan
    """
    try:
        # 样条模型
        Xs = dmatrix(f"cr(x, df={df})", {"x": x}, return_type="dataframe")
        m_spline = sm.GLM(y, sm.add_constant(Xs, has_constant='add'), family=sm.families.Binomial()).fit()

        # 线性模型
        X_lin = sm.add_constant(pd.DataFrame({"x": x}), has_constant='add')
        m_lin  = sm.GLM(y, X_lin, family=sm.families.Binomial()).fit()

        # 截距-only
        X_null = np.ones((len(y), 1))
        m_null = sm.GLM(y, X_null, family=sm.families.Binomial()).fit()

        k_spline = int(m_spline.params.size)
        k_lin    = int(m_lin.params.size)
        k_null   = int(m_null.params.size)

        p_overall = _lr_pvalue(m_spline.llf, m_null.llf, k_spline - k_null)
        p_nonlin  = _lr_pvalue(m_spline.llf, m_lin.llf,  k_spline - k_lin)

        return m_spline, p_overall, p_nonlin
    except Exception:
        return None, float('nan'), float('nan')

def pred_curve(model: sm.GLM, x_grid: np.ndarray, df: int) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    Xg = dmatrix(f"cr(x, df={df})", {"x": x_grid}, return_type="dataframe")
    pr = model.get_prediction(sm.add_constant(Xg, has_constant='add')).summary_frame()
    return pr["mean"].values, pr.get("mean_ci_lower", pr["mean_ci_lower"]).values, pr.get("mean_ci_upper", pr["mean_ci_upper"]).values

# ---------------- 主流程 ----------------
def main():
    # 读数据
    if not MEAN_MODE_XY.exists():
        raise FileNotFoundError(f"找不到数据：{MEAN_MODE_XY}")
    df_all = pd.read_csv(MEAN_MODE_XY)
    if Y_NAME not in df_all.columns:
        raise KeyError(f"{Y_NAME} 不在 {MEAN_MODE_XY}")

    # 训练索引（缺失则全体）
    tr_idx = _safe_read_idx(TRAIN_IDX)
    if tr_idx is None:
        print(f"[warn] 训练索引缺失：{TRAIN_IDX}，将使用全体样本作为训练集。")
        tr_idx = np.arange(len(df_all))

    y_all = pd.to_numeric(df_all[Y_NAME], errors="coerce").astype("Int64").values
    df = df_all.iloc[tr_idx].reset_index(drop=True)
    y  = y_all[tr_idx].astype(float)  # 后面会做掩码

    # 变量选择：先用预设 TARGETS，若无效则自动挑选
    want = resolve_targets(df, TARGETS)
    if not want:
        auto = pick_auto_continuous(df, pd.Series(y).fillna(0).astype(int).values, exclude=[Y_NAME], k=AUTO_TOPK)
        want = auto
        print(f"[info] 指定变量均不可用，自动挑选: {want}")

    if not want:
        print("[error] 没有可用连续变量，退出。")
        return

    # 输出目录
    FIG_DIR = PROJECT_ROOT / "10_experiments" / VER_OUT / "figs"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # 画格子
    n_plot = len(want)
    ncols = 3
    nrows = int(np.ceil(n_plot / ncols))
    fig = plt.figure(figsize=(5 * ncols, 3.6 * max(nrows, 1)), dpi=240)

    plotted = 0
    for i, v in enumerate(want, 1):
        s_raw = _coerce_numeric(df[v])

        # 与 y 联合去缺失
        mask = s_raw.notna() & pd.notna(y)
        x = s_raw[mask].astype(float).values
        yy = pd.Series(y[mask]).astype(int).values

        if np.unique(x).size < MIN_UNIQ or len(x) < MIN_N:
            print(f"[warn] {v} 有效取值不足（n={len(x)}, uniq={np.unique(x).size}），跳过")
            continue

        # 分位裁剪，防止极端值影响
        lo_q, hi_q = np.nanpercentile(x, CLIP_Q[0]), np.nanpercentile(x, CLIP_Q[1])
        x_clip = np.clip(x, lo_q, hi_q)

        m_spline, p_all, p_non = fit_rcs_glm(yy, x_clip, df=DF_SPLINE)
        if m_spline is None:
            print(f"[warn] {v} 样条拟合失败，跳过")
            continue

        xs = np.linspace(lo_q, hi_q, N_GRID)
        mean, lo, hi = pred_curve(m_spline, xs, DF_SPLINE)

        # 单变量 CSV（便于复核/论文存档）
        csv_path = FIG_DIR / f"Fig2_RCS_{v}.csv"
        pd.DataFrame({
            v: xs, "pred": mean, "ci_lo": lo, "ci_hi": hi,
            "P_overall": [p_all]*len(xs), "P_nonlin": [p_non]*len(xs)
        }).to_csv(csv_path, index=False, encoding="utf-8-sig")

        # 绘图
        ax = fig.add_subplot(nrows, ncols, i)
        ax.plot(xs, mean, lw=2)
        ax.fill_between(xs, lo, hi, alpha=0.20)
        ax.set_ylim(0, 1)
        ax.set_xlabel(VAR_LABELS.get(v, v))
        ax.set_ylabel("Pr(Y=1)")
        ax.set_title(f"{VAR_LABELS.get(v, v)}  (P_overall={p_all:.3g}; P_nonlin={p_non:.3g})", fontsize=9)
        ax.grid(ls="--", alpha=.35)
        plotted += 1

    if plotted == 0:
        print("[error] 没有成功绘制的变量，未生成总图。")
        return

    fig.suptitle("Restricted cubic spline — Training set", y=1.02, fontsize=12)
    out_png = FIG_DIR / "Fig2_RCS_train.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_png.with_suffix(".pdf"))
    print("[save]", out_png)
    print("[done] RCS figure & per-variable CSVs ready.")

if __name__ == "__main__":
    main()
