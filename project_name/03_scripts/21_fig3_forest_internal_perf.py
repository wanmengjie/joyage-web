# -*- coding: utf-8 -*-
# 21_fig3_forest_internal_perf.py — Forest-like summary of AUC/Spec/Sens with bootstrap CI
# 使用 Validation 的 Youden 阈值评估 Internal Test；若 Test 不存在则回退到 Validation

import os
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import yaml

# ================= 配置：从 config.yaml 读取 =================
def repo_root() -> Path:
    # 脚本位于 03_scripts/，项目根在上两级
    return Path(__file__).resolve().parents[1]

CFG = yaml.safe_load(open(repo_root() / "07_config" / "config.yaml", "r", encoding="utf-8"))
PROJECT_ROOT = Path(CFG["paths"]["project_root"])
VER_OUT = CFG.get("run_id_out", CFG.get("run_id"))
Y_NAME_CFG = (CFG.get("outcome", {}) or {}).get("name", None)

PROBS_DIR = PROJECT_ROOT / "10_experiments" / VER_OUT / "final_eval_s7"
PROBS_TEST = PROBS_DIR / "S7_probs_internal_test.csv"
PROBS_VAL  = PROBS_DIR / "S7_probs_internal_val.csv"
THR_JSON   = PROBS_DIR / "S7_thresholds_from_val.json"

OUT_DIR    = PROJECT_ROOT / "10_experiments" / VER_OUT / "figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG    = OUT_DIR / "Fig3_forest_internal.png"
OUT_CSV    = OUT_DIR / "Fig3_forest_internal.csv"

# Bootstrap 参数
N_BOOT = 1000
RANDOM_STATE = 42
# ===========================================================

def _pick_probs_and_title():
    """优先 internal_test；否则回退 internal_val。返回 (df, title, scen_key)"""
    if PROBS_TEST.exists():
        return pd.read_csv(PROBS_TEST), \
               "Internal Test — ROC / Specificity / Sensitivity (bootstrap; thresholds from Validation)", \
               "internal_test"
    if PROBS_VAL.exists():
        return pd.read_csv(PROBS_VAL), \
               "Internal Validation — ROC / Specificity / Sensitivity (bootstrap; thresholds from Validation)", \
               "internal_val"
    raise FileNotFoundError(f"缺少概率文件：\n- {PROBS_TEST}\n- {PROBS_VAL}")

def _find_y_col(df: pd.DataFrame) -> str:
    for cand in [Y_NAME_CFG, "y"]:
        if cand and cand in df.columns:
            return cand
    raise KeyError(f"概率文件中找不到标签列（尝试了 {Y_NAME_CFG!r} 与 'y'）")

def _load_thr_map(prob_df_val: pd.DataFrame | None) -> dict:
    """
    读取 Validation 阈值；若无 json，则用 Validation 概率即时算 Youden。
    返回 dict：{model_key(lower): thr}
    """
    if THR_JSON.exists():
        try:
            data = json.loads(THR_JSON.read_text(encoding="utf-8")) or {}
            return {str(k).lower(): float(v) for k, v in data.items()}
        except Exception:
            pass

    # 没有 json，用 Validation 现算
    if prob_df_val is None and PROBS_VAL.exists():
        prob_df_val = pd.read_csv(PROBS_VAL)

    thr_map = {}
    if prob_df_val is not None:
        y_col = _find_y_col(prob_df_val)
        yv = prob_df_val[y_col].values.astype(int)
        for c in prob_df_val.columns:
            if not c.startswith("p_"):
                continue
            key = c.replace("p_", "").lower()
            try:
                fpr, tpr, th = roc_curve(yv, prob_df_val[c].values.astype(float))
                j = tpr - fpr
                k = int(np.argmax(j))
                thr_map[key] = float(th[k])
            except Exception:
                # 留空：后面会在当前数据上兜底
                pass
    return thr_map

def _youden_thr(y, p):
    fpr, tpr, th = roc_curve(y, p)
    j = tpr - fpr
    k = int(np.argmax(j))
    return float(th[k])

def _spec_sens(y, p, thr):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= float(thr)).astype(int)
    tn = np.sum((y == 0) & (pred == 0))
    fp = np.sum((y == 0) & (pred == 1))
    fn = np.sum((y == 1) & (pred == 0))
    tp = np.sum((y == 1) & (pred == 1))
    spec = tn / max(tn + fp, 1)
    sens = tp / max(tp + fn, 1)
    return spec, sens

def _boot_metrics(y, p, thr_fixed):
    """
    固定阈值 thr_fixed，bootstrap N_BOOT 次，返回 AUC/Spec/Sens 的 mean、sd、95%CI。
    若某次自举样本只有单一类别，则跳过该次（AUC 不可计算）。
    """
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(y)
    idx = np.arange(n)

    aucs, specs, senss = [], [], []
    for _ in range(N_BOOT):
        b = rng.choice(idx, size=n, replace=True)
        yb, pb = y[b], p[b]
        if yb.min() == yb.max():
            continue
        try:
            aucs.append(roc_auc_score(yb, pb))
        except Exception:
            continue
        s0, s1 = _spec_sens(yb, pb, thr_fixed)
        specs.append(s0); senss.append(s1)

    def _summ(a):
        a = np.asarray(a, dtype=float)
        a = a[~np.isnan(a)]
        if a.size == 0:
            return dict(mean=np.nan, sd=np.nan, lo=np.nan, hi=np.nan)
        return dict(mean=float(np.mean(a)),
                    sd=float(np.std(a, ddof=1)),
                    lo=float(np.quantile(a, 0.025)),
                    hi=float(np.quantile(a, 0.975)))
    return _summ(aucs), _summ(specs), _summ(senss)

def _fmt(stat):
    vals = [stat.get("mean", np.nan), stat.get("sd", np.nan),
            stat.get("lo", np.nan), stat.get("hi", np.nan)]
    if any(np.isnan(vals)):
        return "NA"
    return f"{stat['mean']:.3f}±{stat['sd']:.3f}\n95% CI [{stat['lo']:.3f}, {stat['hi']:.3f}]"

def _forest_plot(rows, title, out_png, out_csv):
    """
    rows: list of dict:
        {"model": "xgb",
         "AUC": {...}, "Spec": {...}, "Sens": {...}, "threshold": float}
    """
    # 按 AUC 均值降序
    order = np.argsort([- (r["AUC"]["mean"] if np.isfinite(r["AUC"]["mean"]) else -1) for r in rows])
    rows  = [rows[i] for i in order]
    models = [r["model"].upper() for r in rows]

    panels = [("AUC",  "AUC"),
              ("Spec", "Specificity"),
              ("Sens", "Sensitivity")]

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 5.0), dpi=240, sharey=True)
    y = np.arange(len(models))

    for ax, (key, xlabel) in zip(axes, panels):
        stats = [r[key] for r in rows]
        m  = np.array([s["mean"] for s in stats], dtype=float)
        lo = np.array([s["lo"]   for s in stats], dtype=float)
        hi = np.array([s["hi"]   for s in stats], dtype=float)

        # 误差线（水平）
        xerr = np.vstack([m - lo, hi - m])
        xerr = np.nan_to_num(xerr, nan=0.0)
        ax.errorbar(m, y, xerr=xerr, fmt='s', ms=4, capsize=3, elinewidth=1.2)

        # 文字标注（点的左侧，防止出界）
        for yi, mi, st in zip(y, m, stats):
            xt = min(0.98, max(0.0, mi) - 0.03) if np.isfinite(mi) else 0.02
            ax.text(xt, yi, _fmt(st), va="center", ha="right", fontsize=8)

        ax.set_xlabel(xlabel); ax.set_xlim(0.0, 1.0)
        ax.grid(ls="--", alpha=0.3)
        ax.set_yticks(y); ax.set_yticklabels(models)

    # 最优在上
    for ax in axes:
        ax.invert_yaxis()

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)
    print("[save]", out_png)

    # 同步导出 CSV
    recs = []
    for r in rows:
        recs.append({
            "Model": r["model"].upper(),
            "Threshold": r.get("threshold", np.nan),
            "AUC_mean": r["AUC"]["mean"],  "AUC_sd": r["AUC"]["sd"],  "AUC_lo": r["AUC"]["lo"],  "AUC_hi": r["AUC"]["hi"],
            "Spec_mean": r["Spec"]["mean"],"Spec_sd": r["Spec"]["sd"],"Spec_lo": r["Spec"]["lo"],"Spec_hi": r["Spec"]["hi"],
            "Sens_mean": r["Sens"]["mean"],"Sens_sd": r["Sens"]["sd"],"Sens_lo": r["Sens"]["lo"],"Sens_hi": r["Sens"]["hi"],
        })
    pd.DataFrame(recs).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("[save]", out_csv)

def main():
    prob_df, title, scen = _pick_probs_and_title()

    # 尝试提前读取 Validation 用于阈值；如果当前就是 val，直接复用当前 df
    prob_df_for_val = pd.read_csv(PROBS_VAL) if PROBS_VAL.exists() else (prob_df if scen=="internal_val" else None)
    thr_map = _load_thr_map(prob_df_for_val)

    y_col = _find_y_col(prob_df)
    y = prob_df[y_col].values.astype(int)

    # 模型概率列
    prob_cols = [c for c in prob_df.columns if c.startswith("p_")]
    if not prob_cols:
        raise RuntimeError("概率文件中未找到以 'p_' 开头的模型列。")

    rows = []
    for c in prob_cols:
        model_key = c.replace("p_", "").lower()
        p = prob_df[c].values.astype(float)

        # 阈值：优先 Validation（json 或现算）；仍缺失则在当前数据上用 Youden 兜底
        thr = thr_map.get(model_key, None)
        if thr is None:
            thr = _youden_thr(y, p)
            print(f"[WARN] 阈值缺失：{model_key}，已在 {scen} 上用 Youden 兜底：{thr:.4f}")

        auc_stat, spec_stat, sens_stat = _boot_metrics(y, p, thr_fixed=thr)
        rows.append({"model": model_key, "AUC": auc_stat, "Spec": spec_stat, "Sens": sens_stat, "threshold": thr})

    _forest_plot(rows, title, OUT_PNG, OUT_CSV)

if __name__ == "__main__":
    main()
