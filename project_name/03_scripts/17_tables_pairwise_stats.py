# -*- coding: utf-8 -*-
r"""
17_tables_pairwise_stats.py — 复现 PDF 表 S7/S8(性能&对数损失) + S9/S11(DeLong) + S10/S12(NRI/IDI)

前提：
- 已运行 14_final_eval_s7.py，存在 10_experiments/<VER_OUT>/final_eval_s7/
  ├─ S7_probs_internal_train.csv
  └─ S7_probs_internal_test.csv
- 如存在 S7_thresholds_from_val.json，则二分类指标优先采用其中的 Youden 阈值，否则回退 0.5
"""

from pathlib import Path
from typing import Dict, Tuple, List
import argparse, json, time
import numpy as np
import pandas as pd
import yaml

# ================== 读取配置 ==================
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

CFG = yaml.safe_load(open(repo_root() / "07_config" / "config.yaml", "r", encoding="utf-8"))

PROJECT_ROOT = Path(CFG["paths"]["project_root"])
VER_OUT = CFG.get("run_id_out", CFG.get("run_id"))
OUT_DIR   = PROJECT_ROOT / "10_experiments" / VER_OUT / "pairwise_stats"
PROBS_DIR = PROJECT_ROOT / "10_experiments" / VER_OUT / "final_eval_s7"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# outcome 名称（若概率csv里没有该列会自动回退到 'y'）
Y_NAME_CFG = (CFG.get("outcome", {}) or {}).get("name", None)

# NRI/IDI 默认参数（可被命令行覆盖）
DEF_N_BOOT_NRI = 600
DEF_SEED_NRI   = 2024
DEF_ALPHA      = 0.05


# ---------- 基础工具 ----------
def fmt_float(x: float, k: int = 3) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return ""
    return f"{float(x):.{k}f}"

def fmt_ci(lo: float, hi: float, k: int = 3) -> str:
    if any([v is None for v in (lo, hi)]) or any([isinstance(v, float) and np.isnan(v) for v in (lo, hi)]):
        return ""
    return f"{float(lo):.{k}f}–{float(hi):.{k}f}"

def fmt_p(p: float) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    try:
        p = float(p)
    except Exception:
        return ""
    return "<0.001" if p < 0.001 else f"{p:.3f}"

def _load_thresholds(th_json: Path) -> Dict[str, float]:
    if th_json.exists():
        try:
            raw = json.loads(th_json.read_text(encoding="utf-8"))
            return {str(k).lower(): float(v) for k, v in raw.items()}
        except Exception:
            return {}
    return {}

def _common_models(*dicts: Dict[str, np.ndarray]) -> List[str]:
    s = set(dicts[0].keys())
    for d in dicts[1:]:
        s &= set(d.keys())
    return sorted(s)

def _sanitize_proba(p: np.ndarray) -> np.ndarray:
    """确保概率为浮点并裁剪到[0,1]；对非常规值做轻微清理。"""
    p = np.asarray(p, dtype=float)
    p = np.where(np.isfinite(p), p, np.nan)
    # 若存在极少量 NaN，用中位数填一下，避免指标计算直接崩
    if np.isnan(p).any():
        med = np.nanmedian(p)
        p = np.where(np.isnan(p), med, p)
    return np.clip(p, 1e-12, 1-1e-12)

# ---------- 读概率文件 ----------
def load_probs(path: Path) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    if not path.exists():
        raise FileNotFoundError(f"缺少概率文件：{path}")
    df = pd.read_csv(path)

    # 自动识别标签列：优先 config 的 outcome 名，其次 'y'
    y_col = None
    for cand in [Y_NAME_CFG, "y"]:
        if cand and cand in df.columns:
            y_col = cand
            break
    if not y_col:
        raise KeyError(f"找不到标签列（尝试了 {Y_NAME_CFG!r} 和 'y'）：{path}")

    y = df[y_col].values.astype(int)
    model_cols = [c for c in df.columns if c.startswith("p_")]
    if not model_cols:
        raise ValueError(f"未发现模型概率列（p_*）：{path}")

    data = {}
    for c in model_cols:
        mk = c.replace("p_", "").lower()
        data[mk] = _sanitize_proba(df[c].values)
    return y, data

# ---------- 指标 ----------
def bin_metrics(y, p, thr=0.5):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    yhat = (p >= thr).astype(int)
    tn = np.sum((y==0)&(yhat==0)); fp = np.sum((y==0)&(yhat==1))
    fn = np.sum((y==1)&(yhat==0)); tp = np.sum((y==1)&(yhat==1))
    spec = tn / max((tn+fp), 1)
    npv  = tn / max((tn+fn), 1)
    return dict(
        Accuracy=float(accuracy_score(y,yhat)),
        Precision=float(precision_score(y,yhat,zero_division=0)),
        Recall=float(recall_score(y,yhat,zero_division=0)),
        Specificity=float(spec),
        NPV=float(npv),
        F1=float(f1_score(y,yhat,zero_division=0)),
    )

def logloss(y, p):
    y = np.asarray(y).astype(int); p = np.clip(p, 1e-15, 1-1e-15)
    return float(np.mean(-(y*np.log(p) + (1-y)*np.log(1-p))))

def auc_score(y, p):
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y, p))
    except Exception:
        return float("nan")

def auprc_score(y, p):
    from sklearn.metrics import average_precision_score
    try:
        return float(average_precision_score(y, p))
    except Exception:
        return float("nan")

# ---------- DeLong（相关） ----------
def _psi_matrix(pos: np.ndarray, neg: np.ndarray, block: int = 4096):
    pos = np.asarray(pos, dtype=float); neg = np.asarray(neg, dtype=float)
    m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        return np.array([]), np.array([]), float("nan")
    v10 = np.empty(n, dtype=float); s10 = 0.0
    start = 0
    while start < n:
        nb = neg[start:start+block]
        comp = (pos[:, None] > nb[None, :]).astype(float) + 0.5*(pos[:, None] == nb[None, :])
        v10[start:start+len(nb)] = comp.mean(axis=0)
        s10 += v10[start:start+len(nb)].sum()
        start += block
    auc = s10 / n
    v01 = np.empty(m, dtype=float)
    start = 0
    while start < m:
        pb = pos[start:start+block]
        comp = (pb[:, None] > neg[None, :]).astype(float) + 0.5*(pb[:, None] == neg[None, :])
        v01[start:start+len(pb)] = comp.mean(axis=1)
        start += block
    return v01, v10, float(auc)

def _delong_components(y: np.ndarray, p: np.ndarray):
    y = np.asarray(y).astype(int); p = np.asarray(p, dtype=float)
    pos = p[y==1]; neg = p[y==0]
    m, n = len(pos), len(neg)
    if m==0 or n==0:
        return np.nan, np.nan, np.array([]), np.array([])
    v01, v10, auc = _psi_matrix(pos, neg)
    if not np.isfinite(auc):
        return np.nan, np.nan, np.array([]), np.array([])
    s01 = np.var(v01, ddof=1) if m>1 else 0.0
    s10 = np.var(v10, ddof=1) if n>1 else 0.0
    var = s10/n + s01/m
    return auc, float(var), v01, v10

def delong_test_correlated(y: np.ndarray, p_new: np.ndarray, p_base: np.ndarray, alpha: float):
    y = np.asarray(y).astype(int)
    p1 = np.asarray(p_new, dtype=float)
    p0 = np.asarray(p_base, dtype=float)
    auc1, var1, v01_1, v10_1 = _delong_components(y, p1)
    auc0, var0, v01_0, v10_0 = _delong_components(y, p0)
    if np.isnan(auc1) or np.isnan(auc0):
        return dict(new_auc=np.nan, base_auc=np.nan, diff=np.nan, lo=np.nan, hi=np.nan, z=np.nan, p=np.nan)
    m = np.sum(y==1); n = np.sum(y==0)
    cov01 = (np.cov(v01_1, v01_0, ddof=1)[0,1] if m>1 else 0.0)
    cov10 = (np.cov(v10_1, v10_0, ddof=1)[0,1] if n>1 else 0.0)
    cov = cov10/n + cov01/m
    var_diff = max(var1 + var0 - 2*cov, 1e-12)
    diff = auc1 - auc0
    se = np.sqrt(var_diff)
    from scipy.stats import norm
    z = diff / se
    p = 2*(1 - norm.cdf(abs(z)))
    lo = diff + norm.ppf(alpha/2)*se
    hi = diff + norm.ppf(1-alpha/2)*se
    return dict(new_auc=float(auc1), base_auc=float(auc0), diff=float(diff),
                lo=float(lo), hi=float(hi), z=float(z), p=float(p))

# ---------- NRI / IDI（连续） ----------
def nri_idi_continuous(y, p_new, p_base, n_boot: int, seed: int, alpha: float):
    y  = np.asarray(y).astype(int)
    p1 = np.asarray(p_new, dtype=float)
    p0 = np.asarray(p_base, dtype=float)
    rng = np.random.default_rng(seed)

    # 观测统计量
    up_case   = np.mean((p1[y==1] > p0[y==1]).astype(float))
    down_case = np.mean((p1[y==1] < p0[y==1]).astype(float))
    up_ctrl   = np.mean((p1[y==0] > p0[y==0]).astype(float))
    down_ctrl = np.mean((p1[y==0] < p0[y==0]).astype(float))
    nri = (up_case - down_case) + (down_ctrl - up_ctrl)

    disc1 = p1[y==1].mean() - p1[y==0].mean()
    disc0 = p0[y==1].mean() - p0[y==0].mean()
    idi = disc1 - disc0

    # 自举分布
    stats = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy, q1, q0 = y[idx], p1[idx], p0[idx]
        up_case   = np.mean((q1[yy==1] > q0[yy==1]).astype(float))
        down_case = np.mean((q1[yy==1] < q0[yy==1]).astype(float))
        up_ctrl   = np.mean((q1[yy==0] > q0[yy==0]).astype(float))
        down_ctrl = np.mean((q1[yy==0] < q0[yy==0]).astype(float))
        nri_b = (up_case - down_case) + (down_ctrl - up_ctrl)
        idi_b = (q1[yy==1].mean() - q1[yy==0].mean()) - (q0[yy==1].mean() - q0[yy==0].mean())
        stats.append((nri_b, idi_b))
    arr = np.asarray(stats)
    lo_nri, hi_nri = np.percentile(arr[:,0], [100*alpha/2, 100*(1-alpha/2)])
    lo_idi, hi_idi = np.percentile(arr[:,1], [100*alpha/2, 100*(1-alpha/2)])

    # 正态近似 p（基于自举分布）
    def p_from_boot(obs, samp):
        m = float(np.nanmean(samp)); s = float(np.nanstd(samp, ddof=1))
        if s == 0:
            return 1.0
        from scipy.stats import norm
        z = (obs - m) / s
        return float(2*(1-norm.cdf(abs(z))))

    return dict(NRI=float(nri), NRI_lo=float(lo_nri), NRI_hi=float(hi_nri), NRI_p=p_from_boot(nri, arr[:,0]),
                IDI=float(idi), IDI_lo=float(lo_idi), IDI_hi=float(hi_idi), IDI_p=p_from_boot(idi, arr[:,1]))

# ---------- 主流程 ----------
def run(make_parts: List[str], force: bool, boots: int, alpha: float, seed: int):
    t_all0 = time.time()
    print(f"[cfg ] VER_OUT={VER_OUT}")
    print(f"[io  ] PROBS_DIR={PROBS_DIR}")
    print(f"[io  ] OUT_DIR  ={OUT_DIR}")
    print(f"[args] make={make_parts}, force={force}, boots={boots}, alpha={alpha}, seed={seed}")

    # 读概率与阈值
    y_tr, probs_tr = load_probs(PROBS_DIR / "S7_probs_internal_train.csv")
    y_te, probs_te = load_probs(PROBS_DIR / "S7_probs_internal_test.csv")
    thr_map = _load_thresholds(PROBS_DIR / "S7_thresholds_from_val.json")  # 可能不存在
    print(f"[hint] thresholds loaded: {len(thr_map)} (用于性能阈值；缺失回退 0.5)")

    # 仅保留 train/test 都有的模型
    models = _common_models(probs_tr, probs_te)
    if not models:
        raise RuntimeError("train 与 test 没有共同的模型列，无法做成对比较。")
    print(f"[data] models={len(models)} -> {models}")

    # 成对组合
    pairs = [(models[i], models[j]) for i in range(len(models)) for j in range(i+1, len(models))]
    print(f"[data] pairwise comparisons={len(pairs)}")

    # 路径
    p_S7 = OUT_DIR / "S7_logloss_train_test.csv"
    p_S8 = OUT_DIR / "S8_performance_train_test.csv"      # 增：含 AUROC/AUPRC
    p_S9 = OUT_DIR / "S9_delong_train.csv"
    p_S11= OUT_DIR / "S11_delong_test.csv"
    p_S10= OUT_DIR / "S10_nri_idi_train.csv"
    p_S12= OUT_DIR / "S12_nri_idi_test.csv"

    # ---------- S7 & S8 ----------
    if ("S7S8" in make_parts) or ("all" in make_parts):
        if not force and p_S7.exists() and p_S8.exists():
            print(f"[skip] S7/S8 已存在（--force 可覆盖）")
        else:
            t0 = time.time()
            rows_ll, rows_perf = [], []
            for m in models:
                p_tr = probs_tr[m]; p_te = probs_te[m]
                rows_ll.append({
                    "Model": m.upper(),
                    "Log_Loss_Train": fmt_float(logloss(y_tr, p_tr), 3),
                    "Log_Loss_Test":  fmt_float(logloss(y_te, p_te), 3),
                })
                thr = float(thr_map.get(m, 0.5))
                perf_tr = bin_metrics(y_tr, p_tr, thr=thr)
                perf_te = bin_metrics(y_te, p_te, thr=thr)
                rows_perf.append({
                    "Dataset": "Training set", "Model": m.upper(),
                    "Accuracy": fmt_float(perf_tr["Accuracy"], 3),
                    "Precision": fmt_float(perf_tr["Precision"], 3),
                    "Recall": fmt_float(perf_tr["Recall"], 3),
                    "Specificity": fmt_float(perf_tr["Specificity"], 3),
                    "NPV": fmt_float(perf_tr["NPV"], 3),
                    "F1": fmt_float(perf_tr["F1"], 3),
                    "AUROC": fmt_float(auc_score(y_tr, p_tr), 3),
                    "AUPRC": fmt_float(auprc_score(y_tr, p_tr), 3),
                })
                rows_perf.append({
                    "Dataset": "Test set", "Model": m.upper(),
                    "Accuracy": fmt_float(perf_te["Accuracy"], 3),
                    "Precision": fmt_float(perf_te["Precision"], 3),
                    "Recall": fmt_float(perf_te["Recall"], 3),
                    "Specificity": fmt_float(perf_te["Specificity"], 3),
                    "NPV": fmt_float(perf_te["NPV"], 3),
                    "F1": fmt_float(perf_te["F1"], 3),
                    "AUROC": fmt_float(auc_score(y_te, p_te), 3),
                    "AUPRC": fmt_float(auprc_score(y_te, p_te), 3),
                })
            pd.DataFrame(rows_ll).sort_values("Model").to_csv(p_S7, index=False, encoding="utf-8-sig")
            pd.DataFrame(rows_perf).sort_values(["Dataset","Model"]).to_csv(p_S8, index=False, encoding="utf-8-sig")
            print(f"[save] {p_S7}")
            print(f"[save] {p_S8}")
            print(f"[time] S7/S8 用时 {time.time()-t0:.1f}s")

    # ---------- S9 / S11：DeLong（train / test） ----------
    if ("S9S11" in make_parts) or ("all" in make_parts):
        if len(pairs) == 0:
            print("[skip] S9/S11：仅有一个模型，无成对比较可做（需要至少两个模型）")
        elif not force and p_S9.exists() and p_S11.exists():
            print(f"[skip] S9/S11 已存在（--force 可覆盖）")
        else:
            t0 = time.time()
            rows_tr, rows_te = [], []
            use_alpha = DEF_ALPHA if alpha is None else alpha
            for a, b in pairs:
                dtr = delong_test_correlated(y_tr, probs_tr[a], probs_tr[b], alpha=use_alpha)
                dte = delong_test_correlated(y_te, probs_te[a], probs_te[b], alpha=use_alpha)
                rows_tr.append({
                    "New model": a.upper(), "Baseline model": b.upper(),
                    "New AUC": fmt_float(dtr["new_auc"], 3),
                    "Baseline AUC": fmt_float(dtr["base_auc"], 3),
                    "Difference of AUC": fmt_float(dtr["diff"], 3),
                    "95% CI low": fmt_float(dtr["lo"], 3),
                    "95% CI high": fmt_float(dtr["hi"], 3),
                    "Z statistic": fmt_float(dtr["z"], 3),
                    "P": fmt_p(dtr["p"]),
                })
                rows_te.append({
                    "New model": a.upper(), "Baseline model": b.upper(),
                    "New AUC": fmt_float(dte["new_auc"], 3),
                    "Baseline AUC": fmt_float(dte["base_auc"], 3),
                    "Difference of AUC": fmt_float(dte["diff"], 3),
                    "95% CI low": fmt_float(dte["lo"], 3),
                    "95% CI high": fmt_float(dte["hi"], 3),
                    "Z statistic": fmt_float(dte["z"], 3),
                    "P": fmt_p(dte["p"]),
                })
            if rows_tr:
                pd.DataFrame(rows_tr).sort_values(["New model","Baseline model"]).to_csv(p_S9, index=False, encoding="utf-8-sig")
                print(f"[save] {p_S9}")
            if rows_te:
                pd.DataFrame(rows_te).sort_values(["New model","Baseline model"]).to_csv(p_S11, index=False, encoding="utf-8-sig")
                print(f"[save] {p_S11}")
            print(f"[time] S9/S11 用时 {time.time()-t0:.1f}s")

    # ---------- S10 / S12：NRI / IDI（train / test） ----------
    if ("S10S12" in make_parts) or ("all" in make_parts):
        if len(pairs) == 0:
            print("[skip] S10/S12：仅有一个模型，无成对比较可做（需要至少两个模型）")
        elif not force and p_S10.exists() and p_S12.exists():
            print(f"[skip] S10/S12 已存在（--force 可覆盖）")
        else:
            t0 = time.time()
            rows_tr, rows_te = [], []
            use_boots = boots or DEF_N_BOOT_NRI
            use_seed  = seed  or DEF_SEED_NRI
            use_alpha = alpha or DEF_ALPHA
            for a, b in pairs:
                ntr = nri_idi_continuous(y_tr, probs_tr[a], probs_tr[b],
                                         n_boot=use_boots, seed=use_seed, alpha=use_alpha)
                nte = nri_idi_continuous(y_te, probs_te[a], probs_te[b],
                                         n_boot=use_boots, seed=use_seed, alpha=use_alpha)
                rows_tr.append({
                    "New model": a.upper(), "Baseline model": b.upper(),
                    "NRI": fmt_float(ntr["NRI"], 3),
                    "NRI 95%CI": fmt_ci(ntr['NRI_lo'], ntr['NRI_hi'], 3),
                    "P value (NRI)": fmt_p(ntr["NRI_p"]),
                    "IDI": fmt_float(ntr["IDI"], 3),
                    "IDI 95%CI": fmt_ci(ntr['IDI_lo'], ntr['IDI_hi'], 3),
                    "P value (IDI)": fmt_p(ntr["IDI_p"]),
                })
                rows_te.append({
                    "New model": a.upper(), "Baseline model": b.upper(),
                    "NRI": fmt_float(nte["NRI"], 3),
                    "NRI 95%CI": fmt_ci(nte['NRI_lo'], nte['NRI_hi'], 3),
                    "P value (NRI)": fmt_p(nte["NRI_p"]),
                    "IDI": fmt_float(nte["IDI"], 3),
                    "IDI 95%CI": fmt_ci(nte['IDI_lo'], nte['IDI_hi'], 3),
                    "P value (IDI)": fmt_p(nte["IDI_p"]),
                })
            if rows_tr:
                pd.DataFrame(rows_tr).sort_values(["New model","Baseline model"]).to_csv(p_S10, index=False, encoding="utf-8-sig")
                print(f"[save] {p_S10}")
            if rows_te:
                pd.DataFrame(rows_te).sort_values(["New model","Baseline model"]).to_csv(p_S12, index=False, encoding="utf-8-sig")
                print(f"[save] {p_S12}")
            print(f"[time] S10/S12 用时 {time.time()-t0:.1f}s")

    print(f"[done] 全部完成，总用时 {time.time()-t_all0:.1f}s")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make", type=str, default="all",
                    help="生成哪些表：all | S7S8 | S9S11 | S10S12，多个用逗号，如 S7S8,S10S12")
    ap.add_argument("--force", action="store_true", help="覆盖已存在文件")
    ap.add_argument("--boots", type=int, default=None, help=f"NRI/IDI 自举次数（默认 {DEF_N_BOOT_NRI}）")
    ap.add_argument("--alpha", type=float, default=None, help=f"置信区间alpha（默认 {DEF_ALPHA}）")
    ap.add_argument("--seed",  type=int, default=None, help=f"自举随机种子（默认 {DEF_SEED_NRI}）")
    args = ap.parse_args()
    mk = [x.strip() for x in args.make.split(",") if x.strip()]
    mk = [m for m in mk if m in ("all","S7S8","S9S11","S10S12")]
    if not mk:
        mk = ["all"]
    return mk, args.force, args.boots, args.alpha, args.seed

if __name__ == "__main__":
    parts, force, boots, alpha, seed = parse_args()
    run(parts, force, boots, alpha, seed)
