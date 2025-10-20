# -*- coding: utf-8 -*-
r"""
19_tables_S12_S13.py — NRI/IDI（S12）& 敏感性指标（S13），基于 S7 流水线的“测试集”概率

前提：
- 已运行 14_final_eval_s7.py，存在：
  10_experiments/<VER_OUT>/final_eval_s7/S7_probs_internal_test.csv
- 可选阈值文件：
  10_experiments/<VER_OUT>/final_eval_s7/S7_thresholds_from_val.json

增强点：
- 读取 07_config/config.yaml 自动定位路径
- 命令行：--make all|S12|S13（可逗号组合）, --force 覆盖
- --boots / --alpha / --seed 控制 NRI/IDI 自举；--ordered 输出成对比较双向
- 概率列清洗：NaN→中位数，随后 clip 到 (1e-12, 1-1e-12)
- 已存在即跳过；打印清晰的进度/耗时
"""

from pathlib import Path
import argparse, json, time
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score, roc_auc_score, brier_score_loss, f1_score, confusion_matrix
)
from sklearn.utils import check_random_state


# ================== 读取配置 ==================
def repo_root() -> Path:
    # 本脚本位于 03_scripts/ 下；父级为项目根
    return Path(__file__).resolve().parents[1]

CFG = yaml.safe_load(open(repo_root() / "07_config" / "config.yaml", "r", encoding="utf-8"))
PROJECT_ROOT = Path(CFG["paths"]["project_root"])
VER_OUT = CFG.get("run_id_out", CFG.get("run_id"))

PROBS_DIR = PROJECT_ROOT / "10_experiments" / VER_OUT / "final_eval_s7"
PROBS_FILE_TEST = PROBS_DIR / "S7_probs_internal_test.csv"
THRESHOLDS_JSON = PROBS_DIR / "S7_thresholds_from_val.json"

OUT_DIR = PROJECT_ROOT / "10_experiments" / VER_OUT / "tables_S12_S13"
OUT_DIR.mkdir(parents=True, exist_ok=True)

Y_NAME_CFG = (CFG.get("outcome", {}) or {}).get("name", None)

# 默认统计设置
DEF_N_BOOT = 1000
DEF_SEED   = 42
DEF_ALPHA  = 0.05
# ===============================================================


# ---------- 格式化 ----------
def pfmt(p):
    if p is None or (isinstance(p, float) and (np.isnan(p) or np.isinf(p))):
        return ""
    return "<0.0001" if p < 1e-4 else (f"{p:.2f}" if p >= 0.01 else f"{p:.3f}")

def ci_str(est, lo, hi, k=4):
    if any(pd.isna([est, lo, hi])):
        return ""
    return f"{est:.{k}f}({lo:.{k}f}–{hi:.{k}f})"


# ---------- IO ----------
def _sanitize_proba(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    a[~np.isfinite(a)] = np.nan
    if np.isnan(a).any():
        med = np.nanmedian(a)
        a = np.where(np.isnan(a), med, a)
    return np.clip(a, 1e-12, 1-1e-12)

def load_probs(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"未找到概率文件：{path}\n请先运行 14_final_eval_s7.py。")
    df = pd.read_csv(path)

    # 自动识别标签列：优先 config 中 outcome 名，其次 'y'
    y_col = None
    for cand in [Y_NAME_CFG, "y"]:
        if cand and cand in df.columns:
            y_col = cand
            break
    if not y_col:
        raise RuntimeError(f"{path} 中找不到标签列（尝试了 {Y_NAME_CFG!r} 和 'y'）")

    y = df[y_col].values.astype(int)
    model_cols = [c for c in df.columns if c.startswith("p_")]
    if not model_cols:
        raise RuntimeError(f"{path} 中未发现模型概率列（以 'p_' 开头）")

    probs = {}
    for c in model_cols:
        key = c.replace("p_", "").lower()
        probs[key] = _sanitize_proba(df[c].values)
    return y, probs

def load_thresholds(path: Path):
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return {str(k).lower(): float(v) for k, v in raw.items()}
        except Exception:
            return {}
    return {}


# ---------- 指标（S13） ----------
def binary_metrics(y_true, p_hat, thr=0.5):
    y_true = np.asarray(y_true).astype(int)
    p_hat = np.asarray(p_hat, dtype=float)
    y_pred = (p_hat >= thr).astype(int)

    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, p_hat)
    except Exception:
        auc = np.nan
    brier = brier_score_loss(y_true, p_hat)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv  = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    f1   = f1_score(y_true, y_pred) if (tp + fp > 0 and tp + fn > 0) else np.nan
    return {
        "Accuracy": acc, "AUC": auc, "Brier": brier, "Sensitivity": sens,
        "Specificity": spec, "Precision": prec, "NPV": npv, "F1": f1
    }


# ---------- NRI / IDI（S12，连续） ----------
def _nri_continuous(y, p_new, p_base):
    """连续 NRI： (↑cases-↓cases) + (↓controls-↑controls)"""
    y = np.asarray(y).astype(int)
    p_new = np.asarray(p_new, dtype=float)
    p_base = np.asarray(p_base, dtype=float)
    ev = (y == 1)
    nev = (y == 0)

    if not ev.any() or not nev.any():
        return np.nan

    up_ev   = np.mean((p_new[ev] > p_base[ev]).astype(float))
    down_ev = np.mean((p_new[ev] < p_base[ev]).astype(float))
    up_nev   = np.mean((p_new[nev] < p_base[nev]).astype(float))  # 控制组“下降”为改进
    down_nev = np.mean((p_new[nev] > p_base[nev]).astype(float))
    return (up_ev - down_ev) + (up_nev - down_nev)

def _idi(y, p_new, p_base):
    """IDI：新模型与基线模型的分离度差 (mean[p|Y=1]-mean[p|Y=0])"""
    y = np.asarray(y).astype(int)
    p_new = np.asarray(p_new, dtype=float)
    p_base = np.asarray(p_base, dtype=float)
    ev = (y == 1); nev = (y == 0)
    if not (ev.any() and nev.any()):
        return np.nan
    return (p_new[ev].mean() - p_new[nev].mean()) - (p_base[ev].mean() - p_base[nev].mean())

def bootstrap_ci_p(stat_fn, y, p_new, p_base, n_boot: int, seed: int, alpha: float):
    """自举置信区间 + 双侧 p（以0为原假设，经验分布）"""
    rng = check_random_state(seed)
    n = len(y)
    vals = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.randint(0, n, n)
        vals[i] = stat_fn(y[idx], p_new[idx], p_base[idx])
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return np.nan, (np.nan, np.nan), np.nan
    est = stat_fn(y, p_new, p_base)
    lo, hi = np.percentile(vals, [100*alpha/2, 100*(1-alpha/2)])
    # 双侧 p 值：经验法
    p_two = 2 * min((vals <= 0).mean(), (vals >= 0).mean())
    return est, (lo, hi), p_two


# ---------- 主流程 ----------
def run(make_parts, force: bool, boots: int, alpha: float, seed: int, ordered: bool):
    t0_all = time.time()
    print(f"[cfg ] VER_OUT={VER_OUT}")
    print(f"[io  ] PROBS_DIR={PROBS_DIR}")
    print(f"[io  ] OUT_DIR  ={OUT_DIR}")
    print(f"[args] make={make_parts}, force={force}, boots={boots}, alpha={alpha}, seed={seed}, ordered={ordered}")

    # 读测试集概率 + 阈值
    y_test, probs = load_probs(PROBS_FILE_TEST)
    thr_map = load_thresholds(THRESHOLDS_JSON)
    models = sorted(probs.keys())
    print(f"[data] test rows={len(y_test)}, models={models}")
    print(f"[hint] thresholds loaded: {len(thr_map)}（缺失回退 0.5）")

    # 路径
    p_S12 = OUT_DIR / "S12_nri_idi_test.csv"
    p_S13 = OUT_DIR / "S13_sensitivity_test.csv"

    # ---------- S13 ----------
    if ("S13" in make_parts) or ("all" in make_parts):
        if not force and p_S13.exists():
            print(f"[skip] {p_S13} 已存在（--force 可覆盖）")
        else:
            t1 = time.time()
            rows = []
            for m in models:
                thr = float(thr_map.get(m, 0.5))
                met = binary_metrics(y_test, probs[m], thr=thr)
                rows.append({
                    "Model": m.upper(), "Dataset": "Test set", "Threshold": thr,
                    "Accuracy": met["Accuracy"], "AUC": met["AUC"], "Brier": met["Brier"],
                    "Sensitivity": met["Sensitivity"], "Specificity": met["Specificity"],
                    "Precision": met["Precision"], "NPV": met["NPV"], "F1": met["F1"],
                })
            df = pd.DataFrame(rows)
            # 美化到 4 位 & 排序
            for c in ["Accuracy","AUC","Brier","Sensitivity","Specificity","Precision","NPV","F1"]:
                df[c] = df[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
            df = df.sort_values(["Accuracy","AUC"], ascending=False)
            df.to_csv(p_S13, index=False, encoding="utf-8-sig")
            print(f"[save] {p_S13}  (time {time.time()-t1:.1f}s)")

    # ---------- S12 ----------
    if ("S12" in make_parts) or ("all" in make_parts):
        if not force and p_S12.exists():
            print(f"[skip] {p_S12} 已存在（--force 可覆盖）")
        else:
            if len(models) < 2:
                print("[warn] 模型不足 2 个，无法进行成对比较（S12 跳过）。")
            else:
                t2 = time.time()
                pairs = []
                if ordered:
                    for i, a in enumerate(models):
                        for j, b in enumerate(models):
                            if i != j:
                                pairs.append((a, b))
                else:
                    for i in range(len(models)):
                        for j in range(i+1, len(models)):
                            pairs.append((models[i], models[j]))

                rows = []
                n_boot = int(boots or DEF_N_BOOT)
                a_ci   = float(alpha or DEF_ALPHA)
                sd     = int(seed  or DEF_SEED)

                print(f"[S12 ] pairs={len(pairs)}, boots={n_boot}, alpha={a_ci}, seed={sd}")
                for new, base in pairs:
                    p_new, p_base = probs[new], probs[base]
                    # NRI
                    nri_est, (nri_lo, nri_hi), p_nri = bootstrap_ci_p(
                        _nri_continuous, y_test, p_new, p_base,
                        n_boot=n_boot, seed=sd, alpha=a_ci
                    )
                    # IDI（换一个种子避免完全相同抽样）
                    idi_est, (idi_lo, idi_hi), p_idi = bootstrap_ci_p(
                        _idi, y_test, p_new, p_base,
                        n_boot=n_boot, seed=sd+1, alpha=a_ci
                    )
                    rows.append({
                        "New model": new.upper(),
                        "Baseline model": base.upper(),
                        "NRI": ci_str(nri_est, nri_lo, nri_hi, k=4),
                        "P value (NRI)": pfmt(p_nri),
                        "IDI": ci_str(idi_est, idi_lo, idi_hi, k=4),
                        "P value (IDI)": pfmt(p_idi),
                    })
                pd.DataFrame(rows).sort_values(["New model","Baseline model"]).to_csv(
                    p_S12, index=False, encoding="utf-8-sig"
                )
                print(f"[save] {p_S12}  (time {time.time()-t2:.1f}s)")

    print(f"[done] 完成，总用时 {time.time()-t0_all:.1f}s")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make", type=str, default="all",
                    help="生成哪些表：all | S12 | S13，多个用逗号，如 S12,S13")
    ap.add_argument("--force", action="store_true", help="覆盖已存在文件")
    ap.add_argument("--boots", type=int, default=None, help=f"NRI/IDI 自举次数（默认 {DEF_N_BOOT}）")
    ap.add_argument("--alpha", type=float, default=None, help=f"置信区间 alpha（默认 {DEF_ALPHA}）")
    ap.add_argument("--seed",  type=int, default=None, help=f"随机种子（默认 {DEF_SEED}）")
    ap.add_argument("--ordered", action="store_true", help="S12 输出双向成对比较（默认只输出唯一组合 i<j）")
    args = ap.parse_args()

    mk = [x.strip() for x in args.make.split(",") if x.strip()]
    mk = [m for m in mk if m in ("all","S12","S13")]
    if not mk:
        mk = ["all"]
    return mk, args.force, args.boots, args.alpha, args.seed, args.ordered


if __name__ == "__main__":
    make_parts, force, boots, alpha, seed, ordered = parse_args()
    run(make_parts, force, boots, alpha, seed, ordered)
