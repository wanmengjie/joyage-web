# -*- coding: utf-8 -*-
r"""
16_tables_S1_S4.py — 复现 PDF 的表 S1/S2（插补前后对比）与 S3/S4（不同插补法敏感性）

改进点：
- 统一读 YAML（project_root / run_id / y_name）
- 兼容并标准化指标别名（Brier/Brier_Score，Sensitivity/Recall 等）
- S3/S4 先读场景 CSV，缺失→回退 summary；仍缺→再回退 S4 的 feature_sweep summary
- S1/S2 连续：中位数(IQR)；分类：逐水平计数(%)；p 值格式化
"""
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import warnings, yaml
warnings.filterwarnings("ignore", category=UserWarning)
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, precision_score,
    recall_score, f1_score, brier_score_loss
)

# ================== 统一配置 ==================
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

CFG = yaml.safe_load(open(repo_root()/ "07_config"/"config.yaml","r",encoding="utf-8"))

PROJECT_ROOT = Path(CFG["paths"]["project_root"])
VER_IN  = CFG.get("run_id_in",  CFG.get("run_id"))
VER_OUT = CFG.get("run_id_out", CFG.get("run_id"))
Y_NAME  = (CFG.get("outcome", {}) or {}).get("name", "depression_bin")

DATA_DIR   = PROJECT_ROOT / "02_processed_data" / VER_IN
FROZEN_DIR = DATA_DIR / "frozen"
SPLIT_DIR  = DATA_DIR / "splits"

# “插补后”——固定使用 mean_mode（带 y）
IMPUTED_XY = FROZEN_DIR / "charls_mean_mode_Xy.csv"

# “插补前”——带缺失的原始 Xy（候选路径，存在一个即可）
RAW_XY_CANDIDATES = [
    FROZEN_DIR / "charls_raw_Xy.csv",
    FROZEN_DIR / "charls_before_impute_Xy.csv",
    FROZEN_DIR / "charls_Xy_with_na.csv",
]

# 划分索引
TRAIN_IDX = SPLIT_DIR / "charls_train_idx.csv"
TEST_IDX  = SPLIT_DIR / "charls_test_idx.csv"

# 07 的评估目录（优先）
EVAL_DIR    = PROJECT_ROOT / "10_experiments" / VER_OUT / "multi_model_eval"
SUMMARY_CSV = EVAL_DIR / "impute_model_selection_summary.csv"

# 额外回退：S4 的目录
FEATURE_SWEEP_DIR = PROJECT_ROOT / "10_experiments" / VER_OUT / "feature_sweep"

OUT_DIR = PROJECT_ROOT / "10_experiments" / VER_OUT / "tables_S1_S4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Step 14 outputs (if available)
FINAL_S7_DIR = PROJECT_ROOT / "10_experiments" / VER_OUT / "final_eval_s7"

# 目标指标与别名映射
METRIC_ALIAS = {
    "AUROC": "AUROC",
    "AUPRC": "AUPRC",
    "Accuracy": "Accuracy",
    "Precision": "Precision",
    "Recall": "Recall",          # = Sensitivity
    "Sensitivity": "Recall",
    "F1_Score": "F1_Score",
    "Brier": "Brier",
    "Brier_Score": "Brier",
    "Specificity": "Specificity",
    "NPV": "NPV",
}

DISPLAY_RENAME = {  # 最终列名
    "AUROC": "AUC",
    "AUPRC": "AUPRC",
    "Accuracy": "Accuracy",
    "Precision": "Precision",
    "Recall": "Sensitivity",
    "F1_Score": "F1",
    "Brier": "Brier",
    "Specificity": "Specificity",
    "NPV": "NPV",
}

SCEN_FILES = {
    "internal_train": EVAL_DIR / "impute_model_selection_internal_train.csv",
    "internal_test":  EVAL_DIR / "impute_model_selection_internal_test.csv",
}

# ================== 工具 ==================
def _first_exists(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None

def load_idx(p: Path) -> Optional[np.ndarray]:
    if not p.exists():
        return None
    s = pd.read_csv(p).iloc[:,0]
    return s.values.astype(int)

def read_yaml_types_from_cfg() -> Tuple[List[str], List[str]]:
    vt = (CFG.get("preprocess", {}) or {}).get("variable_types", {}) or {}
    num_cols = vt.get("continuous", []) or []
    cat_cols = vt.get("categorical", []) or []
    return list(num_cols), list(cat_cols)

def infer_types(df: pd.DataFrame,
                seed_num: Optional[List[str]]=None,
                seed_cat: Optional[List[str]]=None) -> Tuple[List[str], List[str]]:
    """自动推断列类型：小整数(≤10)视作分类型；优先使用 YAML 种子。"""
    seed_num = set(seed_num or [])
    seed_cat = set(seed_cat or [])
    feats = [c for c in df.columns if c != Y_NAME]
    num, cat = [], []
    for c in feats:
        s = df[c]
        if c in seed_num:
            num.append(c); continue
        if c in seed_cat:
            cat.append(c); continue
        if pd.api.types.is_numeric_dtype(s):
            nunq = s.dropna().nunique()
            if pd.api.types.is_integer_dtype(s) and nunq <= 10:
                cat.append(c)
            else:
                num.append(c)
        else:
            cat.append(c)
    return num, cat

def _format_p(p: Optional[float]) -> str:
    if p is None or (isinstance(p, float) and (np.isnan(p) or np.isinf(p))):
        return ""
    return "<0.001" if p < 0.001 else f"{p:.3f}"

def format_median_iqr(x: pd.Series) -> str:
    x = pd.to_numeric(x, errors="coerce")
    if x.notna().sum() == 0:
        return "NA"
    q1, med, q3 = np.nanpercentile(x, [25,50,75])
    return f"{med:.2f} ({q1:.2f}-{q3:.2f})"

def mannwhitney_p(a: pd.Series, b: pd.Series) -> float:
    from scipy.stats import mannwhitneyu
    a = pd.to_numeric(a, errors="coerce").dropna().astype(float)
    b = pd.to_numeric(b, errors="coerce").dropna().astype(float)
    if len(a)==0 or len(b)==0:
        return np.nan
    _, p = mannwhitneyu(a, b, alternative="two-sided")
    return float(p)

def chisq_or_fisher_p(counts_before: np.ndarray, counts_after: np.ndarray) -> float:
    from scipy.stats import chi2_contingency, fisher_exact
    tab = np.vstack([counts_before, counts_after])  # shape (2,2)
    try:
        chi2, p, _, exp = chi2_contingency(tab, correction=False)
        if (exp < 5).any():
            _, p = fisher_exact(tab)
        return float(p)
    except Exception:
        return np.nan

# ================== S1/S2：插补前后对比 ==================
def table_before_after(df_raw: pd.DataFrame, df_imp: pd.DataFrame,
                       idx: Optional[np.ndarray], num_cols: List[str], cat_cols: List[str]) -> pd.DataFrame:
    rows = []

    # 连续型
    for c in num_cols:
        if c not in df_raw.columns or c not in df_imp.columns:
            continue
        a = df_raw.loc[idx, c] if idx is not None else df_raw[c]
        b = df_imp .loc[idx, c] if idx is not None else df_imp [c]
        rows.append({
            "Characteristic": c,
            "Before multiple imputation": format_median_iqr(a),
            "After multiple imputation":  format_median_iqr(b),
            "p value": _format_p(mannwhitney_p(a,b))
        })

    # 分类（多分类逐水平）
    for c in cat_cols:
        if c not in df_raw.columns or c not in df_imp.columns:
            continue
        a = df_raw.loc[idx, c] if idx is not None else df_raw[c]
        b = df_imp .loc[idx, c] if idx is not None else df_imp [c]

        # 汇总全部出现过的水平
        levels = pd.Series(pd.concat([a, b], ignore_index=True)).dropna().unique().tolist()
        # 尽量把二分类的“1”放首位
        if set(levels) == {0, 1}:
            levels = [1, 0]

        # 先算整体 2xK 的 p 值（更贴近“变量级”的差异显著性）
        try:
            from scipy.stats import chi2_contingency
            cont = np.vstack([[(a==lv).sum() for lv in levels],
                              [(b==lv).sum() for lv in levels]])
            _, p_var, _, _ = chi2_contingency(cont, correction=False)
        except Exception:
            p_var = np.nan

        for i, lv in enumerate(levels):
            a_cnt = int((a == lv).sum()); b_cnt = int((b == lv).sum())
            a_den = int(a.notna().sum());   b_den = int(b.notna().sum())
            a_pct = (a_cnt / max(a_den, 1)) * 100.0
            b_pct = (b_cnt / max(b_den, 1)) * 100.0

            # 若只想 2x2 水平级的 p，改用 chisq_or_fisher_p
            p_show = p_var if i == 0 else ""
            rows.append({
                "Characteristic": f"{c} == {lv}" if i > 0 else c,
                "Before multiple imputation": f"{a_cnt} ({a_pct:.2f}%)",
                "After multiple imputation":  f"{b_cnt} ({b_pct:.2f}%)",
                "p value": _format_p(p_var) if i == 0 else ""
            })

    out = pd.DataFrame(rows)
    # 让连续变量在前、分类变量在后（粗略排序）
    cont_mask = out["Characteristic"].isin(num_cols)
    out = pd.concat([out[cont_mask], out[~cont_mask]], ignore_index=True)
    return out

# ================== S3/S4：不同插补法敏感性 ==================
def _normalize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """统一指标列名到标准名；保留 method, model 及可用指标。"""
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    col_map = {}
    for c in df.columns:
        if c in METRIC_ALIAS:
            col_map[c] = METRIC_ALIAS[c]
    df = df.rename(columns=col_map)
    keep = ["method","model"] + sorted(set(METRIC_ALIAS.values()) & set(df.columns))
    keep = [c for c in keep if c in df.columns]
    if not keep:
        return pd.DataFrame()
    return df[keep]

def _load_or_fallback_eval(scen_key: str, nice_name: str) -> Optional[pd.DataFrame]:
    """优先加载 07 场景 CSV；若不存在则从 summary 里回退；仍缺则回退 S4 的 feature_sweep 汇总。"""
    # 1) 场景 CSV
    p = SCEN_FILES.get(scen_key)
    if p and p.exists():
        df = pd.read_csv(p)
        df = _normalize_metrics(df)
        if not df.empty:
            df["Dataset"] = nice_name
            return df

    # 2) 07 summary
    if SUMMARY_CSV.exists():
        s = pd.read_csv(SUMMARY_CSV)
        s = s[s.get("scenario","") == scen_key].copy()
        s = _normalize_metrics(s)
        if not s.empty:
            s["Dataset"] = nice_name
            return s

    # 3) 回退 S4：读取所有模型的 S4_RFE_<model>_summary.csv 并拼起来
    parts = []
    if FEATURE_SWEEP_DIR.exists():
        for f in FEATURE_SWEEP_DIR.glob("S4_RFE_*_summary.csv"):
            try:
                sub = pd.read_csv(f)
                # 标准列与 scenario 过滤
                if "scenario" in sub.columns:
                    sub = sub[sub["scenario"] == scen_key]
                sub = _normalize_metrics(sub)
                if not sub.empty:
                    # method/model 兜底
                    if "method" not in sub.columns:
                        sub.insert(0, "method", "mean_mode")
                    if "model" not in sub.columns:
                        # 从文件名推断
                        mk = f.name.replace("S4_RFE_","").replace("_summary.csv","")
                        sub.insert(1, "model", mk)
                    parts.append(sub)
            except Exception:
                continue
    if parts:
        df = pd.concat(parts, ignore_index=True)
        df["Dataset"] = nice_name
        return df

    return None

def _load_s7_probs(scen_key: str) -> Optional[pd.DataFrame]:
    """Load probability table from step 14 if present: S7_probs_<scenario>.csv"""
    fname = f"S7_probs_{scen_key}.csv"
    p = FINAL_S7_DIR / fname
    if p.exists():
        try:
            df = pd.read_csv(p)
            return df
        except Exception:
            return None
    return None

def _metrics_from_probs(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p).clip(1e-8, 1-1e-8)
    yhat = (p >= thr).astype(int)
    tn = int(np.sum((y == 0) & (yhat == 0)))
    fp = int(np.sum((y == 0) & (yhat == 1)))
    fn = int(np.sum((y == 1) & (yhat == 0)))
    tp = int(np.sum((y == 1) & (yhat == 1)))
    def _safe(a, b):
        return float(a) / max(float(b), 1e-12)
    try:
        auroc = roc_auc_score(y, p)
    except Exception:
        auroc = float("nan")
    try:
        auprc = average_precision_score(y, p)
    except Exception:
        auprc = float("nan")
    return {
        "AUROC": auroc,
        "AUPRC": auprc,
        "Accuracy": accuracy_score(y, yhat),
        "Precision": precision_score(y, yhat, zero_division=0),
        "Recall": recall_score(y, yhat, zero_division=0),
        "F1_Score": f1_score(y, yhat, zero_division=0),
        "Brier": brier_score_loss(y, p),
        "Specificity": _safe(tn, tn + fp),
        "NPV": _safe(tn, tn + fn),
    }

def _metrics_from_s7(scen_key: str, nice_name: str) -> Optional[pd.DataFrame]:
    """Build metrics table from step 14 probability files for a scenario."""
    dfp = _load_s7_probs(scen_key)
    if dfp is None or dfp.empty or ("y" not in dfp.columns):
        return None
    y = dfp["y"].values
    rows = []
    for c in dfp.columns:
        if not c.startswith("p_"):
            continue
        model = c[2:]
        met = _metrics_from_probs(y, dfp[c].values, thr=0.5)
        row = {"method": "mean_mode", "model": model, **met, "Dataset": nice_name}
        rows.append(row)
    if not rows:
        return None
    return pd.DataFrame(rows)

def build_s3_s4():
    pieces = []
    # 1) Prefer using step 14 probs if available
    for scen_key, nice in [("internal_train","Training set"),
                           ("internal_test","Test set")]:
        sub_s7 = _metrics_from_s7(scen_key, nice)
        if sub_s7 is not None and not sub_s7.empty:
            pieces.append(sub_s7)

    # 2) Fallback to previous CSV summaries if S7 not available
    if not pieces:
        for scen_key, nice in [("internal_train","Training set"),
                               ("internal_test","Test set")]:
            sub = _load_or_fallback_eval(scen_key, nice)
            if sub is not None and not sub.empty:
                pieces.append(sub)

    if not pieces:
        print("[WARN] 找不到 07 或 S4 的评估结果，S3/S4 跳过")
        return

    big = pd.concat(pieces, ignore_index=True)

    # 只保留真实存在的指标列，并按 DISPLAY_RENAME 的顺序与命名输出
    metric_order = ["Accuracy","AUROC","AUPRC","Brier","Recall","Specificity","Precision","NPV","F1_Score"]
    metric_cols = [m for m in metric_order if m in big.columns]
    ordered = ["method","model"] + metric_cols + ["Dataset"]
    big = big[ordered].copy()
    rename_map = {k:v for k,v in DISPLAY_RENAME.items() if k in big.columns}
    big = big.rename(columns=rename_map)

    # 分 scen 存成两张表
    for scen_name, out_name in [("Training set","S3_imputation_sensitivity_train.csv"),
                                ("Test set","S4_imputation_sensitivity_test.csv")]:
        sub = big[big["Dataset"]==scen_name].drop(columns=["Dataset"])
        sub.to_csv(OUT_DIR / out_name, index=False, encoding="utf-8-sig")
        print(f"[save] {OUT_DIR / out_name}")

# ================== 主流程 ==================
def main():
    # 读插补后数据
    if not IMPUTED_XY.exists():
        raise FileNotFoundError(f"找不到插补后数据：{IMPUTED_XY}")
    df_imp = pd.read_csv(IMPUTED_XY)
    if Y_NAME not in df_imp.columns:
        raise KeyError(f"{Y_NAME} 不在 {IMPUTED_XY}")

    # 读原始（带缺失）Xy：可选
    raw_path = _first_exists(RAW_XY_CANDIDATES)
    if raw_path is None:
        print("[WARN] 找不到带缺失的原始 Xy，表 S1/S2 将跳过（仅生成 S3/S4）")
        df_raw = None
    else:
        df_raw = pd.read_csv(raw_path)

    # 对齐列
    if df_raw is not None:
        common_cols = [c for c in df_imp.columns if c in df_raw.columns]
        df_imp = df_imp[common_cols].copy()
        df_raw = df_raw[common_cols].copy()

    # 变量类型（优先 YAML）
    seed_num, seed_cat = read_yaml_types_from_cfg()
    num_cols, cat_cols = infer_types(df_imp, seed_num=seed_num, seed_cat=seed_cat)

    # 表 S1 / S2
    if df_raw is not None:
        tr_idx = load_idx(TRAIN_IDX)
        te_idx = load_idx(TEST_IDX)
        # 若索引缺失，退化为全体
        if tr_idx is None: tr_idx = np.arange(len(df_imp))
        if te_idx is None: te_idx = np.arange(len(df_imp))

        s1 = table_before_after(df_raw, df_imp, tr_idx, num_cols, cat_cols)
        s2 = table_before_after(df_raw, df_imp, te_idx, num_cols, cat_cols)

        s1.to_csv(OUT_DIR / "S1_before_after_train.csv", index=False, encoding="utf-8-sig")
        s2.to_csv(OUT_DIR / "S2_before_after_test.csv",  index=False, encoding="utf-8-sig")
        print(f"[save] {OUT_DIR/'S1_before_after_train.csv'}")
        print(f"[save] {OUT_DIR/'S2_before_after_test.csv'}")

    # 表 S3 / S4
    build_s3_s4()
    print("[done] S1–S4 tables ready.")

if __name__ == "__main__":
    main()
