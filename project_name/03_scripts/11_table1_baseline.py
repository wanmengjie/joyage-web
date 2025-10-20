# -*- coding: utf-8 -*-
r"""
11_table1_baseline.py — 基线特征对比表（CHARLS vs KLOSA，基于冻结后的 mean_mode / transfer）
改进点：
- 统一配置读取（project_root / run_id_in / run_id_out）
- 更稳健的标签标准化与数值识别
- 分类：2x2 用 Fisher / 期望充足时卡方；多水平用卡方
- 连续：Mann–Whitney U；同时给出 SMD；分类给 Cramér’s V
- 导出 CSV / XLSX / LaTeX 到 10_experiments/<VER_OUT>/table1/
"""

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu

warnings.filterwarnings("ignore")

# ---------------- paths & config ----------------
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

CFG = yaml.safe_load(open(repo_root() / "07_config" / "config.yaml", "r", encoding="utf-8"))

PROJECT_ROOT = Path(CFG["paths"]["project_root"])
VER_IN  = CFG.get("run_id_in",  CFG.get("run_id"))
VER_OUT = CFG.get("run_id_out", CFG.get("run_id"))

IN_DIR  = PROJECT_ROOT / "02_processed_data" / VER_IN / "frozen"
OUT_DIR = PROJECT_ROOT / "10_experiments" / VER_OUT / "table1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHARLS_FILE = IN_DIR / "charls_mean_mode_Xy.csv"
KLOSA_FILE  = IN_DIR / "klosa_transfer_Xy.csv"

Y_NAME = CFG["outcome"]["name"]  # 一般为 depression_bin / y

# ---------------- variable lists ----------------
# 备注：把 ramomeducl、radadeducl、peninc 视为“分类变量”
CATEGORICAL_VARS = [
    "ragender","raeducl","ramomeducl","radadeducl","mstath","rural","shlta",
    "hlthlm","adlfive","hibpe","diabe","cancre","lunge","hearte","stroke","arthre",
    "livere","painfr","stayhospital","fall","hipriv",
    "drinkev","smokev","socwk","work","momliv","dadliv","lvnear","kcntf","ftrsp","ftrkids",
    "pubpen","peninc"
]
CONTINUOUS_VARS = [
    "agey","child","hhres","comparable_hexp","comparable_exp","comparable_frec",
    "comparable_itearn","comparable_ipubpen","comparable_tgiv"
]
DOMAINS = {
    "Demographic": ["ragender","agey","raeducl","ramomeducl","radadeducl","mstath","rural","child","hhres"],
    "Health Status and Medical-Related Factors": [
        "shlta","hlthlm","adlfive","hibpe","diabe","cancre","lunge","hearte","stroke","arthre",
        "livere","painfr","stayhospital","fall","hipriv"
    ],
    "Lifestyle and Behaviors": ["drinkev","smokev","socwk","work"],
    "Family Relationships and Social Support": ["momliv","dadliv","lvnear","kcntf","ftrsp","ftrkids"],
    "Economic Status": ["pubpen","peninc","comparable_hexp","comparable_exp","comparable_frec",
                        "comparable_itearn","comparable_ipubpen","comparable_tgiv"]
}

# ---------------- helpers ----------------
def _as_num(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def coerce_numeric_like(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """把“看起来像数字”的字符串转为数值（列内可识别比例 > threshold 才转换），避免误伤分类变量。"""
    out = df.copy()
    for c in out.columns:
        s = out[c]
        if pd.api.types.is_numeric_dtype(s):
            continue
        s2 = s.astype(str).str.strip()
        num = s2.str.extract(r"([-+]?\d*\.?\d+)", expand=False)
        if num.notna().mean() > threshold:
            out[c] = pd.to_numeric(num, errors="coerce")
    return out

def standardize_levels(df: pd.DataFrame) -> pd.DataFrame:
    """把常见数值编码恢复成人类可读标签；未覆盖的维持原样。"""
    out = df.copy()

    def map_binary(col, yes="Yes", no="No"):
        if col not in out.columns: 
            return
        s = out[col].astype(str)
        # 优先识别“纯 0/1 或前缀是 0/1 的值”
        vals = s.str.extract(r"([-+]?\d*\.?\d+)")[0].map(_as_num)
        mask = vals.isin([0.0, 1.0])
        out.loc[mask, col] = vals.map({0.0: no, 1.0: yes})

    def map_edu3(col):
        if col not in out.columns:
            return
        v = out[col].astype(str).str.extract(r"(\d+)")[0].map(_as_num)
        m = v.isin([1.0, 2.0, 3.0])
        out.loc[m, col] = v.map({
            1.0: "Less than upper secondary",
            2.0: "Upper secondary/Vocational",
            3.0: "Tertiary education"
        }).fillna(out[col])

    # 性别 1/2 -> Male/Female
    if "ragender" in out.columns:
        v = out["ragender"].astype(str).str.extract(r"(\d+)")[0].map(_as_num)
        m = v.isin([1.0, 2.0])
        out.loc[m, "ragender"] = v.map({1.0: "Male", 2.0: "Female"}).fillna(out["ragender"])

    # 教育（本人/母亲/父亲）
    map_edu3("raeducl"); map_edu3("ramomeducl"); map_edu3("radadeducl")

    # 婚姻
    if "mstath" in out.columns:
        v = out["mstath"].astype(str).str.extract(r"(\d+)")[0].map(_as_num)
        mapping = {
            1.0: "Married/Partnered",
            2.0: "Married, spouse absent",
            4.0: "Separated",
            5.0: "Divorced",
            7.0: "Widowed",
            8.0: "Never married"
        }
        out["mstath"] = v.map(mapping).fillna(out["mstath"])

    # 城乡
    if "rural" in out.columns:
        v = out["rural"].astype(str).str.extract(r"(\d+)")[0].map(_as_num)
        m = v.isin([0.0, 1.0])
        out.loc[m, "rural"] = v.map({0.0: "Urban", 1.0: "Rural"}).fillna(out["rural"])

    # 自评健康（1~5）
    if "shlta" in out.columns:
        v = out["shlta"].astype(str).str.extract(r"(\d+)")[0].map(_as_num)
        mapping = {1.0:"Excellent",2.0:"Very good",3.0:"Good",4.0:"Fair",5.0:"Poor"}
        out["shlta"] = v.map(mapping).fillna(out["shlta"])

    # 二元类变量统一为 Yes/No（除上面单独映射过的）
    bin_vars = [v for v in CATEGORICAL_VARS if v in out.columns and v not in
                ["ragender","raeducl","ramomeducl","radadeducl","mstath","rural","shlta"]]
    for v in bin_vars:
        map_binary(v)

    return out

# ---- summaries & tests ----
def summarize_continuous(df, var):
    if var not in df.columns: return "N/A"
    vals = pd.to_numeric(df[var], errors="coerce").dropna()
    if len(vals) == 0: return "N/A"
    med = vals.median(); q25 = vals.quantile(0.25); q75 = vals.quantile(0.75)
    return f"{med:.1f} ({q25:.1f}–{q75:.1f})"

def summarize_categorical(df, var, level_order="desc"):
    """返回 {level: 'n (p%)'}，按频数降序或字母序。"""
    if var not in df.columns: return {}
    s = df[var].astype("string")
    vc = s.value_counts(dropna=True)
    if level_order == "alpha":
        vc = vc.sort_index()
    total = (s.notna()).sum()
    res = {}
    for level, cnt in vc.items():
        pct = 100.0 * cnt / total if total > 0 else 0.0
        res[str(level)] = f"{int(cnt)} ({pct:.1f}%)"
    return res

def p_value_continuous(df1, df2, var):
    if var not in df1.columns or var not in df2.columns: return np.nan
    v1 = pd.to_numeric(df1[var], errors="coerce").dropna()
    v2 = pd.to_numeric(df2[var], errors="coerce").dropna()
    if len(v1) == 0 or len(v2) == 0: return np.nan
    try:
        _, p = mannwhitneyu(v1, v2, alternative="two-sided")
        return float(p)
    except Exception:
        return np.nan

def _cont_table_2xn(s1: pd.Series, s2: pd.Series, levels: list) -> np.ndarray:
    return np.array([[(s1 == lv).sum(), (s2 == lv).sum()] for lv in levels])

def p_value_categorical(df1, df2, var):
    if var not in df1.columns or var not in df2.columns: return np.nan
    s1 = df1[var].astype("string"); s2 = df2[var].astype("string")
    levels = sorted(set(s1.dropna().unique()) | set(s2.dropna().unique()))
    if len(levels) < 2: return np.nan
    cont = _cont_table_2xn(s1, s2, levels)  # shape = (L, 2)
    if cont.sum() == 0: return np.nan
    try:
        if cont.shape == (2, 2):
            # 先看期望频数决定 Fisher or 卡方
            chi2, p_chi, _, exp = chi2_contingency(cont, correction=False)
            if (exp >= 5).all():
                return float(p_chi)
            else:
                _, p_f = fisher_exact(cont)
                return float(p_f)
        else:
            _, p, _, _ = chi2_contingency(cont, correction=False)
            return float(p)
    except Exception:
        return np.nan

# ---- effect sizes ----
def smd_continuous(df1, df2, var) -> float:
    """标准化均值差（绝对值），若有缺失或方差为 0 则返回 NaN。"""
    if var not in df1.columns or var not in df2.columns: return np.nan
    x = pd.to_numeric(df1[var], errors="coerce").dropna()
    y = pd.to_numeric(df2[var], errors="coerce").dropna()
    if len(x) < 2 or len(y) < 2: return np.nan
    m1, m2 = x.mean(), y.mean()
    s1, s2 = x.std(ddof=1), y.std(ddof=1)
    sp2 = ((len(x)-1)*s1**2 + (len(y)-1)*s2**2) / max(len(x)+len(y)-2, 1)
    if sp2 <= 0: return np.nan
    return float(abs(m1 - m2) / math.sqrt(sp2))

def cramers_v(df1, df2, var) -> float:
    """Cramér's V（分类效应量）。"""
    if var not in df1.columns or var not in df2.columns: return np.nan
    s1 = df1[var].astype("string"); s2 = df2[var].astype("string")
    levels = sorted(set(s1.dropna().unique()) | set(s2.dropna().unique()))
    if len(levels) < 2: return np.nan
    cont = _cont_table_2xn(s1, s2, levels)  # (L, 2)
    if cont.sum() == 0: return np.nan
    try:
        chi2, _, _, _ = chi2_contingency(cont, correction=False)
        n = cont.sum()
        r, k = cont.shape  # r=L, k=2
        denom = n * (min(r, k) - 1)
        if denom <= 0: return np.nan
        return float(math.sqrt(chi2 / denom))
    except Exception:
        return np.nan

def fmt_p(p):
    if pd.isna(p): return ""
    return "<0.001" if p < 0.001 else f"{p:.3f}"

def fmt_eff(x):
    if pd.isna(x): return ""
    return f"{x:.3f}"

# ---------------- main flow ----------------
def main():
    # 读冻结数据
    ch_raw = pd.read_csv(CHARLS_FILE)
    kl_raw = pd.read_csv(KLOSA_FILE)

    # 仅保留分析需要的列
    use_cols = list(set(CATEGORICAL_VARS + CONTINUOUS_VARS + [Y_NAME]))
    ch = ch_raw[[c for c in ch_raw.columns if c in use_cols]].copy()
    kl = kl_raw[[c for c in kl_raw.columns if c in use_cols]].copy()

    # 轻量数值化（不影响分类变量；只是把看起来像数的东西转成数，便于连续型汇总）
    ch = coerce_numeric_like(ch)
    kl = coerce_numeric_like(kl)

    # 标签标准化（把 0/1 等映射成人类可读标签）
    ch_std = standardize_levels(ch)
    kl_std = standardize_levels(kl)

    # 基线表
    rows = []
    n_ch, n_kl = len(ch_std), len(kl_std)

    for domain, vars_in_dom in DOMAINS.items():
        # 域标题行
        rows.append({
            "Characteristic": f"[{domain}]",
            "Level": "",
            "CHARLS_2018": "",
            "KLOSA_2018": "",
            "P_value": "",
            "Effect_size": ""
        })

        for v in vars_in_dom:
            present = (v in ch_std.columns) or (v in kl_std.columns)
            if not present:
                continue

            if v in CONTINUOUS_VARS:
                # 连续变量：用中位数(IQR)，Mann–Whitney U，SMD
                ch_s = summarize_continuous(ch_std, v)
                kl_s = summarize_continuous(kl_std, v)
                p = p_value_continuous(ch_std, kl_std, v)
                eff = smd_continuous(ch_std, kl_std, v)
                rows.append({
                    "Characteristic": v,
                    "Level": "",
                    "CHARLS_2018": ch_s,
                    "KLOSA_2018": kl_s,
                    "P_value": fmt_p(p),
                    "Effect_size": fmt_eff(eff)
                })
            else:
                # 分类变量：按频数降序展示每个 level；2×2 用 Fisher/卡方，>2 水平用卡方；效应量 Cramér’s V
                ch_s = summarize_categorical(ch_std, v, level_order="desc")
                kl_s = summarize_categorical(kl_std, v, level_order="desc")
                p = p_value_categorical(ch_std, kl_std, v)
                eff = cramers_v(ch_std, kl_std, v)

                levels = list(dict.fromkeys(list(ch_s.keys()) + list(kl_s.keys())))  # 合并保持降序
                if not levels:
                    # 没有有效水平，补一行占位
                    rows.append({
                        "Characteristic": v, "Level": "", "CHARLS_2018": "", "KLOSA_2018": "",
                        "P_value": "", "Effect_size": ""
                    })
                    continue

                for i, lev in enumerate(levels):
                    rows.append({
                        "Characteristic": v if i == 0 else "",
                        "Level": lev,
                        "CHARLS_2018": ch_s.get(lev, "0 (0.0%)"),
                        "KLOSA_2018":  kl_s.get(lev, "0 (0.0%)"),
                        "P_value": fmt_p(p) if i == 0 else "",
                        "Effect_size": fmt_eff(eff) if i == 0 else ""
                    })

    table = pd.DataFrame(rows)
    table.columns = [
        "Characteristic", "Level",
        f"CHARLS_2018 (N={n_ch:,})",
        f"KLOSA_2018 (N={n_kl:,})",
        "P_value", "Effect_size"
    ]

    # 导出
    csv_path  = OUT_DIR / "table1_baseline.csv"
    xlsx_path = OUT_DIR / "table1_baseline.xlsx"
    tex_path  = OUT_DIR / "table1_baseline.tex"

    table.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(xlsx_path) as w:
        table.to_excel(w, sheet_name="Table1", index=False)

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(table.to_latex(index=False, escape=False))

    print("Saved:")
    print(" -", csv_path)
    print(" -", xlsx_path)
    print(" -", tex_path)

if __name__ == "__main__":
    main()
