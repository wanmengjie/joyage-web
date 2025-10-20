# -*- coding: utf-8 -*-
"""
08_freeze_mean_mode_merged.py — 合并 05 & 08（稳健版，列名规范+标签唯一）
"""
from __future__ import annotations
import json, re, shutil, yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

CFG = yaml.safe_load(open(repo_root()/ "07_config"/"config.yaml", "r", encoding="utf-8"))

PROJ = Path(CFG["paths"]["project_root"])
VER_IN  = CFG.get("run_id_in",  CFG.get("run_id"))
VER_OUT = CFG.get("run_id_out", CFG.get("run_id"))

DATA_DIR    = PROJ / "02_processed_data" / VER_IN
IMP_DIR     = DATA_DIR / "imputed"
IMPTR_DIR   = DATA_DIR / "imputers"
SEL_MM_DIR  = DATA_DIR / "selected_imputation" / "mean_mode"
SEL_PMM_DIR = DATA_DIR / "selected_imputation" / "mice_pmm"
FROZEN_DIR  = DATA_DIR / "frozen"
WITHNAN_DIR = DATA_DIR / "with_nan"
EXP_DIR     = PROJ / "10_experiments" / VER_OUT
SUPP_DIR    = PROJ / "05_supplement" / VER_OUT

for p in [SEL_MM_DIR, SEL_PMM_DIR, FROZEN_DIR, WITHNAN_DIR, EXP_DIR, SUPP_DIR, IMPTR_DIR]:
    p.mkdir(parents=True, exist_ok=True)

IDS    = CFG.get("ids", ["ID", "householdID"])
Y_SRC  = CFG["outcome"]["source"]
Y_NAME = CFG["outcome"]["name"]          # 想要的最终标签列名（例如 'depression'）
Y_THR  = float(CFG["outcome"]["threshold"])
ENC    = "utf-8-sig"

# ---------- helpers ----------
def _abs_or_join(p_like: str|Path) -> Path:
    p = Path(p_like); return p if p.is_absolute() else (PROJ / p)

def _guess_age_column(df: pd.DataFrame, preferred: str | None):
    if preferred and preferred in df.columns:
        s = pd.to_numeric(df[preferred], errors="coerce"); return preferred, s
    cands = ["agey","ragey","age","age_years","age_2018"] + [c for c in df.columns if "age" in c.lower()]
    seen, best = set(), (None, None, -1.0)
    for c in cands:
        if c in seen or c not in df.columns: continue
        seen.add(c)
        s = pd.to_numeric(df[c], errors="coerce")
        valid = s.notna().mean()
        if valid <= 0.30: continue
        rng_ok = ((s >= 40) & (s <= 120)).mean()
        score = float(valid + 0.1 * rng_ok)
        if score > best[2]: best = (c, s, score)
    return best[0], best[1]

def apply_age_filter(df: pd.DataFrame) -> pd.DataFrame:
    af = (CFG.get("preprocess") or {}).get("age_filter") or {}
    if not af.get("enable", False):
        print("[age-filter] disabled -> skip"); return df
    prefer  = af.get("column","agey")
    min_age = float(af.get("min",60))
    col, s  = _guess_age_column(df, preferred=prefer)
    if col is None:
        print(f"[age-filter] no age-like column (preferred='{prefer}') -> skip"); return df
    before = len(df)
    mask = pd.to_numeric(s, errors="coerce") >= min_age
    df2 = df.loc[mask].copy()
    print(f"[age-filter] use '{col}', min={min_age} -> kept {len(df2)}/{before}")
    return df2

def _is_binary_series(s: pd.Series) -> bool:
    s2 = pd.to_numeric(s, errors="coerce").dropna()
    if s2.empty: return False
    uniq = pd.unique(s2.astype(float))
    return set(uniq).issubset({0.0, 1.0}) and len(uniq) <= 2

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # 去首尾空格，保留原顺序
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df

def _label_like_columns(cols) -> list[str]:
    out=[]
    for c in cols:
        c0=c.strip()
        if re.fullmatch(r'(?i)(depressed|depression|depression[_\-\.\s]*bin)(?:\.\d+)?', c0):
            out.append(c0)
    return out

def build_outcome(df: pd.DataFrame) -> pd.DataFrame:
    """
    - 规范列名
    - 若 source 为二元：直接复制 → Y_NAME
    - 否则用阈值二分 → Y_NAME
    - 清理所有标签相似列（仅保留 Y_NAME）
    """
    df = _normalize_columns(df)
    if Y_SRC not in df.columns:
        raise KeyError(f"outcome.source '{Y_SRC}' 不在原始表中（可用列示例：{df.columns[:10].tolist()} …）")

    s = pd.to_numeric(df[Y_SRC], errors="coerce")
    df = df.loc[s.notna()].copy()

    if _is_binary_series(s):
        df[Y_NAME] = s.astype(int)
    else:
        df[Y_NAME] = (s >= Y_THR).astype(int)

    # 删除其它“标签相似名”的列，确保只留 Y_NAME
    labelish = _label_like_columns(df.columns)
    for c in labelish:
        if c != Y_NAME:
            df.drop(columns=c, inplace=True, errors="ignore")

    # 二分类校验
    if df[Y_NAME].dropna().nunique() != 2:
        vc = df[Y_NAME].value_counts(dropna=False).to_dict()
        raise ValueError(f"{Y_NAME} 不是二分类（分布={vc}）。请检查 outcome.source/threshold 或预处理筛选。")
    return df

def coerce_numeric_like(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        s = out[c]
        if not pd.api.types.is_numeric_dtype(s):
            s2 = s.astype(str).str.strip()
            num = s2.str.extract(r"([-+]?\d*\.?\d+)", expand=False)
            if num.notna().mean() > 0.5:
                out[c] = pd.to_numeric(num, errors="coerce")
    return out

def _safe_read_07_table(filename: str):
    for fn in [filename, filename.replace("impute_model_selection_", "impute_selection_")]:
        p = EXP_DIR / fn
        if p.exists():
            df = pd.read_csv(p)
            if "method" in df.columns:
                df["method"] = df["method"].replace({"mm":"mean_mode"})
            return df
    return None

def _pick_mm_with_scenario(df: pd.DataFrame | None, scen: str):
    if df is None or "method" not in df.columns: return None
    x = df[df["method"]=="mean_mode"].copy()
    if x.empty: return None
    if "scenario" in x.columns: x["scenario"] = scen
    else: x.insert(0,"scenario",scen)
    return x

def _train_stats(df_train: pd.DataFrame, ref_cols: list[str]):
    num_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in ref_cols if c not in num_cols]
    num_map, cat_map = {}, {}
    for c in num_cols:
        v=float(pd.to_numeric(df_train[c], errors="coerce").median(skipna=True))
        num_map[c] = 0.0 if np.isnan(v) else v
    for c in cat_cols:
        md = pd.Series(df_train[c]).mode(dropna=True)
        cat_map[c] = (md.iloc[0] if not md.empty else "missing")
    return num_cols, cat_cols, num_map, cat_map

def _fill_by_maps(X_raw: pd.DataFrame, num_cols: list[str], cat_cols: list[str], num_map: dict, cat_map: dict) -> pd.DataFrame:
    X = X_raw.copy()
    for c in num_cols:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(num_map[c])
    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].astype(object).where(~pd.isna(X[c]), cat_map[c]).fillna(cat_map[c])
    return X

def _triplet_exists(dir_path: Path) -> bool:
    return all((dir_path / f"charls_{sp}.csv").exists() for sp in ("train","val","test"))

def _find_reference_triplet() -> Tuple[str, Path]:
    if _triplet_exists(SEL_MM_DIR):  return "sel_mean_mode", SEL_MM_DIR
    if _triplet_exists(SEL_PMM_DIR): return "sel_mice_pmm", SEL_PMM_DIR

    all_xy = list(IMP_DIR.glob("charls_*_*_Xy.csv"))
    by_method: Dict[str, Dict[str, Path]] = {}
    pat = re.compile(r"^charls_(train|val|test)_(.+)_Xy\.csv$", re.IGNORECASE)
    for p in all_xy:
        m = pat.match(p.name)
        if not m: continue
        sp, method = m.group(1).lower(), m.group(2)
        by_method.setdefault(method, {})[sp] = p
    for method, parts in by_method.items():
        if {"train","val","test"}.issubset(parts.keys()):
            tmp = DATA_DIR / "_tmp_ref_triplet"; tmp.mkdir(exist_ok=True, parents=True)
            for sp in ("train","val","test"):
                shutil.copy2(parts[sp], tmp / f"charls_{sp}.csv")
            return f"imputed_{method}", tmp

    raise FileNotFoundError("未找到参考三件套（mean_mode / mice_pmm / 任一 *_Xy 三件套）。")

# ---------- main ----------
if __name__ == "__main__":
    print("=========== MERGED FREEZE (08 + 05) START ===========")
    print(f"[info] VER_IN={VER_IN}  VER_OUT={VER_OUT}")

    mm_frames=[]
    for fn, scen in [
        ("impute_model_selection_internal_val.csv",   "internal_val"),
        ("impute_model_selection_internal_test.csv",  "internal_test"),
        ("impute_model_selection_external_main.csv",  "external_main"),
        ("impute_model_selection_external_transfer.csv","external_transfer"),
    ]:
        df = _safe_read_07_table(fn)
        pick = _pick_mm_with_scenario(df, scen)
        if pick is not None: mm_frames.append(pick)
    if mm_frames:
        mm_all = pd.concat(mm_frames, ignore_index=True)
        keep = [c for c in ["scenario","method","AUROC","AUPRC","Brier_Score","Cal_intercept","Cal_slope","LogLoss"] if c in mm_all.columns]
        mm_all[keep].to_csv(EXP_DIR/"mean_mode_summary.csv", index=False, encoding=ENC)
        print(f"[ok] summary -> {EXP_DIR/'mean_mode_summary.csv'}")
    else:
        print("[warn] no mean_mode rows from 07 tables (skip summary)")

    src_tag, ref_dir = _find_reference_triplet()
    print(f"[ref] use triplet from: {src_tag} -> {ref_dir}")

    if src_tag != "sel_mean_mode":
        for sp in ("train","val","test"):
            src = ref_dir / f"charls_{sp}.csv"
            dst = SEL_MM_DIR / f"charls_{sp}.csv"
            if not dst.exists() or str(src.resolve()) != str(dst.resolve()):
                shutil.copy2(src, dst)
        print(f"[ok] mirrored ref triplet -> {SEL_MM_DIR}")

    # 参考 train（也规范列名）
    Xtr_imp = _normalize_columns(pd.read_csv(SEL_MM_DIR / "charls_train.csv"))
    
    # 确保从参考列中排除cesd和其他应排除列
    EXCLUDE_FROM_FEATURES = set(IDS + [Y_SRC, Y_NAME]) | set((CFG.get("preprocess") or {}).get("drop_cols", []))
    ref_cols = [c for c in Xtr_imp.columns.tolist() if c not in EXCLUDE_FROM_FEATURES]
    print(f"[info] 从参考特征列表中排除了这些列: {EXCLUDE_FROM_FEATURES}")

    num_cols_train, cat_cols_train, num_stat_map, cat_stat_map = _train_stats(Xtr_imp, ref_cols)

    paths = CFG["paths"]
    df_ch = _normalize_columns(pd.read_csv(_abs_or_join(paths["charls"])))
    df_ch = apply_age_filter(df_ch)
    df_ch = build_outcome(df_ch)

    df_kl = None
    if paths.get("klosa"):
        p_kl = _abs_or_join(paths["klosa"])
        if p_kl.exists():
            df_kl = _normalize_columns(pd.read_csv(p_kl))
            df_kl = apply_age_filter(df_kl)
            df_kl = build_outcome(df_kl)

    EXCLUDE = set(IDS + [Y_SRC, Y_NAME]) | set((CFG.get("preprocess") or {}).get("drop_cols", []))
    X_ch_raw = df_ch[[c for c in df_ch.columns if c not in EXCLUDE]].copy()
    X_ch_raw = coerce_numeric_like(X_ch_raw)
    for c in ref_cols:
        if c not in X_ch_raw.columns: X_ch_raw[c] = np.nan
    X_ch_raw = X_ch_raw.reindex(columns=ref_cols)
    y_ch = df_ch[[Y_NAME]].reset_index(drop=True)

    with_nan_xy = pd.concat([X_ch_raw.reset_index(drop=True), y_ch], axis=1)
    WITHNAN_DIR.mkdir(parents=True, exist_ok=True)
    with_nan_xy.to_csv(WITHNAN_DIR / "charls_Xy_with_nan.csv", index=False, encoding=ENC)
    print(f"[ok] with_nan -> {WITHNAN_DIR/'charls_Xy_with_nan.csv'} (rows={len(with_nan_xy)})")

    X_ch_imp = _fill_by_maps(X_ch_raw, num_cols_train, cat_cols_train, num_stat_map, cat_stat_map).reset_index(drop=True)
    Xy_ch = pd.concat([X_ch_imp, y_ch], axis=1)

    FROZEN_DIR.mkdir(parents=True, exist_ok=True)
    X_ch_imp.to_csv(FROZEN_DIR / "charls_mean_mode_X.csv", index=False, encoding=ENC)
    y_ch.to_csv(  FROZEN_DIR / "charls_mean_mode_y.csv", index=False, encoding=ENC)
    Xy_ch.to_csv( FROZEN_DIR / "charls_mean_mode_Xy.csv", index=False, encoding=ENC)
    print(f"[ok] frozen CHARLS -> {FROZEN_DIR/'charls_mean_mode_Xy.csv'} (rows={len(Xy_ch)})")

    stats = {"center": "median(mode for cats)", "num_map": num_stat_map, "cat_map": cat_stat_map, "ref_cols": ref_cols}
    (IMPTR_DIR / "mean_mode_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] stats -> {IMPTR_DIR/'mean_mode_stats.json'}")

    if df_kl is not None:
        X_kl_raw = df_kl[[c for c in df_kl.columns if c not in EXCLUDE]].copy()
        X_kl_raw = coerce_numeric_like(X_kl_raw)
        for c in ref_cols:
            if c not in X_kl_raw.columns: X_kl_raw[c] = np.nan
        X_kl_raw = X_kl_raw.reindex(columns=ref_cols)
        y_kl = df_kl[[Y_NAME]].reset_index(drop=True)

        X_kl_imp = _fill_by_maps(X_kl_raw, num_cols_train, cat_cols_train, num_stat_map, cat_stat_map).reset_index(drop=True)
        Xy_kl = pd.concat([X_kl_imp, y_kl], axis=1)

        X_kl_imp.to_csv(FROZEN_DIR / "klosa_transfer_X.csv", index=False, encoding=ENC)
        y_kl.to_csv(  FROZEN_DIR / "klosa_transfer_y.csv", index=False, encoding=ENC)
        Xy_kl.to_csv( FROZEN_DIR / "klosa_transfer_Xy.csv", index=False, encoding=ENC)
        print(f"[ok] frozen KLOSA -> {FROZEN_DIR/'klosa_transfer_Xy.csv'} (rows={len(Xy_kl)})")

    meta = {
        "chosen_imputation": "mean_mode",
        "ref_triplet_source": src_tag,
        "selected_imputation_dir": str(SEL_MM_DIR),
        "frozen": {
            "charls_mean_mode_X":  str(FROZEN_DIR/"charls_mean_mode_X.csv"),
            "charls_mean_mode_y":  str(FROZEN_DIR/"charls_mean_mode_y.csv"),
            "charls_mean_mode_Xy": str(FROZEN_DIR/"charls_mean_mode_Xy.csv"),
            "klosa_transfer_X":    str(FROZEN_DIR/"klosa_transfer_X.csv") if (FROZEN_DIR/"klosa_transfer_X.csv").exists() else None,
            "klosa_transfer_y":    str(FROZEN_DIR/"klosa_transfer_y.csv") if (FROZEN_DIR/"klosa_transfer_y.csv").exists() else None,
            "klosa_transfer_Xy":   str(FROZEN_DIR/"klosa_transfer_Xy.csv") if (FROZEN_DIR/"klosa_transfer_Xy.csv").exists() else None
        },
        "with_nan": {"charls_Xy_with_nan": str(WITHNAN_DIR/"charls_Xy_with_nan.csv")}
    }
    (EXP_DIR/"chosen_imputation.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] meta -> {EXP_DIR/'chosen_imputation.json'}")
    print("=========== MERGED FREEZE END ========================")
