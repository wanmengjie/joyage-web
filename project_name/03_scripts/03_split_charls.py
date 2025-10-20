# -*- coding: utf-8 -*-
"""
03_split_charls.py — 三划分：train/val/test（分层 + 可选分组）+ 可选年龄筛选 + 同时落盘索引与整行数据
运行：python 03_split_charls.py

依赖：
  - 07_config/config.yaml
  - CFG.paths.charls            指向 CHARLS 原始数据（csv, 必需）
  - CFG.paths.klosa             指向 KLOSA 外部数据（csv, 可选）
  - CFG.preprocess.age_filter   启用与阈值（enable/min/column）
  - CFG.outcome.{source,threshold,name}
  - CFG.split.{n_splits,val_size,test_size,strat_col,group_col}

产物：
  02_processed_data/<VER>/splits/
    ├─ charls_train.csv
    ├─ charls_val.csv
    ├─ charls_test.csv
    ├─ charls_train_idx.csv
    ├─ charls_val_idx.csv
    ├─ charls_test_idx.csv
    ├─ klosa_external.csv              # 若 CFG.paths.klosa 存在
    └─ splits_meta.json
"""

from __future__ import annotations
from pathlib import Path
import json, yaml, numpy as np, pandas as pd

# ---------------- 通用配置 ----------------
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

CFG = yaml.safe_load(open(repo_root() / "07_config" / "config.yaml", "r", encoding="utf-8"))

PROJECT_ROOT = Path(CFG["paths"]["project_root"])
VER_IN  = CFG.get("run_id_in", CFG.get("run_id"))
RS      = int(CFG.get("random_state", CFG.get("seed", 2025)))

# 输入：CHARLS（必需） / KLOSA（可选）
def _abs_or_join(base: Path, p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (base / pp)

CHARLS_CSV = _abs_or_join(PROJECT_ROOT, CFG["paths"]["charls"])
KLOSA_CSV  = _abs_or_join(PROJECT_ROOT, CFG["paths"]["klosa"]) if CFG["paths"].get("klosa") else None

OUT_DIR = PROJECT_ROOT / "02_processed_data" / VER_IN / "splits"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- 年龄筛选 ----------------
def _guess_age_column(df: pd.DataFrame, preferred: str | None):
    """优先用 preferred；否则在含 'age' 的列中自动识别一列看起来像年龄的列。"""
    if preferred and preferred in df.columns:
        s = pd.to_numeric(df[preferred], errors="coerce")
        return preferred, s

    candidates = ["agey", "ragey", "age", "age_years", "age_2018"]
    candidates += [c for c in df.columns if "age" in c.lower()]
    seen, best = set(), (None, None, -1.0)
    for c in candidates:
        if c in seen or c not in df.columns:
            continue
        seen.add(c)
        s = pd.to_numeric(df[c], errors="coerce")
        valid = s.notna().mean()
        if valid <= 0.30:
            continue
        rng_ok = ((s >= 10) & (s <= 120)).mean()
        score = float(valid + 0.1 * rng_ok)
        if score > best[2]:
            best = (c, s, score)
    return best[0], best[1]

def apply_age_filter(df: pd.DataFrame, cfg: dict, who: str) -> pd.DataFrame:
    """按配置过滤年龄（默认 enable=false；启用时默认 min=60）。"""
    af = (cfg.get("preprocess") or {}).get("age_filter") or {}
    if not af.get("enable", False):
        print(f"[age-filter] {who}: disabled -> skip")
        return df

    prefer  = af.get("column", "agey")
    min_age = float(af.get("min", 60))
    col, s  = _guess_age_column(df, preferred=prefer)

    if col is None:
        print(f"[age-filter] {who}: no age-like column found (preferred='{prefer}') -> skip")
        return df

    before = len(df)
    s_num  = pd.to_numeric(s, errors="coerce")
    mask   = s_num >= min_age
    df2    = df.loc[mask].copy()
    after  = len(df2)
    print(f"[age-filter] {who}: use '{col}', min={min_age} -> kept {after}/{before} (dropped {before - after})")
    return df2

# ---------------- 标签构建 ----------------
def build_binary_label(df: pd.DataFrame, outcome_cfg: dict) -> pd.DataFrame:
    y_src = outcome_cfg["source"]
    y_thr = outcome_cfg["threshold"]
    y_nm  = outcome_cfg["name"]
    df[y_src] = pd.to_numeric(df[y_src], errors="coerce")
    n0 = len(df)
    df = df.dropna(subset=[y_src]).copy()
    print(f"[label] dropna on '{y_src}': kept {len(df)}/{n0} (dropped {n0 - len(df)})")
    df[y_nm] = (df[y_src] >= y_thr).astype(int)
    return df

# ---------------- 加载数据并预处理 ----------------
df_charls = pd.read_csv(CHARLS_CSV)
print(f"[load] CHARLS n={len(df_charls)} | file={CHARLS_CSV}")
df_charls = apply_age_filter(df_charls, CFG, "CHARLS")
df_charls = build_binary_label(df_charls, CFG["outcome"])

df_klosa = None
if KLOSA_CSV and Path(KLOSA_CSV).exists():
    df_klosa = pd.read_csv(KLOSA_CSV)
    print(f"[load] KLOSA  n={len(df_klosa)} | file={KLOSA_CSV}")
    df_klosa = apply_age_filter(df_klosa, CFG, "KLOSA")
    df_klosa = build_binary_label(df_klosa, CFG["outcome"])

# ---------------- 生成三划分（分层 + 可选分组） ----------------
Y_NAME = CFG["outcome"]["name"]
STRAT  = (CFG.get("split", {}) or {}).get("strat_col") or Y_NAME
GROUP  = (CFG.get("split", {}) or {}).get("group_col")

X = df_charls.drop(columns=[Y_NAME])
y = df_charls[STRAT].astype(int).to_numpy()
groups = None
if GROUP and GROUP in df_charls.columns:
    groups = df_charls[GROUP].astype(str).to_numpy()
elif GROUP:
    print(f"[warn] group_col '{GROUP}' not in dataframe; fall back to no-group split")

N_SPLITS = int((CFG.get("split", {}) or {}).get("n_splits", 10))
VAL_SIZE = float((CFG.get("split", {}) or {}).get("val_size", 0.15))
TEST_SIZE= float((CFG.get("split", {}) or {}).get("test_size", 0.15))

from sklearn.model_selection import StratifiedKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold
    SGK = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RS) if groups is not None else None
except Exception:
    SGK = None

fold_indices: list[np.ndarray] = []
if groups is None or SGK is None:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RS)
    for _, te in skf.split(X, y):
        fold_indices.append(te)
else:
    for _, te in SGK.split(X, y, groups=groups):
        fold_indices.append(te)

n_test = max(1, round(TEST_SIZE * N_SPLITS))
n_val  = max(1, round(VAL_SIZE  * N_SPLITS))
order  = np.random.RandomState(RS).permutation(N_SPLITS)
test_folds = order[:n_test]
val_folds  = order[n_test:n_test + n_val]
train_folds= order[n_test + n_val:]

def collect(folds: np.ndarray) -> np.ndarray:
    if len(folds) == 0: return np.array([], dtype=int)
    idx = np.concatenate([fold_indices[i] for i in folds])
    return np.sort(idx)

tr_idx = collect(train_folds if len(train_folds)>0 else test_folds)  # 兜底
va_idx = collect(val_folds)
te_idx = collect(test_folds)

def _no_overlap(a, b) -> bool:
    return len(set(a) & set(b)) == 0

ok_overlap = _no_overlap(tr_idx, va_idx) and _no_overlap(tr_idx, te_idx) and _no_overlap(va_idx, te_idx)
print(f"[overlap] train/val/test disjoint = {ok_overlap}")

# ---------------- 保存索引与整行数据 ----------------
pd.DataFrame({"index": tr_idx}).to_csv(OUT_DIR / "charls_train_idx.csv", index=False)
pd.DataFrame({"index": va_idx}).to_csv(OUT_DIR / "charls_val_idx.csv",   index=False)
pd.DataFrame({"index": te_idx}).to_csv(OUT_DIR / "charls_test_idx.csv",  index=False)

# 确保排除cesd列，避免它被用作特征
Y_SRC = CFG["outcome"]["source"]  # 这是'cesd'
if Y_SRC in df_charls.columns:
    print(f"[info] 从数据集中排除原始标签源列 '{Y_SRC}'")
    df_charls = df_charls.drop(columns=[Y_SRC])

# 创建训练/验证/测试集
df_train = df_charls.iloc[tr_idx].reset_index(drop=True)
df_val   = df_charls.iloc[va_idx].reset_index(drop=True)
df_test  = df_charls.iloc[te_idx].reset_index(drop=True)

df_train.to_csv(OUT_DIR/"charls_train.csv", index=False)
df_val.to_csv(OUT_DIR/"charls_val.csv", index=False)
df_test.to_csv(OUT_DIR/"charls_test.csv", index=False)

if df_klosa is not None:
    # 同样确保排除KLOSA中的cesd列
    Y_SRC = CFG["outcome"]["source"]
    if Y_SRC in df_klosa.columns:
        print(f"[info] 从KLOSA数据集中排除原始标签源列 '{Y_SRC}'")
        df_klosa = df_klosa.drop(columns=[Y_SRC])
    
    (OUT_DIR/"klosa_external.csv").parent.mkdir(parents=True, exist_ok=True)
    df_klosa.to_csv(OUT_DIR/"klosa_external.csv", index=False)

meta = {
    "ver_in": VER_IN,
    "random_state": RS,
    "inputs": {
        "charls_csv": str(CHARLS_CSV),
        "klosa_csv": str(KLOSA_CSV) if KLOSA_CSV else None
    },
    "age_filter": (CFG.get("preprocess") or {}).get("age_filter"),
    "outcome": CFG.get("outcome"),
    "split": CFG.get("split"),
    "sizes": {
        "n_total_after_filters": int(len(df_charls)),
        "n_train": int(len(tr_idx)),
        "n_val": int(len(va_idx)),
        "n_test": int(len(te_idx)),
        "pos_rate_train": float(np.mean(df_train[Y_NAME])) if len(df_train) else None,
        "pos_rate_val":   float(np.mean(df_val[Y_NAME])) if len(df_val) else None,
        "pos_rate_test":  float(np.mean(df_test[Y_NAME])) if len(df_test) else None,
        "n_external": int(len(df_klosa)) if df_klosa is not None else 0
    }
}
with open(OUT_DIR / "splits_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

def _dist(name, df):
    pr = float(df[Y_NAME].mean()) if len(df)>0 else float("nan")
    print(f"{name:>5s} n={len(df):>6d}  pos_rate={pr:.3f}")

_dist("train", df_train)
_dist("val",   df_val)
_dist("test",  df_test)
print("Saved ->", OUT_DIR)
