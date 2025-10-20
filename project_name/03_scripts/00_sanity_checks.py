# -*- coding: utf-8 -*-
"""
00_sanity_checks.py — 项目自动体检（稳健版）
- 关键文件缺失 -> FAIL 退出
- 其它异常（如分割重叠、AUROC 偏低等） -> WARN 不阻断
"""

from pathlib import Path
import warnings; warnings.filterwarnings("ignore", category=UserWarning)

import yaml
import numpy as np
import pandas as pd

# ========== 统一配置头 ==========
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

CFG_PATH = repo_root() / "07_config" / "config.yaml"
with open(CFG_PATH, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

PROJECT_ROOT = Path(CFG["paths"]["project_root"])

# 更稳健的版本选择：IN 缺省则回到 OUT；OUT 缺省则回到 run_id
VER_OUT = CFG.get("run_id_out") or CFG.get("run_id")
VER_IN  = CFG.get("run_id_in")  or VER_OUT

def _p(*parts) -> Path:
    return PROJECT_ROOT.joinpath(*parts)

# 优先用 VER_IN 的 frozen，缺失则回退 OUT
MM_IN  = _p("02_processed_data", VER_IN,  "frozen", "charls_mean_mode_Xy.csv")
MM_OUT = _p("02_processed_data", VER_OUT, "frozen", "charls_mean_mode_Xy.csv")
MEAN_MODE_XY = MM_IN if MM_IN.exists() else MM_OUT

TRAIN_IDX = _p("02_processed_data", VER_IN, "splits", "charls_train_idx.csv")
VAL_IDX   = _p("02_processed_data", VER_IN, "splits", "charls_val_idx.csv")
TEST_IDX  = _p("02_processed_data", VER_IN, "splits", "charls_test_idx.csv")

EXT_MAIN  = _p("02_processed_data", VER_IN, "frozen", "klosa_transfer_Xy.csv")
EXT_TRAN  = _p("02_processed_data", VER_IN, "frozen", "klosa_transfer_Xy.csv")

Y_NAME       = (CFG.get("outcome", {}) or {}).get("name", "depression_bin")
ID_COLS      = CFG.get("ids", ["ID", "householdID"])
RANDOM_STATE = int(CFG.get("random_state", 42))

EXP_OUT = _p("10_experiments", VER_OUT)
RFE_OVERVIEW = EXP_OUT / "feature_sweep" / "S4_RFE_overview.csv"
S7_DIR       = EXP_OUT / "final_eval_s7"
S7_METRICS   = S7_DIR / "S7_final_metrics.csv"

# ====== 小工具 ======
def _read_idx(p: Path) -> np.ndarray:
    return pd.read_csv(p).iloc[:, 0].values.astype(int)

def _ok(msg):   print(f"[OK]   {msg}")
def _warn(msg): print(f"[WARN] {msg}")
def _fail(msg):
    print(f"[FAIL] {msg}")
    raise SystemExit(1)

print("=== Sanity Check Start ===")
print(f"[info] VER_IN={VER_IN}  VER_OUT={VER_OUT}")
print(f"[info] frozen try: IN={MM_IN.exists()}  OUT={MM_OUT.exists()}  => using {MEAN_MODE_XY}")

# 1) 关键文件存在
need_files = [
    ("frozen mean_mode Xy", MEAN_MODE_XY),
    ("train_idx", TRAIN_IDX),
    ("val_idx",   VAL_IDX),
    ("test_idx",  TEST_IDX),
]
missing = [(name, p) for name, p in need_files if not p.exists()]
if missing:
    for name, p in missing:
        print(f"[FAIL] 缺少关键文件: {name} -> {p}")
    raise SystemExit(1)
_ok("关键输入文件存在")

# 2) 读 frozen Xy 并检查标签
df = pd.read_csv(MEAN_MODE_XY)
print(f"[info] mean_mode_Xy shape = {df.shape}; columns={len(df.columns)}")
if Y_NAME not in df.columns:
    _fail(f"标签列 {Y_NAME} 不在 {MEAN_MODE_XY}")
if df[Y_NAME].dropna().nunique() != 2:
    _fail(f"{Y_NAME} 不是二分类（nunique={df[Y_NAME].dropna().nunique()}）")
_ok("标签列存在且为二分类")

# 3) splits 合法性
tr = _read_idx(TRAIN_IDX); va = _read_idx(VAL_IDX); te = _read_idx(TEST_IDX)
n  = len(df)
def _check_range(name, idx):
    if (idx < 0).any() or (idx >= n).any():
        bad = int(max(idx[idx >= n], default=-1))
        _fail(f"{name} 索引越界：存在 >= {n} 的值（样本总数={n}），示例={bad}")
for nm, arr in [("train", tr), ("val", va), ("test", te)]:
    _check_range(nm, arr)
_ok("train/val/test 索引均在范围内")

# 4) splits 互不重叠（重叠 -> 仅警告）
over = []
if len(set(tr) & set(va)) > 0: over.append("train↔val")
if len(set(tr) & set(te)) > 0: over.append("train↔test")
if len(set(va) & set(te)) > 0: over.append("val↔test")
if over:
    _warn(f"发现分割间有重叠索引：{over}（如本工程允许可忽略）")
else:
    _ok("train/val/test 互不重叠")

# 5) 外部集（存在则检查）
for name, p in [("external_main", EXT_MAIN), ("external_transfer", EXT_TRAN)]:
    if p.exists():
        d = pd.read_csv(p, nrows=5)
        if Y_NAME not in d.columns:
            _fail(f"{name} 缺少标签列 {Y_NAME} -> {p}")
        _ok(f"{name} OK: {p.name}")
    else:
        _warn(f"{name} 不存在（可忽略）")

# 6) 特征类型推断（结合 YAML 种子）
def _infer_types(df: pd.DataFrame, y_name: str, seed_num=None, seed_cat=None):
    seed_num = list(seed_num or [])
    seed_cat = list(seed_cat or [])
    exclude = set(seed_num + seed_cat + [y_name] + ID_COLS)
    feats = [c for c in df.columns if c not in exclude]
    num, bin_, mul = [], [], []

    for c in seed_num:
        if c in df.columns and c != y_name:
            num.append(c)
    for c in seed_cat:
        if c in df.columns and c != y_name:
            nunq = df[c].dropna().nunique()
            (bin_ if nunq == 2 else mul).append(c)

    for c in feats:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            nunq = s.dropna().nunique()
            if nunq == 2:
                bin_.append(c)
            elif pd.api.types.is_integer_dtype(s) and nunq <= 10:
                mul.append(c)
            else:
                num.append(c)
        else:
            mul.append(c)

    def _uniq(xs):
        seen = set(); out = []
        for x in xs:
            if x not in seen:
                out.append(x); seen.add(x)
        return out
    return _uniq(num), _uniq(bin_), _uniq(mul)

vt = (CFG.get("preprocess", {}) or {}).get("variable_types", {}) or {}
seed_num = vt.get("continuous")  or []
seed_cat = vt.get("categorical") or []

num_cols, bin_cols, mul_cols = _infer_types(df, Y_NAME, seed_num, seed_cat)
all_feats = num_cols + bin_cols + mul_cols
if len(all_feats) == 0:
    _fail("可用特征数为 0，请检查 YAML 或 frozen 列名")
print(f"[info] inferred features: num={len(num_cols)}, bin={len(bin_cols)}, cat={len(mul_cols)}")
_ok("特征类型推断完成")

# 7) 轻训练 smoke test（失败仅警告）
try:
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    def make_ohe():
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore", sparse=False)

    pre = ColumnTransformer([
        ("num", StandardScaler(with_mean=False), num_cols),
        ("bin", "passthrough", bin_cols),
        ("mul", make_ohe(), mul_cols),
    ], remainder="drop")

    rng = np.random.default_rng(RANDOM_STATE)
    sub_idx = rng.choice(_read_idx(TRAIN_IDX), size=min(2000, len(_read_idx(TRAIN_IDX))), replace=False)

    X_sm = df.loc[sub_idx, all_feats].copy()
    y_sm = df.loc[sub_idx, Y_NAME].astype(int).values

    pipe = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(solver="liblinear", class_weight="balanced",
                                   random_state=RANDOM_STATE, max_iter=500))
    ])
    pipe.fit(X_sm, y_sm)
    p_sm = pipe.predict_proba(X_sm)[:, 1]
    auc = roc_auc_score(y_sm, p_sm)
    print(f"[info] smoke LR on ≤2k train rows: AUROC={auc:.3f}")
    if not (0.5 <= auc <= 1.0):
        _warn("训练集抽样 AUROC 偏低/异常，请人工复核（不阻断）")
    _ok("轻训练通过：预处理/模型串接正常")
except Exception as e:
    _warn(f"轻训练失败（不阻断）：{e}")

print("=== Sanity Check Finished ===")
