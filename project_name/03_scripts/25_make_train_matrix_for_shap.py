# -*- coding: utf-8 -*-
"""
25_make_train_matrix_for_shap.py —— 生成 SHAP 所需的训练矩阵与背景样本

功能：
- 读取 config.yaml，定位项目根与版本（VER_IN/VER_OUT、Y_NAME）
- 读取冻结后的均值/众数版本数据：02_processed_data/<VER_IN>/frozen/charls_mean_mode_Xy.csv
- 使用 splits/charls_train_idx.csv 取 Train 子集（越界自动裁剪，给出提示）
- 读取 RFE 或 TopK 的 RF 特征清单（FEATURE_SOURCE='rfe'/'topk'），若缺失自动回退到 S5 importance，再回退到“全部特征（除 y）”
- 构建预处理器：数值(StandardScaler)、二元(FunctionTransformer->float)、类别 OneHot（handle_unknown=ignore，稠密输出）
- 拟合预处理器于 Train，并导出：
    X_train.npy / y_train.npy / X_bg.npy / pre.joblib / feature_names.json / selected_features.json / meta.json
"""

from __future__ import annotations
from pathlib import Path
import os, json, warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import yaml
from joblib import dump

# ================= 用户可调区域 =================
# 使用哪种 RF 特征清单： "rfe" 或 "topk"
FEATURE_SOURCE = "rfe"      # ← 想用 TopK 就改成 "topk"
# 背景样本规模（用于 SHAP explainer 的 background）
BG_MAX = 512
RANDOM_STATE = 42
# =================================================

# ------------- 基础配置与路径 -------------
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

CFG_PATH = repo_root()/ "07_config"/"config.yaml"
if not CFG_PATH.exists():
    raise FileNotFoundError(f"找不到配置文件：{CFG_PATH}")

CFG = yaml.safe_load(open(CFG_PATH, "r", encoding="utf-8")) or {}
if "paths" not in CFG or "project_root" not in CFG["paths"]:
    raise KeyError("config.yaml 缺少 paths.project_root")

PROJECT_ROOT = Path(CFG["paths"]["project_root"])
VER_IN  = CFG.get("run_id_in",  CFG.get("run_id"))
VER_OUT = CFG.get("run_id_out", CFG.get("run_id"))
if not VER_IN or not VER_OUT:
    raise KeyError("config.yaml 需要提供 run_id_in / run_id_out（或 run_id）")

DATA_CSV   = PROJECT_ROOT / "02_processed_data" / VER_IN / "frozen" / "charls_mean_mode_Xy.csv"
TRAIN_IDX  = PROJECT_ROOT / "02_processed_data" / VER_IN / "splits" / "charls_train_idx.csv"
Y_NAME     = (CFG.get("outcome", {}) or {}).get("name", "depression_bin")

SWEEP_DIR  = PROJECT_ROOT / "10_experiments" / VER_OUT / "feature_sweep"
IMP_DIR    = PROJECT_ROOT / "10_experiments" / VER_OUT / "feature_importance"
CACHE_DIR  = PROJECT_ROOT / "10_experiments" / VER_OUT / "cache" / "shap_rf"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ------------- sklearn 预处理与模型组件 -------------
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

def make_ohe():
    """兼容不同 sklearn 版本的 OHE 稠密输出"""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

# 关键修复：用命名函数替代 lambda，避免 joblib/pickle 报错
def _astype_float(X):
    """将传入矩阵/数据框安全地转为 float"""
    try:
        return X.astype(float)
    except Exception:
        return np.asarray(X, dtype=float)

# ------------- 读取/推断变量类型 -------------
def _yaml_variable_types(df: pd.DataFrame):
    vt = (CFG.get("preprocess", {}) or {}).get("variable_types", {}) or {}
    cont = [c for c in (vt.get("continuous", []) or []) if c in df.columns]
    cat  = [c for c in (vt.get("categorical", []) or []) if c in df.columns]
    return cont, cat

def _auto_variable_types(df: pd.DataFrame, exclude: list[str]):
    rest = [c for c in df.columns if c not in set(exclude)]
    num_like = [c for c in rest if pd.api.types.is_numeric_dtype(df[c])]
    obj_like = [c for c in rest if c not in num_like]
    # 仅把“严格二元”的放到 bin（其余即便是小整数也按 mul，避免误判）
    bin_cols = [c for c in num_like if df[c].dropna().nunique() == 2]
    num_cols = [c for c in num_like if c not in bin_cols]
    mul_cols = obj_like[:]  # 其余全部当作多分类
    return num_cols, bin_cols, mul_cols

def get_feature_types(df: pd.DataFrame, y_col: str):
    """YAML 优先，自动补齐；二元列单独归 bin。"""
    yaml_num, yaml_cat = _yaml_variable_types(df)
    yaml_bin = [c for c in (yaml_num + yaml_cat) if c in df.columns and df[c].dropna().nunique() == 2]
    yaml_num = [c for c in yaml_num if c not in yaml_bin]
    yaml_cat = [c for c in yaml_cat if c not in yaml_bin]

    covered = set([y_col] + yaml_num + yaml_cat + yaml_bin)
    auto_num, auto_bin, auto_mul = _auto_variable_types(df, exclude=list(covered))

    num_cols = list(dict.fromkeys(yaml_num + auto_num))
    bin_cols = list(dict.fromkeys(yaml_bin + auto_bin))
    mul_cols = list(dict.fromkeys(yaml_cat + auto_mul))
    return num_cols, bin_cols, mul_cols

# ------------- 构建预处理器（返回所用列清单） -------------
def build_preprocessor(num_cols, bin_cols, mul_cols, selected=None):
    """返回 (preprocessor, {'num':[], 'bin':[], 'mul':[]})"""
    if selected is not None:
        s = set(selected)
        num_cols = [c for c in num_cols if c in s]
        bin_cols = [c for c in bin_cols if c in s]
        mul_cols = [c for c in mul_cols if c in s]
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=False), num_cols),
            ("bin", Pipeline([("astype_float", FunctionTransformer(_astype_float))]), bin_cols),
            ("mul", make_ohe(), mul_cols),
        ],
        remainder="drop"
    )
    return pre, {"num": num_cols, "bin": bin_cols, "mul": mul_cols}

# ------------- 其它小工具 -------------
def load_idx_csv(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"找不到索引文件：{path}")
    s = pd.read_csv(path).iloc[:, 0].astype(int).values
    return s

def _clip_index(idx: np.ndarray, n: int) -> np.ndarray:
    """索引越界保护（并打印提示）"""
    bad = (idx < 0) | (idx >= n)
    if bad.any():
        print(f"[warn] 发现 {bad.sum()} 个越界索引（n={n}），已自动剔除。")
        idx = idx[~bad]
    return idx

def _feature_list_from_source(source: str) -> list[str] | None:
    """按指定来源读取特征；不存在返回 None，不抛错。"""
    source = (source or "").lower().strip()
    if source == "rfe":
        f = SWEEP_DIR / "S4_RFE_rf_best_features.csv"
    elif source == "topk":
        # 允许大小写/命名差异的两个候选
        cands = [
            SWEEP_DIR / "S4_TopK_rf_best_features.csv",
            SWEEP_DIR / "S4_topk_rf_best_features.csv"
        ]
        f = next((p for p in cands if p.exists()), None)
        if f is None:
            return None
    else:
        return None

    if f and f.exists():
        try:
            df = pd.read_csv(f)
            col = "feature" if "feature" in df.columns else df.columns[0]
            feats = df[col].dropna().astype(str).tolist()
            return feats
        except Exception:
            return None
    return None

def _feature_list_from_importance() -> list[str] | None:
    """从 S5 importance 读取（按 importance 降序）；不存在返回 None。"""
    cand = [
        IMP_DIR / "rf_importance.csv",
        IMP_DIR / "RF_importance.csv"
    ]
    f = next((p for p in cand if p.exists()), None)
    if f is None:
        return None
    try:
        df = pd.read_csv(f)
        col = "feature" if "feature" in df.columns else df.columns[0]
        feats = df[col].astype(str).tolist()
        if "importance" in df.columns:
            feats = (df[[col, "importance"]]
                     .sort_values("importance", ascending=False)[col]
                     .astype(str).tolist())
        return feats
    except Exception:
        return None

def resolve_selected_features(df: pd.DataFrame) -> list[str]:
    """按 RFE/TopK -> Importance -> 全特征 的优先级获取特征列表，并与数据列取交集。"""
    feats = _feature_list_from_source(FEATURE_SOURCE)
    src = FEATURE_SOURCE.upper()
    if feats is None:
        feats = _feature_list_from_importance()
        src = "IMPORTANCE"
    if feats is None:
        feats = [c for c in df.columns if c != Y_NAME]
        src = "ALL_COLUMNS"

    feats = [c for c in feats if c in df.columns and c != Y_NAME]
    if not feats:
        raise RuntimeError("选中特征列表为空（与数据列无交集）。请检查版本与特征文件。")
    print(f"[feat] source={src}, selected_features={len(feats)}")
    return feats

def safe_feature_names_from_pre(pre: ColumnTransformer) -> list[str]:
    """尽可能使用 sklearn 提供的 get_feature_names_out；否则手工拼接。"""
    try:
        return list(pre.get_feature_names_out())
    except Exception:
        names = []
        # 从 transformers_ 读取
        mapper = {name: cols for (name, _, cols) in pre.transformers_}
        # 数值 / 二元：原名
        for c in mapper.get("num", []) or []:
            names.append(f"num__{c}")
        for c in mapper.get("bin", []) or []:
            names.append(f"bin__{c}")
        # 类别：尝试从 OHE 读 categories_
        try:
            ohe = pre.named_transformers_["mul"]
            cats = list(ohe.categories_)
            mul_cols = mapper.get("mul", []) or []
            for col, cat_list in zip(mul_cols, cats):
                for cat in cat_list:
                    names.append(f"mul__{col}={cat}")
        except Exception:
            # 兜底：只给出列前缀
            for c in mapper.get("mul", []) or []:
                names.append(f"mul__{c}")
        return names

def ensure_float32(a: np.ndarray) -> np.ndarray:
    if not isinstance(a, np.ndarray):
        a = np.asarray(a)
    if a.dtype != np.float32:
        a = a.astype(np.float32, copy=False)
    return a

def stratified_background_idx(y: np.ndarray, n_max: int, seed: int) -> np.ndarray:
    """分层采样背景索引：尽量保持标签比例；样本不足时退化为无放回随机。"""
    n = len(y)
    n_use = int(min(n_max, n))
    if n_use <= 0:
        return np.arange(0)
    rng = np.random.default_rng(seed)
    u = np.unique(y)
    if len(u) < 2 or n_use >= n:
        return rng.choice(n, size=n_use, replace=False)
    # 按比例分配
    idx_all = np.arange(n)
    out = []
    for lab in u:
        idx_lab = idx_all[y == lab]
        k = max(1, int(round(len(idx_lab) / n * n_use)))
        k = min(k, len(idx_lab))
        out.append(rng.choice(idx_lab, size=k, replace=False))
    sel = np.concatenate(out, axis=0)
    # 若因为四舍五入导致长度不等于 n_use，做微调
    if len(sel) > n_use:
        sel = rng.choice(sel, size=n_use, replace=False)
    elif len(sel) < n_use:
        remain = np.setdiff1d(idx_all, sel, assume_unique=False)
        add = rng.choice(remain, size=(n_use - len(sel)), replace=False)
        sel = np.concatenate([sel, add], axis=0)
    return sel

# ------------- 主流程 -------------
def main():
    if not DATA_CSV.exists():
        raise FileNotFoundError(f"找不到数据：{DATA_CSV}")
    if not TRAIN_IDX.exists():
        raise FileNotFoundError(f"找不到 Train 索引：{TRAIN_IDX}")

    df = pd.read_csv(DATA_CSV)
    if Y_NAME not in df.columns:
        raise KeyError(f"标签列 {Y_NAME} 不在 {DATA_CSV}")

    # 按配置丢弃潜在泄露列（如 cesd）
    drop_cols = list(((CFG.get("preprocess", {}) or {}).get("drop_cols", []) or []))
    if drop_cols:
        keep = [c for c in df.columns if c not in drop_cols]
        if len(keep) < len(df.columns):
            df = df[keep].copy()
            print(f"[clean] dropped columns per config: {drop_cols}")

    # 索引裁剪与去重
    idx_raw = load_idx_csv(TRAIN_IDX)
    idx = _clip_index(idx_raw, len(df))
    if len(idx) == 0:
        raise RuntimeError("Train 索引为空（或全部越界）。")

    df_tr = df.iloc[idx].copy()

    # 变量类型
    num_cols, bin_cols, mul_cols = get_feature_types(df, Y_NAME)
    all_feats = [c for c in (num_cols + bin_cols + mul_cols) if c in df.columns]
    print(f"[data] train_rows={len(df_tr)}, all_features={len(all_feats)}")

    # 特征清单（RFE/TopK -> importance -> all）
    rf_feats = resolve_selected_features(df)
    rf_feats = [c for c in rf_feats if c in all_feats]
    if not rf_feats:
        raise RuntimeError("选中特征在 all_features 中为空，请检查变量类型设置或 YAML。")

    # 预处理器（仅对选中特征生效）
    pre, used = build_preprocessor(num_cols, bin_cols, mul_cols, selected=rf_feats)

    X_tr = df_tr[rf_feats]
    y_tr = df_tr[Y_NAME].astype(int).values

    print("[fit ] fitting preprocessor on Train ...")
    Xt = pre.fit_transform(X_tr)   # numpy 数组（dense）
    Xt = ensure_float32(np.asarray(Xt))
    print(f"[done] Xt shape = {Xt.shape}, y shape = {y_tr.shape}")

    # 背景样本（分层抽样）
    bg_idx = stratified_background_idx(y_tr, BG_MAX, RANDOM_STATE)
    X_bg = Xt[bg_idx]
    print(f"[bg  ] background rows = {len(bg_idx)}  (stratified)")

    # 列名（变换后）
    feat_names = safe_feature_names_from_pre(pre)

    # ----------- 保存产物 -----------
    np.save(CACHE_DIR / "X_train.npy", Xt)
    np.save(CACHE_DIR / "y_train.npy", y_tr.astype(np.int32, copy=False))
    np.save(CACHE_DIR / "X_bg.npy", X_bg)

    with open(CACHE_DIR / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(list(map(str, feat_names)), f, ensure_ascii=False, indent=2)

    with open(CACHE_DIR / "selected_features.json", "w", encoding="utf-8") as f:
        json.dump(rf_feats, f, ensure_ascii=False, indent=2)

    meta = {
        "ver_in": VER_IN,
        "ver_out": VER_OUT,
        "feature_source": FEATURE_SOURCE,
        "train_rows": int(len(Xt)),
        "bg_rows": int(len(bg_idx)),
        "x_cols": int(Xt.shape[1]),
        "y_name": Y_NAME,
        "used_cols": used,  # 各类型原始列
    }
    with open(CACHE_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 保存预处理器（已修复可序列化）
    dump(pre, CACHE_DIR / "pre.joblib")

    print("\n[save]", CACHE_DIR / "X_train.npy")
    print("[save]", CACHE_DIR / "y_train.npy")
    print("[save]", CACHE_DIR / "X_bg.npy")
    print("[save]", CACHE_DIR / "feature_names.json")
    print("[save]", CACHE_DIR / "selected_features.json")
    print("[save]", CACHE_DIR / "meta.json")
    print("[save]", CACHE_DIR / "pre.joblib")
    print("\n[done] SHAP 训练矩阵与背景样本就绪。")

if __name__ == "__main__":
    main()
