# -*- coding: utf-8 -*-
"""
11_tune_models_nestedcv.py — 在 Train 上做内层CV超参调优（AUC），为 S7 冻结最优超参
产物：
  10_experiments/<VER_OUT>/tuning/best_params.json
  10_experiments/<VER_OUT>/tuning/cv_results_<model>.csv

改动要点（快速稳定版）：
- 避免嵌套并行：外层并行，模型内部单线程
- scoring="roc_auc"（不再用 needs_proba，减少红字）
- GridSearchCV(verbose=2) + 打印候选×折数
- LightGBM 先用小网格跑通
- 每个模型完成后增量写 best_params.json；单个配置出错不影响全局（error_score=np.nan）

新增：
- 读取并排除 id_cols，不把 ID 当作特征
- 运行前检查必须文件（frozen Xy 与 train 索引）是否存在
"""

from pathlib import Path
import os, json, warnings, time
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import yaml

# ============ 统一读取配置 ============
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

CFG = yaml.safe_load(open(repo_root()/ "07_config"/"config.yaml","r",encoding="utf-8"))

PROJECT_ROOT = Path(CFG["paths"]["project_root"])
VER_IN  = CFG.get("run_id_in",  CFG.get("run_id"))
VER_OUT = CFG.get("run_id_out", CFG.get("run_id"))

MEAN_MODE_XY = PROJECT_ROOT / "02_processed_data" / VER_IN  / "frozen" / "charls_mean_mode_Xy.csv"
TRAIN_IDX    = PROJECT_ROOT / "02_processed_data" / VER_IN  / "splits" / "charls_train_idx.csv"
FEATURE_SWEEP_DIR = PROJECT_ROOT / "10_experiments" / VER_OUT / "feature_sweep"

Y_NAME  = (CFG.get("outcome", {}) or {}).get("name", "depression_bin")
ID_COLS = list(CFG.get("id_cols") or []) or ["ID", "householdID"]  # <-- 新增：读取 ID 列

OUT_DIR = PROJECT_ROOT / "10_experiments" / VER_OUT / "tuning"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 计算相关 —— 并行度降到 2–4（Windows 更稳）
_cpu = os.cpu_count() or 4
N_JOBS = max(1, min(4, _cpu - 1))
RANDOM_STATE = int(CFG.get("random_state", 42))
INNER_CV = 5

# ============ sklearn 基础 ============
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV

def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # sklearn >=1.4
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)         # sklearn <1.4

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
)

# 可选库
try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except Exception:
    _HAS_LGBM = False

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

try:
    from catboost import CatBoostClassifier
    _HAS_CAT = True
except Exception:
    _HAS_CAT = False

# ============ 工具 ============
def load_idx_csv(path: Path) -> np.ndarray:
    return pd.read_csv(path).iloc[:,0].values.astype(int)

def _yaml_variable_types(df: pd.DataFrame):
    vt = (CFG.get("preprocess", {}) or {}).get("variable_types", {}) or {}
    cont = [c for c in (vt.get("continuous", []) or []) if c in df.columns]
    cat  = [c for c in (vt.get("categorical", []) or []) if c in df.columns]
    return cont, cat

def _auto_variable_types(df: pd.DataFrame, exclude: list):
    rest = [c for c in df.columns if c not in set(exclude)]
    num_like = [c for c in rest if pd.api.types.is_numeric_dtype(df[c])]
    obj_like = [c for c in rest if c not in num_like]
    bin_cols = [c for c in num_like if df[c].dropna().nunique()==2]
    num_cols = [c for c in num_like if c not in bin_cols]
    mul_cols = obj_like[:]  # 其余当作多分类
    return num_cols, bin_cols, mul_cols

def get_feature_types(df: pd.DataFrame, y_col: str, id_cols: list[str]):  # <-- 修改：接受 id_cols
    yaml_num, yaml_cat = _yaml_variable_types(df)
    yaml_bin = [c for c in (yaml_num + yaml_cat) if df[c].dropna().nunique()==2]
    yaml_num = [c for c in yaml_num if c not in yaml_bin]
    yaml_cat = [c for c in yaml_cat if c not in yaml_bin]
    # 排除 y + ID 列 + 已由 YAML 指定的列
    covered = set([y_col] + id_cols + yaml_num + yaml_cat + yaml_bin)
    auto_num, auto_bin, auto_mul = _auto_variable_types(df, exclude=list(covered))
    num_cols = list(dict.fromkeys(yaml_num + auto_num))
    bin_cols = list(dict.fromkeys(yaml_bin + auto_bin))
    mul_cols = list(dict.fromkeys(yaml_cat + auto_mul))
    return num_cols, bin_cols, mul_cols

def build_preprocessor(num_cols, bin_cols, mul_cols, selected=None, model_type="lr"):
    """构建预处理器，根据模型类型使用不同的预处理策略：
    - 树模型（rf, xgb等）：直接转float，不做标准化和One-Hot
    - 线性模型（lr等）：标准化连续变量，One-Hot分类变量
    """
    if selected is not None:
        s = set(selected)
        num_cols = [c for c in num_cols if c in s]
        bin_cols = [c for c in bin_cols if c in s]
        mul_cols = [c for c in mul_cols if c in s]
    
    # 树模型使用简单预处理（只转float，不做标准化和One-Hot）
    if model_type.lower() in ["rf", "extra_trees", "gb", "adaboost", "lgb", "xgb", "catboost"]:
        return ColumnTransformer(
            transformers=[
                ("num", FunctionTransformer(lambda X: X.astype(float)), num_cols),
                ("bin", FunctionTransformer(lambda X: X.astype(float)), bin_cols),
                ("cat", FunctionTransformer(lambda X: X.astype(float)), mul_cols),
            ],
            remainder="drop"
        )
    # 线性模型使用标准预处理（标准化+One-Hot）
    else:
        return ColumnTransformer(
            transformers=[
                ("num", StandardScaler(with_mean=False), num_cols),
                ("bin", Pipeline([("astype_float", FunctionTransformer(lambda X: X.astype(float))) ]), bin_cols),
                ("mul", make_ohe(), mul_cols),
            ],
            remainder="drop"
        )

def _class_ratio(y: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    pos = y.sum(); neg = len(y) - pos
    return (neg / max(pos, 1))

# ============ 模型工厂（内部单线程） ============
def make_estimator(model_key: str, y_train: np.ndarray):
    mk = model_key.lower()
    if mk == "lr":
        return LogisticRegression(solver="liblinear", penalty="l2",
                                  max_iter=500, class_weight="balanced",
                                  random_state=RANDOM_STATE)
    elif mk == "rf":
        return RandomForestClassifier(class_weight="balanced", n_jobs=1,
                                      random_state=RANDOM_STATE)
    elif mk == "extra_trees":
        return ExtraTreesClassifier(class_weight="balanced", n_jobs=1,
                                    random_state=RANDOM_STATE)
    elif mk == "gb":
        return GradientBoostingClassifier(random_state=RANDOM_STATE)
    elif mk == "adaboost":
        return AdaBoostClassifier(algorithm="SAMME", random_state=RANDOM_STATE)
    elif mk == "lgb":
        if _HAS_LGBM:
            return LGBMClassifier(objective="binary",
                                  class_weight="balanced",
                                  random_state=RANDOM_STATE,
                                  n_jobs=1,          # 关键：模型内部单线程
                                  verbose=-1)
        return GradientBoostingClassifier(random_state=RANDOM_STATE)
    elif mk == "xgb":
        if _HAS_XGB:
            return XGBClassifier(objective="binary:logistic",
                                 random_state=RANDOM_STATE,
                                 n_jobs=1,          # 关键：模型内部单线程
                                 scale_pos_weight=_class_ratio(y_train),
                                 verbosity=0)
        return GradientBoostingClassifier(random_state=RANDOM_STATE)
    elif mk == "catboost":
        if _HAS_CAT:
            w1 = _class_ratio(y_train)
            return CatBoostClassifier(loss_function="Logloss", eval_metric="AUC",
                                      class_weights=[1.0, w1],
                                      random_seed=RANDOM_STATE,
                                      thread_count=1,  # 关键：模型内部单线程
                                      verbose=False)
        return None
    else:
        raise ValueError(f"Unsupported model: {model_key}")

# ============ 参数网格 ============
def param_grid_for(model_key: str):
    mk = model_key.lower()
    if mk == "lr":
        return {"clf__C":[0.01,0.1,1,10]}
    if mk == "rf":
        return {"clf__n_estimators":[300,600],
                "clf__max_depth":[None,8,14],
                "clf__min_samples_leaf":[1,3,5]}
    if mk == "extra_trees":
        return {"clf__n_estimators":[300,700],
                "clf__max_depth":[None,8,14],
                "clf__min_samples_leaf":[1,3]}
    if mk == "gb":
        return {"clf__n_estimators":[200,400,700],
                "clf__learning_rate":[0.05,0.1],
                "clf__max_depth":[2,3,4]}
    if mk == "adaboost":
        return {"clf__n_estimators":[300,600,900],
                "clf__learning_rate":[0.03,0.05,0.1]}
    if mk == "lgb" and _HAS_LGBM:
        # 先小空间跑通
        return {"clf__n_estimators":[300,600],
                "clf__learning_rate":[0.05,0.1],
                "clf__num_leaves":[31,63],
                "clf__colsample_bytree":[0.8]}
    if mk == "xgb" and _HAS_XGB:
        return {"clf__n_estimators":[400,800],
                "clf__learning_rate":[0.05,0.1],
                "clf__max_depth":[3,5],
                "clf__subsample":[0.8],
                "clf__colsample_bytree":[0.8]}
    if mk == "catboost" and _HAS_CAT:
        return {"clf__iterations":[400,800],
                "clf__learning_rate":[0.05,0.1],
                "clf__depth":[4,6]}
    return {}

# 保守加载特征集合（无文件则用全量）
def load_best_features(model_key: str, all_feats: list) -> list:
    f = FEATURE_SWEEP_DIR / f"S4_RFE_{model_key}_best_features.csv"
    if f.exists():
        feats = pd.read_csv(f)["feature"].dropna().astype(str).tolist()
        return [c for c in feats if c in all_feats]
    return list(all_feats)

# 增量保存 best_params.json
def save_best_params_incremental(path: Path, best_params: dict):
    existed = {}
    if path.exists():
        try:
            existed = json.load(open(path, "r", encoding="utf-8"))
        except Exception:
            existed = {}
    existed.update(best_params)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(existed, f, ensure_ascii=False, indent=2)

# ============ 主流程 ============
def main():
    # 运行前的存在性检查（友好报错）
    for p in [MEAN_MODE_XY, TRAIN_IDX]:
        if not Path(p).exists():
            raise FileNotFoundError(f"缺少必要文件：{p}。请先完成 03_split_charls.py 与 08_freeze_mean_mode.py")

    df = pd.read_csv(MEAN_MODE_XY)
    assert Y_NAME in df.columns, f"{Y_NAME} 不在 {MEAN_MODE_XY}"
    tr_idx = load_idx_csv(TRAIN_IDX)
    df_tr  = df.iloc[tr_idx].copy()

    # 变量类型：YAML 优先，自动补齐（并排除 ID 列）
    num_cols, bin_cols, mul_cols = get_feature_types(df, Y_NAME, ID_COLS)  # <-- 使用 ID_COLS
    all_feats = [c for c in (num_cols + bin_cols + mul_cols) if c in df.columns]

    # 确保完全排除 ID 列（双保险）
    all_feats = [c for c in all_feats if c not in set(ID_COLS)]

    X_tr_full = df_tr[all_feats]
    y_tr = df_tr[Y_NAME].astype(int).values

    print(f"[data] train_rows={len(df_tr)}, features={len(all_feats)}; "
          f"num={len(num_cols)}, bin={len(bin_cols)}, mul={len(mul_cols)}")
    print(f"[cfg ] INNER_CV={INNER_CV}, N_JOBS={N_JOBS}, RANDOM_STATE={RANDOM_STATE}")

    MODEL_LIST = ["lr","rf","extra_trees","gb","adaboost","lgb","xgb"]  # 禁用 catboost

    best_params_all = {}
    for model_key in MODEL_LIST:
        t0 = time.time()
        print(f"\n[tune] {model_key}  (outer n_jobs={N_JOBS}, model threads=1)", flush=True)

        feats = load_best_features(model_key, all_feats)
        pre   = build_preprocessor(num_cols, bin_cols, mul_cols, selected=feats, model_type=model_key)
        est   = make_estimator(model_key, y_tr)
        if est is None:
            print(f"  [skip] {model_key} 不可用"); continue

        pipe  = Pipeline([("pre", pre), ("clf", est)])
        grid  = param_grid_for(model_key)
        cv    = StratifiedKFold(n_splits=INNER_CV, shuffle=True, random_state=RANDOM_STATE)

        # 任务量估算 + 细粒度日志
        n_cand = int(np.prod([len(v) for v in grid.values()])) if grid else 1
        print(f"  [cv] candidates={n_cand} × folds={INNER_CV} ≈ total fits={n_cand*INNER_CV}", flush=True)

        gs = GridSearchCV(
            pipe,
            grid if grid else {"clf__random_state":[RANDOM_STATE]},
            scoring="roc_auc",
            cv=cv,
            n_jobs=N_JOBS,          # 外层并行
            refit=True,
            verbose=2,              # 细粒度进度
            return_train_score=False,
            pre_dispatch="2*n_jobs",
            error_score=np.nan      # 单个配置失败不崩溃
        )

        try:
            gs.fit(X_tr_full[feats], y_tr)
        except KeyboardInterrupt:
            print("  [warn] 收到 KeyboardInterrupt，尝试保存已完成部分…", flush=True)
            if hasattr(gs, "cv_results_"):
                cv_res = pd.DataFrame(gs.cv_results_)
                cv_res.to_csv(OUT_DIR / f"cv_results_{model_key}.csv", index=False, encoding="utf-8-sig")
            raise

        # 保存 CV 明细
        cv_res = pd.DataFrame(gs.cv_results_)
        cv_res.to_csv(OUT_DIR / f"cv_results_{model_key}.csv", index=False, encoding="utf-8-sig")
        print(f"  [save] cv_results_{model_key}.csv", flush=True)

        # 记录最优参数
        bp = {}
        for k, v in gs.best_params_.items():
            if k.startswith("clf__"):
                bp[k.replace("clf__","")] = v
        best_params_all[model_key] = {"features": feats, "params": bp}

        # 增量写 best_params.json（保证模型级断点也能落盘）
        save_best_params_incremental(OUT_DIR / "best_params.json",
                                     {model_key: best_params_all[model_key]})

        dt = time.time() - t0
        print(f"  [best] {model_key} -> {bp} (|feats|={len(feats)}), time={dt/60:.1f} min", flush=True)

    # 再整体写一遍汇总（与增量一致）
    with open(OUT_DIR / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params_all, f, ensure_ascii=False, indent=2)
    print(f"\n[save] {OUT_DIR / 'best_params.json'}")

if __name__ == "__main__":
    main()
