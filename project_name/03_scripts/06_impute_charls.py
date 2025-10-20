# -*- coding: utf-8 -*-
r"""
06_impute_charls.py — 多种插补方法（CHARLS 主 + KLOSA 外部）

支持插补方法:
- mean_mode: 简单均值/众数插补
- knn: K近邻插补 (k=5)
- cart: 决策树插补
- random_forest: 随机森林插补
- bayes_logreg_polyreg: 贝叶斯岭回归(连续变量)+逻辑回归(二分变量)+多项回归(分类变量)
- pmm_logreg_polyreg: PMM(连续变量)+逻辑回归(二分变量)+多项回归(分类变量)
- mice_pmm: 经典MICE+PMM组合

依赖：
- pandas, numpy, scikit-learn, joblib, pyyaml
- 输入：02_processed_data/<VER>/splits/{charls_train,val,test}.csv, klosa_external.csv(可选)
- 输出：
  02_processed_data/<VER>/selected_imputation/<method>/{charls_train,val,test,klosa_external}.csv
  02_processed_data/<VER>/imputed/charls_{split}_{suffix}_Xy.csv
  02_processed_data/<VER>/imputers/<method>/{params.json, imputer.joblib}
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
import re

from sklearn.linear_model import BayesianRidge, LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor, KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.experimental import enable_iterative_imputer  # 必须导入这个才能使用IterativeImputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

# ---------------- 通用配置头 ----------------
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

CFG = yaml.safe_load(open(repo_root() / "07_config" / "config.yaml", "r", encoding="utf-8"))

PROJECT_ROOT = Path(CFG["paths"]["project_root"])
VER  = CFG.get("run_id_out", CFG.get("run_id"))
DATA_DIR = PROJECT_ROOT / "02_processed_data" / VER
SPLITS   = DATA_DIR / "splits"
# SEL_DIR和IMPUTERS将在主函数中根据选择的方法动态设置
IMPUTED  = DATA_DIR / "imputed"

IMPUTED.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = int(CFG.get("seed", CFG.get("random_state", 42)))
Y_NAME  = (CFG.get("outcome", {}) or {}).get("name", "depression_bin")
ID_COLS = list(CFG.get("ids", ["ID", "householdID"]))

# ---------------- 数据清洗辅助 ----------------
def _coerce_numeric_like(df: pd.DataFrame, exclude: list[str]) -> pd.DataFrame:
    """把“看起来像数字”的 object 列尽可能转成数值；保留纯文本/混合类别为 object。"""
    out = df.copy()
    for c in out.columns:
        if c in exclude:
            continue
        s = out[c]
        if pd.api.types.is_object_dtype(s):
            samp = s.dropna().astype(str).head(100)
            if len(samp) == 0:
                continue
            def _is_numlike(x: str) -> bool:
                x = x.strip().replace(",", "")
                if x in ("", ".", "-", "NA", "NaN", "nan", "None"):
                    return True
                try:
                    float(x); return True
                except Exception:
                    return False
            ratio = np.mean([_is_numlike(x) for x in samp])
            if ratio >= 0.8:
                out[c] = pd.to_numeric(s, errors="coerce")
    return out

# ---------------- 稳定分类器（用于分类变量的条件模型） ----------------
def _make_stable_classifier(n_classes: int):
    if n_classes <= 2:
        return LogisticRegression(
            solver="liblinear", penalty="l2",
            max_iter=200, tol=1e-3, warm_start=True,
            class_weight="balanced", random_state=RANDOM_STATE,
        )
    else:
        return LogisticRegression(
            solver="lbfgs", penalty="l2",
            max_iter=200, tol=1e-3, warm_start=True,
            multi_class="auto", random_state=RANDOM_STATE,
        )

def _fallback_classifier():
    return SGDClassifier(
        loss="log_loss", early_stopping=True, n_iter_no_change=5,
        max_iter=2000, tol=1e-3, class_weight="balanced",
        random_state=RANDOM_STATE,
    )

# ---------------- PMM 核心 ----------------
def _pmm_numeric(y_obs_pred, y_mis_pred, y_obs_true, k=5, rng=None):
    rng = rng or np.random.default_rng(RANDOM_STATE)
    y_obs_pred = np.asarray(y_obs_pred).reshape(-1, 1)
    y_mis_pred = np.asarray(y_mis_pred).reshape(-1, 1)
    if len(y_obs_pred) < 1:
        return np.repeat(np.nan, len(y_mis_pred))
    nn = NearestNeighbors(n_neighbors=max(1, min(k, len(y_obs_pred))), metric="euclidean")
    nn.fit(y_obs_pred)
    idxs = nn.kneighbors(y_mis_pred, return_distance=False)
    return np.asarray([y_obs_true[rng.choice(row)] for row in idxs])

def _pmm_categorical(P_obs, P_mis, y_obs_true, k=5, rng=None):
    rng = rng or np.random.default_rng(RANDOM_STATE)
    if len(P_obs) < 1:
        return np.repeat(np.nan, len(P_mis))
    nn = NearestNeighbors(n_neighbors=max(1, min(k, len(P_obs))), metric="euclidean")
    nn.fit(P_obs)
    idxs = nn.kneighbors(P_mis, return_distance=False)
    return np.asarray([y_obs_true[rng.choice(row)] for row in idxs], dtype=int)

# ---------------- 统一编码器：前缀数字提取 + 因子映射（未知类别 -> NaN） ----------------
_NUM_RE = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)")  # 抓前缀数字

def _extract_leading_number(s):
    """从字符串前缀提取数字；提不到返回 np.nan"""
    if pd.isna(s):
        return np.nan
    if isinstance(s, (int, float, np.integer, np.floating)):
        return float(s)
    m = _NUM_RE.match(str(s))
    return float(m.group(1)) if m else np.nan

def build_train_encoders(df_train: pd.DataFrame) -> Dict[str, Dict]:
    """
    针对训练集的 object 列：
      1) 先尝试提取前缀数字（如 '0.Fully Independent' -> 0）
      2) 若提不到数字，则以训练集出现的类别做 factorize，建立 {类别: 编码} 映射
    返回 encoders: {col: {"mode": "number"|"factorize", "map": {...}}}
    """
    encoders: Dict[str, Dict] = {}
    for col in df_train.columns:
        if df_train[col].dtype == "O":
            nums = df_train[col].map(_extract_leading_number)
            if nums.notna().any() and nums.notna().mean() >= 0.5:
                encoders[col] = {"mode": "number"}
            else:
                cats = pd.Categorical(df_train[col].astype("string"))
                mapping = {cat: i for i, cat in enumerate(cats.categories)}
                encoders[col] = {"mode": "factorize", "map": mapping}
    return encoders

def apply_encoders(df: pd.DataFrame, encoders: Dict[str, Dict]) -> pd.DataFrame:
    """
    将 encoders 应用于任一 split（val/test/external）：
      - number 模式：提取前缀数字，提不到→NaN
      - factorize 模式：按训练集映射，未知类别→NaN
    返回新的 DataFrame（不修改原 df）
    """
    out = df.copy()
    for col, spec in encoders.items():
        if col not in out.columns:
            continue
        mode = spec["mode"]
        if mode == "number":
            out[col] = out[col].map(_extract_leading_number)
        else:
            mapping = spec["map"]
            out[col] = out[col].astype("string").map(mapping).astype("float")
    # 尝试把剩余能转的都转为数值
    for col in out.columns:
        if out[col].dtype == "O":
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out

# ---------------- 均值/众数插补 ----------------
def mean_mode_imputation(
    X: pd.DataFrame,
    num_cols: List[str],
    bin_cols: List[str],
    mul_cols: List[str],
    seed: Optional[int] = RANDOM_STATE,
) -> pd.DataFrame:
    """简单的均值/众数插补"""
    X_imp = X.copy()
    
    # 连续变量用均值填充
    for c in num_cols:
        if X_imp[c].isna().any():
            med = np.nanmedian(pd.to_numeric(X_imp[c], errors="coerce").values)
            X_imp.loc[X_imp[c].isna(), c] = 0.0 if np.isnan(med) else med
    
    # 二分类和多分类变量用众数填充
    for c in bin_cols + mul_cols:
        if X_imp[c].isna().any():
            vc = X_imp[c].value_counts(dropna=True)
            X_imp.loc[X_imp[c].isna(), c] = (vc.index[0] if len(vc) else 0)
    
    return X_imp

# ---------------- KNN插补 ----------------
def knn_imputation(
    X: pd.DataFrame,
    num_cols: List[str],
    bin_cols: List[str],
    mul_cols: List[str],
    k: int = 5,
    seed: Optional[int] = RANDOM_STATE,
) -> pd.DataFrame:
    """使用KNN进行插补"""
    # 首先进行简单的均值/众数插补作为初始值
    X_init = mean_mode_imputation(X, num_cols, bin_cols, mul_cols, seed)
    
    # 使用sklearn的KNNImputer
    imputer = KNNImputer(n_neighbors=k, weights="uniform")
    # 转换为numpy数组进行插补
    imputed_array = imputer.fit_transform(X_init)
    
    # 转回DataFrame并保留原始列名
    X_imp = pd.DataFrame(imputed_array, columns=X.columns)
    
    # 确保分类变量被转回整数类型
    for c in bin_cols + mul_cols:
        if c in X_imp.columns:
            X_imp[c] = X_imp[c].round().astype(int)
    
    return X_imp

# ---------------- 决策树插补 (CART) ----------------
def cart_imputation(
    X: pd.DataFrame,
    num_cols: List[str],
    bin_cols: List[str],
    mul_cols: List[str],
    max_iter: int = 5,
    seed: Optional[int] = RANDOM_STATE,
) -> pd.DataFrame:
    """使用决策树进行插补"""
    rng = np.random.default_rng(seed)
    
    # 初始值采用均值/众数
    X_imp = mean_mode_imputation(X, num_cols, bin_cols, mul_cols, seed)
    
    # 配置IterativeImputer使用决策树作为估计器
    imputer = IterativeImputer(
        estimator=DecisionTreeRegressor(random_state=seed),
        random_state=seed,
        max_iter=max_iter,
        initial_strategy='mean',
        skip_complete=True
    )
    
    # 转换为numpy数组进行插补
    imputed_array = imputer.fit_transform(X_imp)
    
    # 转回DataFrame并保留原始列名
    X_imp = pd.DataFrame(imputed_array, columns=X.columns)
    
    # 确保分类变量被转回整数类型
    for c in bin_cols + mul_cols:
        if c in X_imp.columns:
            X_imp[c] = X_imp[c].round().astype(int)
            
    return X_imp

# ---------------- 随机森林插补 ----------------
def random_forest_imputation(
    X: pd.DataFrame,
    num_cols: List[str],
    bin_cols: List[str],
    mul_cols: List[str],
    max_iter: int = 5,
    seed: Optional[int] = RANDOM_STATE,
) -> pd.DataFrame:
    """使用随机森林进行插补"""
    rng = np.random.default_rng(seed)
    
    # 初始值采用均值/众数
    X_imp = mean_mode_imputation(X, num_cols, bin_cols, mul_cols, seed)
    
    # 配置IterativeImputer使用随机森林作为估计器
    imputer = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1),
        random_state=seed,
        max_iter=max_iter,
        initial_strategy='mean',
        skip_complete=True
    )
    
    # 转换为numpy数组进行插补
    imputed_array = imputer.fit_transform(X_imp)
    
    # 转回DataFrame并保留原始列名
    X_imp = pd.DataFrame(imputed_array, columns=X.columns)
    
    # 确保分类变量被转回整数类型
    for c in bin_cols + mul_cols:
        if c in X_imp.columns:
            X_imp[c] = X_imp[c].round().astype(int)
            
    return X_imp

# ---------------- 贝叶斯+逻辑回归+多项回归插补 ----------------
def bayes_logreg_polyreg_imputation(
    X: pd.DataFrame,
    num_cols: List[str],
    bin_cols: List[str],
    mul_cols: List[str],
    max_iter: int = 5,
    seed: Optional[int] = RANDOM_STATE,
) -> pd.DataFrame:
    """使用贝叶斯岭回归(连续)+逻辑回归(二分)+多项回归(分类)组合插补"""
    # 初始值采用均值/众数
    X_imp = mean_mode_imputation(X, num_cols, bin_cols, mul_cols, seed)
    
    # 创建IterativeImputer实例，使用贝叶斯岭回归
    imputer = IterativeImputer(
        estimator=BayesianRidge(),
        random_state=seed,
        max_iter=max_iter,
        initial_strategy='mean',
        skip_complete=True
    )
    
    # 插补
    imputed_array = imputer.fit_transform(X_imp)
    
    # 转回DataFrame并保留原始列名
    X_imp = pd.DataFrame(imputed_array, columns=X.columns)
    
    # 确保分类变量被转回整数类型
    for c in bin_cols + mul_cols:
        if c in X_imp.columns:
            X_imp[c] = X_imp[c].round().astype(int)
            
    return X_imp

# ---------------- MICE + PMM 主体 ----------------
def mice_pmm_once(
    X: pd.DataFrame,
    num_cols: List[str],
    bin_cols: List[str],
    mul_cols: List[str],
    max_iter: int = 5,
    seed: Optional[int] = RANDOM_STATE,
    k_pmm: int = 5,
    use_rf_for_numeric: bool = False,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X_imp = X.copy()

    # 初始填补
    for c in num_cols:
        if X_imp[c].isna().any():
            med = np.nanmedian(pd.to_numeric(X_imp[c], errors="coerce").values)
            X_imp.loc[X_imp[c].isna(), c] = 0.0 if np.isnan(med) else med
    for c in bin_cols + mul_cols:
        if X_imp[c].isna().any():
            vc = X_imp[c].value_counts(dropna=True)
            X_imp.loc[X_imp[c].isna(), c] = (vc.index[0] if len(vc) else 0)

    all_cols = [c for c in (num_cols + bin_cols + mul_cols) if c in X_imp.columns]

    for it in range(max_iter):
        print(f"[MICE] iteration {it+1}/{max_iter}", flush=True)
        for target in all_cols:
            others = [c for c in all_cols if c != target]
            y = X[target].values  # 用原始 y 判定缺失位置
            miss_mask = pd.isna(y)
            obs_mask = ~miss_mask
            if obs_mask.sum() == 0 or miss_mask.sum() == 0:
                continue

            # 仅在当前可用特征上拟合（确保为 float）
            X_obs_np = X_imp.loc[obs_mask, others].to_numpy(dtype=float, copy=True)
            X_mis_np = X_imp.loc[miss_mask, others].to_numpy(dtype=float, copy=True)

            finite_rows = np.isfinite(X_obs_np).all(axis=1)
            if finite_rows.sum() < 2:
                if target in num_cols:
                    X_imp.loc[miss_mask, target] = np.nanmean(pd.to_numeric(y[obs_mask], errors="coerce"))
                else:
                    y_obs_int = pd.to_numeric(y[obs_mask], errors="coerce")
                    y_obs_int = y_obs_int[~np.isnan(y_obs_int)].astype(int)
                    if len(y_obs_int):
                        X_imp.loc[miss_mask, target] = int(np.bincount(y_obs_int).argmax())
                continue

            X_obs_np = X_obs_np[finite_rows]
            y_obs_full = pd.to_numeric(y[obs_mask], errors="coerce")[finite_rows]

            if target in num_cols:
                try:
                    if use_rf_for_numeric:
                        reg = RandomForestRegressor(
                            n_estimators=400, min_samples_leaf=5,
                            random_state=RANDOM_STATE, n_jobs=-1
                        )
                        X_obs_sc, X_mis_sc = X_obs_np, X_mis_np
                    else:
                        reg = BayesianRidge()
                        scaler = StandardScaler(with_mean=True, with_std=True)
                        X_obs_sc = scaler.fit_transform(X_obs_np)
                        X_mis_sc = scaler.transform(X_mis_np)
                    reg.fit(X_obs_sc, y_obs_full)
                    y_obs_pred = reg.predict(X_obs_sc)
                    y_mis_pred = reg.predict(X_mis_sc)
                    donors = _pmm_numeric(y_obs_pred, y_mis_pred, y_obs_full, k=k_pmm, rng=rng)
                    X_imp.loc[miss_mask, target] = donors
                except Exception:
                    X_imp.loc[miss_mask, target] = np.nanmean(y_obs_full)
            else:
                y_obs_int = pd.to_numeric(y_obs_full, errors="coerce").astype(int)
                uniq = np.unique(y_obs_int)
                if uniq.size < 2:
                    maj = int(np.bincount(y_obs_int).argmax()) if len(y_obs_int) else 0
                    X_imp.loc[miss_mask, target] = maj
                    continue
                clf = _make_stable_classifier(int(uniq.size))
                try:
                    clf.fit(X_obs_np, y_obs_int)
                except Exception:
                    try:
                        clf = _fallback_classifier()
                        clf.fit(X_obs_np, y_obs_int)
                    except Exception:
                        X_imp.loc[miss_mask, target] = int(np.bincount(y_obs_int).argmax())
                        continue
                try:
                    P_obs = clf.predict_proba(X_obs_np)
                    P_mis = clf.predict_proba(X_mis_np)
                except Exception:
                    f_obs = np.atleast_2d(clf.decision_function(X_obs_np))
                    f_mis = np.atleast_2d(clf.decision_function(X_mis_np))
                    if f_obs.ndim == 1 or f_obs.shape[1] == 1:
                        sig = lambda z: 1.0/(1.0+np.exp(-z))
                        p1_obs, p1_mis = sig(f_obs.ravel()), sig(f_mis.ravel())
                        P_obs = np.vstack([1-p1_obs, p1_obs]).T
                        P_mis = np.vstack([1-p1_mis, p1_mis]).T
                    else:
                        def _softmax(z):
                            z = z - np.max(z, axis=1, keepdims=True)
                            ez = np.exp(z)
                            return ez/ez.sum(axis=1, keepdims=True)
                        P_obs, P_mis = _softmax(f_obs), _softmax(f_mis)
                donors = _pmm_categorical(P_obs, P_mis, y_obs_int, k=k_pmm, rng=np.random.default_rng(RANDOM_STATE))
                X_imp.loc[miss_mask, target] = donors
    return X_imp

# ---------------- Typing helpers ----------------
def infer_feature_types(
    df: pd.DataFrame,
    y_name: str,
    binary_threshold: int = 2,
    treat_small_int_as_cat: bool = True,
    exclude_cols: Optional[List[str]] = None,
) -> Tuple[List[str], List[str], List[str]]:
    exclude_cols = exclude_cols or []
    exclude = set(exclude_cols + [y_name])
    feats = [c for c in df.columns if c not in exclude]
    num_cols, bin_cols, mul_cols = [], [], []
    for c in feats:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            nunq = s.dropna().nunique()
            if 2 <= nunq <= binary_threshold:
                bin_cols.append(c)
            elif treat_small_int_as_cat and pd.api.types.is_integer_dtype(s) and nunq <= 10:
                (bin_cols if nunq == 2 else mul_cols).append(c)
            else:
                num_cols.append(c)
        else:
            mul_cols.append(c)
    return num_cols, bin_cols, mul_cols

# ---------------- I/O helpers ----------------
def _load_split(name: str) -> pd.DataFrame | None:
    p = SPLITS / f"charls_{name}.csv"
    return pd.read_csv(p) if p.exists() else None

def _load_external() -> pd.DataFrame | None:
    p = SPLITS / "klosa_external.csv"
    return pd.read_csv(p) if p.exists() else None

# ---------------- PMM+逻辑回归+多项回归插补 ----------------
def pmm_logreg_polyreg_imputation(
    X: pd.DataFrame,
    num_cols: List[str],
    bin_cols: List[str],
    mul_cols: List[str],
    max_iter: int = 5,
    seed: Optional[int] = RANDOM_STATE,
    k_pmm: int = 5,
) -> pd.DataFrame:
    """使用PMM(连续)+逻辑回归(二分)+多项回归(分类)组合插补，与MICE+PMM类似但只进行一轮"""
    rng = np.random.default_rng(seed)
    X_imp = X.copy()
    
    # 初始填补，用于PMM匹配
    for c in num_cols:
        if X_imp[c].isna().any():
            med = np.nanmedian(pd.to_numeric(X_imp[c], errors="coerce").values)
            X_imp.loc[X_imp[c].isna(), c] = 0.0 if np.isnan(med) else med
    for c in bin_cols + mul_cols:
        if X_imp[c].isna().any():
            vc = X_imp[c].value_counts(dropna=True)
            X_imp.loc[X_imp[c].isna(), c] = (vc.index[0] if len(vc) else 0)
    
    # 单次迭代处理每列
    all_cols = [c for c in (num_cols + bin_cols + mul_cols) if c in X_imp.columns]
    for c in all_cols:
        na_mask = X.loc[:, c].isna()
        if not na_mask.any():
            continue
        
        # 构建预测用数据集
        xcols = [x for x in all_cols if x != c]
        if len(xcols) < 1:
            continue
        
        # 选择合适的模型（连续/二分/多分类）
        if c in num_cols:
            # 数值变量用BayesianRidge + PMM
            x_train = X_imp.loc[~na_mask, xcols].values
            y_train = X_imp.loc[~na_mask, c].values
            x_miss = X_imp.loc[na_mask, xcols].values
            
            model = BayesianRidge()
            model.fit(x_train, y_train)
            y_pred_missing = model.predict(x_miss)
            y_pred_train = model.predict(x_train)
            
            # PMM 最终选择
            imp_values = _pmm_match(y_train, y_pred_train, y_pred_missing, k=k_pmm, rng=rng)
            X_imp.loc[na_mask, c] = imp_values
        elif c in bin_cols:
            # 二分类变量用LogisticRegression
            x_train = X_imp.loc[~na_mask, xcols].values
            y_train = X_imp.loc[~na_mask, c].values
            x_miss = X_imp.loc[na_mask, xcols].values
            
            try:
                model = LogisticRegression(solver="liblinear", random_state=int(rng.integers(1e8)))
                model.fit(x_train, y_train)
                y_pred = model.predict(x_miss)
                X_imp.loc[na_mask, c] = y_pred
            except:
                # 回退到众数
                mode_val = X_imp.loc[~na_mask, c].mode().iloc[0]
                X_imp.loc[na_mask, c] = mode_val
        else:
            # 多分类变量使用multi-class逻辑回归
            x_train = X_imp.loc[~na_mask, xcols].values
            y_train = X_imp.loc[~na_mask, c].values
            x_miss = X_imp.loc[na_mask, xcols].values
            
            try:
                model = LogisticRegression(multi_class='multinomial', solver='lbfgs', 
                                          random_state=int(rng.integers(1e8)))
                model.fit(x_train, y_train)
                y_pred = model.predict(x_miss)
                X_imp.loc[na_mask, c] = y_pred
            except:
                # 回退到众数
                mode_val = X_imp.loc[~na_mask, c].mode().iloc[0]
                X_imp.loc[na_mask, c] = mode_val
    
    return X_imp

def _combine_xy(X: pd.DataFrame, y: pd.DataFrame|None):
    return pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1) if y is not None else X

# ---------------- 主流程 ----------------
def run_imputation(method, args):
    """运行单个插补方法并保存结果"""
    # 设置输出目录
    SEL_DIR  = DATA_DIR / "selected_imputation" / method
    IMPUTERS = DATA_DIR / "imputers" / method
    
    # 创建必要的目录
    SEL_DIR.mkdir(parents=True, exist_ok=True)
    IMPUTERS.mkdir(parents=True, exist_ok=True)
    
    # 设置后缀
    suffix = method if args.suffix is None else args.suffix
    
    print(f"\n{'='*50}")
    print(f"[info] 开始处理插补方法: {method}")
    print(f"[info] 输出目录: {SEL_DIR}")
    print(f"{'='*50}\n")

    # 1) 读取 splits：train/val/test + external（若有）
    tr = _load_split("train"); va = _load_split("val")
    assert tr is not None and va is not None, "缺少 train/val，请先运行 03_split_charls.py"
    te = _load_split("test")
    ex = _load_external()

    # 排除 ID 列
    drop_ids = [c for c in ID_COLS if c in tr.columns]
    if drop_ids:
        tr = tr.drop(columns=drop_ids)
        va = va.drop(columns=[c for c in drop_ids if c in va.columns])
        if te is not None: te = te.drop(columns=[c for c in drop_ids if c in te.columns])
        if ex is not None: ex = ex.drop(columns=[c for c in drop_ids if c in ex.columns])

    # 先把"看起来像数字"的 object 列转成数值
    tr = _coerce_numeric_like(tr, exclude=[Y_NAME])
    va = _coerce_numeric_like(va, exclude=[Y_NAME])
    if te is not None: te = _coerce_numeric_like(te, exclude=[Y_NAME])
    if ex is not None: ex = _coerce_numeric_like(ex, exclude=[Y_NAME])

    # 2) 类型识别基于 train（口径固定）
    # 确保排除config.yaml中定义的drop_cols列表中的列
    drop_cols = (CFG.get("preprocess") or {}).get("drop_cols", [])
    
    # 明确加入cesd到排除列表(即使config中未指定)，以确保它绝对不会出现在特征中
    if "cesd" not in drop_cols:
        drop_cols.append("cesd")
        print("[info] 明确添加'cesd'到排除列表")
    
    # 创建扩展的排除列表
    exclude_cols = ID_COLS + [Y_NAME] + drop_cols
    print(f"[info] 排除特征: ID列={ID_COLS}, 标签列={Y_NAME}, 额外排除列={drop_cols}")
    
    # 检查并提示任何在drop_cols中但不在tr列中的列
    missing_cols = [c for c in drop_cols if c not in tr.columns]
    if missing_cols:
        print(f"[warn] 以下需要排除的列不在训练数据中: {missing_cols}")
    
    # 使用扩展的排除列表
    num_cols, bin_cols, mul_cols = infer_feature_types(tr, y_name=Y_NAME, exclude_cols=exclude_cols)
    
    # 再次确认cesd不在特征列中
    all_feature_cols = num_cols + bin_cols + mul_cols
    if "cesd" in all_feature_cols:
        print(f"[error] 严重警告: 'cesd'仍然出现在特征列中！移除它...")
        if "cesd" in num_cols: num_cols.remove("cesd")
        if "cesd" in bin_cols: bin_cols.remove("cesd")
        if "cesd" in mul_cols: mul_cols.remove("cesd")
    feats = num_cols + bin_cols + mul_cols

    def pick_X(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for c in feats:
            if c not in df.columns:
                df[c] = np.nan
        return df[feats].copy()

    Xtr, ytr = tr.drop(columns=[Y_NAME]), tr[[Y_NAME]]
    Xva, yva = va.drop(columns=[Y_NAME]), va[[Y_NAME]]
    Xte, yte = (te.drop(columns=[Y_NAME]), te[[Y_NAME]]) if te is not None and Y_NAME in te.columns else (te, None)
    Xex, yex = (ex.drop(columns=[Y_NAME]), ex[[Y_NAME]]) if ex is not None and Y_NAME in ex.columns else (ex, None)

    Xtr, Xva = pick_X(Xtr), pick_X(Xva)
    Xte = pick_X(Xte) if Xte is not None else None
    Xex = pick_X(Xex) if Xex is not None else None

    # === 统一编码：优先提取前缀数字；否则以 train 因子映射；未知类别 -> NaN ===
    encoders = build_train_encoders(Xtr)
    Xtr_enc = apply_encoders(Xtr, encoders)
    Xva_enc = apply_encoders(Xva, encoders)
    Xte_enc = apply_encoders(Xte, encoders) if Xte is not None else None
    Xex_enc = apply_encoders(Xex, encoders) if Xex is not None else None

    # 3) 根据选择的方法应用不同的插补策略
    print(f"[info] 开始在训练集上拟合插补模型: {method}")
    
    if method == "mean_mode":
        # 均值/众数插补
        Xtr_imp = mean_mode_imputation(
            X=Xtr_enc, num_cols=num_cols, bin_cols=bin_cols, mul_cols=mul_cols,
            seed=RANDOM_STATE
        )
        Xva_imp = mean_mode_imputation(
            X=Xva_enc, num_cols=num_cols, bin_cols=bin_cols, mul_cols=mul_cols,
            seed=RANDOM_STATE
        )
        Xte_imp = mean_mode_imputation(
            X=Xte_enc, num_cols=num_cols, bin_cols=bin_cols, mul_cols=mul_cols,
            seed=RANDOM_STATE
        ) if Xte_enc is not None else None
        Xex_imp = mean_mode_imputation(
            X=Xex_enc, num_cols=num_cols, bin_cols=bin_cols, mul_cols=mul_cols,
            seed=RANDOM_STATE
        ) if Xex_enc is not None else None
        
    elif method == "knn":
        # KNN插补
        Xtr_imp = knn_imputation(
            X=Xtr_enc, num_cols=num_cols, bin_cols=bin_cols, mul_cols=mul_cols,
            k=args.k_knn, seed=RANDOM_STATE
        )
        Xva_imp = knn_imputation(
            X=Xva_enc, num_cols=num_cols, bin_cols=bin_cols, mul_cols=mul_cols,
            k=args.k_knn, seed=RANDOM_STATE
        )
        Xte_imp = knn_imputation(
            X=Xte_enc, num_cols=num_cols, bin_cols=bin_cols, mul_cols=mul_cols,
            k=args.k_knn, seed=RANDOM_STATE
        ) if Xte_enc is not None else None
        Xex_imp = knn_imputation(
            X=Xex_enc, num_cols=num_cols, bin_cols=bin_cols, mul_cols=mul_cols,
            k=args.k_knn, seed=RANDOM_STATE
        ) if Xex_enc is not None else None
        
    elif method == "cart":
        # 决策树插补
        Xtr_imp = cart_imputation(
            X=Xtr_enc, num_cols=num_cols, bin_cols=bin_cols, mul_cols=mul_cols,
            max_iter=args.max_iter, seed=RANDOM_STATE
        )
        Xva_imp = cart_imputation(
            X=Xva_enc, num_cols=num_cols, bin_cols=bin_cols, mul_cols=mul_cols,
            max_iter=1, seed=RANDOM_STATE  # 验证集只做一次迭代
        )
        Xte_imp = cart_imputation(
            X=Xte_enc, num_cols=num_cols, bin_cols=bin_cols, mul_cols=mul_cols,
            max_iter=1, seed=RANDOM_STATE  # 测试集只做一次迭代
        ) if Xte_enc is not None else None
        Xex_imp = cart_imputation(
            X=Xex_enc, num_cols=num_cols, bin_cols=bin_cols, mul_cols=mul_cols,
            max_iter=1, seed=RANDOM_STATE  # 外部集只做一次迭代
        ) if Xex_enc is not None else None
        
    elif method == "random_forest":
        # 随机森林插补
        Xtr_imp = random_forest_imputation(
            X=Xtr_enc, num_cols=num_cols, bin_cols=bin_cols, mul_cols=mul_cols,
            max_iter=args.max_iter, seed=RANDOM_STATE
        )
        Xva_imp = random_forest_imputation(
            X=Xva_enc, num_cols=num_cols, bin_cols=bin_cols, mul_cols=mul_cols,
            max_iter=1, seed=RANDOM_STATE  # 验证集只做一次迭代
        )
        Xte_imp = random_forest_imputation(
            X=Xte_enc, num_cols=num_cols, bin_cols=bin_cols, mul_cols=mul_cols,
            max_iter=1, seed=RANDOM_STATE  # 测试集只做一次迭代
        ) if Xte_enc is not None else None
        Xex_imp = random_forest_imputation(
            X=Xex_enc, num_cols=num_cols, bin_cols=bin_cols, mul_cols=mul_cols,
            max_iter=1, seed=RANDOM_STATE  # 外部集只做一次迭代
        ) if Xex_enc is not None else None
        
    elif method == "bayes_logreg_polyreg":
        # 贝叶斯岭回归+逻辑回归+多项回归插补
        Xtr_imp = bayes_logreg_polyreg_imputation(
            X=Xtr_enc, num_cols=num_cols, bin_cols=bin_cols, mul_cols=mul_cols,
            max_iter=args.max_iter, seed=RANDOM_STATE
        )
        Xva_imp = bayes_logreg_polyreg_imputation(
            X=Xva_enc, num_cols=num_cols, bin_cols=bin_cols, mul_cols=mul_cols,
            max_iter=1, seed=RANDOM_STATE  # 验证集只做一次迭代
        )
        Xte_imp = bayes_logreg_polyreg_imputation(
            X=Xte_enc, num_cols=num_cols, bin_cols=bin_cols, mul_cols=mul_cols,
            max_iter=1, seed=RANDOM_STATE  # 测试集只做一次迭代
        ) if Xte_enc is not None else None
        Xex_imp = bayes_logreg_polyreg_imputation(
            X=Xex_enc, num_cols=num_cols, bin_cols=bin_cols, mul_cols=mul_cols,
            max_iter=1, seed=RANDOM_STATE  # 外部集只做一次迭代
        ) if Xex_enc is not None else None
        
    elif method == "pmm_logreg_polyreg":
        # PMM+逻辑回归+多项回归插补
        Xtr_imp = pmm_logreg_polyreg_imputation(
            X=Xtr_enc, num_cols=num_cols, bin_cols=bin_cols, mul_cols=mul_cols,
            max_iter=args.max_iter, seed=RANDOM_STATE, k_pmm=args.k_pmm
        )
        Xva_imp = pmm_logreg_polyreg_imputation(
            X=Xva_enc, num_cols=num_cols, bin_cols=bin_cols, mul_cols=mul_cols,
            max_iter=1, seed=RANDOM_STATE, k_pmm=args.k_pmm  # 验证集只做一次迭代
        )
        Xte_imp = pmm_logreg_polyreg_imputation(
            X=Xte_enc, num_cols=num_cols, bin_cols=bin_cols, mul_cols=mul_cols,
            max_iter=1, seed=RANDOM_STATE, k_pmm=args.k_pmm  # 测试集只做一次迭代
        ) if Xte_enc is not None else None
        Xex_imp = pmm_logreg_polyreg_imputation(
            X=Xex_enc, num_cols=num_cols, bin_cols=bin_cols, mul_cols=mul_cols,
            max_iter=1, seed=RANDOM_STATE, k_pmm=args.k_pmm  # 外部集只做一次迭代
        ) if Xex_enc is not None else None
        
    else:  # method == "mice_pmm"
        # MICE+PMM插补
        Xtr_imp = mice_pmm_once(
            X=Xtr_enc, num_cols=num_cols, bin_cols=bin_cols, mul_cols=mul_cols,
            max_iter=args.max_iter, seed=RANDOM_STATE, k_pmm=args.k_pmm,
            use_rf_for_numeric=args.use_rf_for_numeric,
        )
        
        # 用"已拟合结构"对 val/test/external 进行单次条件插补
        def transform_like_train(Xs_enc: pd.DataFrame) -> pd.DataFrame:
            if Xs_enc is None: return None
            combo = pd.concat([Xtr_enc, Xs_enc], axis=0, ignore_index=True)
            combo.iloc[:len(Xtr_enc), :] = Xtr_imp.values
            Xs_part = combo.iloc[len(Xtr_enc):, :].copy()
            Xs_imp = mice_pmm_once(
                X=Xs_part, num_cols=num_cols, bin_cols=bin_cols, mul_cols=mul_cols,
                max_iter=1, seed=RANDOM_STATE, k_pmm=args.k_pmm,
                use_rf_for_numeric=args.use_rf_for_numeric,
            )
            return Xs_imp
        
        Xva_imp = transform_like_train(Xva_enc)
        Xte_imp = transform_like_train(Xte_enc) if Xte_enc is not None else None
        Xex_imp = transform_like_train(Xex_enc) if Xex_enc is not None else None

    # 5) 写主线输出（selected_imputation）
    _combine_xy(Xtr_imp, ytr).to_csv(SEL_DIR/"charls_train.csv", index=False)
    _combine_xy(Xva_imp, yva).to_csv(SEL_DIR/"charls_val.csv", index=False)
    if Xte_imp is not None and yte is not None:
        _combine_xy(Xte_imp, yte).to_csv(SEL_DIR/"charls_test.csv", index=False)
    if Xex_imp is not None and yex is not None:
        _combine_xy(Xex_imp, yex).to_csv(SEL_DIR/"klosa_external.csv", index=False)

    # 6) 兼容输出（imputed/**_{suffix}_Xy.csv）
    _combine_xy(Xtr_imp, ytr).to_csv(IMPUTED/f"charls_train_{suffix}_Xy.csv", index=False)
    _combine_xy(Xva_imp, yva).to_csv(IMPUTED/f"charls_val_{suffix}_Xy.csv", index=False)
    if Xte_imp is not None and yte is not None:
        _combine_xy(Xte_imp, yte).to_csv(IMPUTED/f"charls_test_{suffix}_Xy.csv", index=False)
    if Xtr_imp is not None and Xva_imp is not None and Xte_imp is not None:
        all_X = pd.concat([Xtr_imp, Xva_imp, Xte_imp], axis=0, ignore_index=True)
        all_y = pd.concat([ytr, yva, yte], axis=0, ignore_index=True) if yte is not None else None
        _combine_xy(all_X, all_y).to_csv(IMPUTED/f"charls_all_{suffix}_Xy.csv", index=False)

    # 7) 保存"插补器参数快照"
    params_dict = {
        "method": method,
        "seed": RANDOM_STATE,
        "feature_order": feats,
        "num_cols": num_cols,
        "bin_cols": bin_cols,
        "mul_cols": mul_cols,
    }
    
    # 添加方法特有的参数
    if method in ["mice_pmm", "pmm_logreg_polyreg"]:
        params_dict.update({
            "max_iter": int(args.max_iter),
            "k_pmm": int(args.k_pmm),
        })
        if method == "mice_pmm":
            params_dict["use_rf_for_numeric"] = bool(args.use_rf_for_numeric)
    elif method == "knn":
        params_dict["k_knn"] = int(args.k_knn)
    elif method in ["cart", "random_forest", "bayes_logreg_polyreg"]:
        params_dict["max_iter"] = int(args.max_iter)
            
    (IMPUTERS/"params.json").write_text(
        pd.Series(params_dict).to_json(force_ascii=False, indent=2),
        encoding="utf-8"
    )
    joblib.dump({
        "encoders": encoders,
        "Xtr_shape": Xtr.shape,
        "Xtr_cols": list(Xtr.columns),
    }, IMPUTERS/"imputer.joblib")
    
    print(f"[ok] selected_imputation -> {SEL_DIR}")
    print(f"[ok] imputed (compat) -> {IMPUTED}")
    print(f"[ok] imputers -> {IMPUTERS}")
    
    return f"[完成] 插补方法 {method} 处理完成"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", type=str, default="mice_pmm", 
                   choices=["mean_mode", "knn", "cart", "random_forest", 
                            "bayes_logreg_polyreg", "pmm_logreg_polyreg", "mice_pmm"],
                   help="要使用的插补方法")
    ap.add_argument("--all", action="store_true", help="运行所有插补方法")
    ap.add_argument("--suffix", type=str, default=None, 
                   help="imputed 兼容输出文件名后缀，默认使用method名称")
    ap.add_argument("--max_iter", type=int, default=5, help="MICE方法的最大迭代次数")
    ap.add_argument("--k_pmm", type=int, default=5, help="PMM方法的k值")
    ap.add_argument("--k_knn", type=int, default=5, help="KNN方法的k值")
    ap.add_argument("--use_rf_for_numeric", action="store_true", help="在MICE中使用随机森林回归")
    args = ap.parse_args()
    
    if args.all:
        all_methods = ["mean_mode", "knn", "cart", "random_forest", 
                      "bayes_logreg_polyreg", "pmm_logreg_polyreg", "mice_pmm"]
        print(f"[info] 将依次运行所有插补方法: {', '.join(all_methods)}")
        
        for method in all_methods:
            try:
                result = run_imputation(method, args)
                print(result)
            except Exception as e:
                print(f"[error] 处理方法 {method} 时发生错误: {e}")
    else:
        run_imputation(args.method, args)


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
