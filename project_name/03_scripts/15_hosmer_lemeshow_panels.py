# -*- coding: utf-8 -*-
r"""
Hosmer–Lemeshow calibration panels (train & test) for all models.
- mean_mode data + S4 selected features
- optional best params from tuning/best_params.json
- isotonic calibration on val (if available)
- apparent vs bootstrap bias-corrected curves
- HL chi-square p-value and MAE

Outputs:
  10_experiments/<VER_OUT>/final_eval_hl/
    - HL_panels.png/pdf
    - bin_tables/<model>_train_bins.csv (optional)
    - bin_tables/<model>_test_bins.csv  (optional)
"""

# ---------- headless matplotlib ----------
import os
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")

from pathlib import Path
from typing import List, Tuple, Optional, Dict
import warnings, json, hashlib
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import yaml
import joblib

# ===================== unified config =====================
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

CFG = yaml.safe_load(open(repo_root()/ "07_config"/"config.yaml","r",encoding="utf-8"))

PROJECT_ROOT = Path(CFG["paths"]["project_root"])
VER_IN  = CFG.get("run_id_in",  CFG.get("run_id"))
VER_OUT = CFG.get("run_id_out", CFG.get("run_id"))

DATA_DIR   = PROJECT_ROOT / "02_processed_data" / VER_IN
FROZEN_DIR = DATA_DIR / "frozen"
SPLIT_DIR  = DATA_DIR / "splits"

MEAN_MODE_XY = FROZEN_DIR / "charls_mean_mode_Xy.csv"
TRAIN_IDX = SPLIT_DIR / "charls_train_idx.csv"
VAL_IDX   = SPLIT_DIR / "charls_val_idx.csv"
TEST_IDX  = SPLIT_DIR / "charls_test_idx.csv"

# S4 & tuning
FEATURE_SWEEP_DIR = PROJECT_ROOT / "10_experiments" / VER_OUT / "feature_sweep"
TUNING_DIR        = PROJECT_ROOT / "10_experiments" / VER_OUT / "tuning"

# labels / ids / models
Y_NAME  = (CFG.get("outcome", {}) or {}).get("name", "depression_bin")
ID_COLS = list(CFG.get("id_cols") or []) or ["ID","householdID"]
_models_cfg = list((CFG.get("eval", {}) or {}).get("models", []))
if not _models_cfg:
    _models_cfg = ["lr","rf","extra_trees","gb","adaboost","lgb","xgb"]
MODEL_LIST = [m for m in _models_cfg if str(m).lower() != "catboost"]

# HL settings (overridable in YAML: eval.hl)
hl_cfg = (CFG.get("eval", {}) or {}).get("hl", {}) or {}
N_BINS = int(hl_cfg.get("n_bins", 10))
N_BOOT = int(hl_cfg.get("n_boot", 200))
SEED   = int(hl_cfg.get("seed", 1234))
SAVE_BIN_TABLES = bool(hl_cfg.get("save_bin_tables", True))

OUT_DIR = PROJECT_ROOT / "10_experiments" / VER_OUT / "final_eval_hl"
(OUT_DIR / "bin_tables").mkdir(parents=True, exist_ok=True)

# models saved by step 14 (final_eval_s7)
FINAL_S7_DIR = PROJECT_ROOT / "10_experiments" / VER_OUT / "final_eval_s7"
S7_MODELS_DIR = FINAL_S7_DIR / "models"

# ---------- sklearn ----------
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, AdaBoostClassifier
)

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


def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def load_idx_csv(p: Optional[Path]) -> Optional[np.ndarray]:
    if p and Path(p).exists():
        return pd.read_csv(p).iloc[:,0].values.astype(int)
    return None

def yaml_variable_types_from_cfg() -> Tuple[List[str], List[str], List[str]]:
    vt = (CFG.get("preprocess", {}) or {}).get("variable_types", {}) or {}
    num_cols = list(vt.get("continuous", []) or [])
    mul_cols = list(vt.get("categorical", []) or [])
    return num_cols, [], mul_cols

def infer_feature_types(df: pd.DataFrame, y: str,
                        init_num: Optional[List[str]]=None,
                        init_mul: Optional[List[str]]=None,
                        id_cols: Optional[List[str]]=None) -> Tuple[List[str], List[str], List[str]]:
    if id_cols is None: id_cols = []
    exclude = set(id_cols + [y])
    feats = [c for c in df.columns if c not in exclude]
    init_num = set(init_num or []); init_mul = set(init_mul or [])

    num_cols, bin_cols, mul_cols = [], [], []
    for c in feats:
        s = df[c]
        if c in init_num:
            num_cols.append(c); continue
        if c in init_mul:
            nunq = s.dropna().nunique()
            (bin_cols if nunq==2 else mul_cols).append(c); continue
        if pd.api.types.is_numeric_dtype(s):
            nunq = s.dropna().nunique()
            if nunq == 2: bin_cols.append(c)
            elif pd.api.types.is_integer_dtype(s) and nunq <= 10:
                (bin_cols if nunq==2 else mul_cols).append(c)
            else: num_cols.append(c)
        else:
            mul_cols.append(c)
    return num_cols, bin_cols, mul_cols

def build_preprocessor(all_num: List[str], all_bin: List[str], all_mul: List[str],
                       selected: Optional[List[str]]=None) -> ColumnTransformer:
    if selected is not None:
        s = set(selected)
        num_cols = [c for c in all_num if c in s]
        bin_cols = [c for c in all_bin if c in s]
        mul_cols = [c for c in all_mul if c in s]
    else:
        num_cols, bin_cols, mul_cols = all_num, all_bin, all_mul
    return ColumnTransformer(
        [("num", StandardScaler(with_mean=False), num_cols),
         ("bin", "passthrough", bin_cols),
         ("mul", make_ohe(), mul_cols)],
        remainder="drop"
    )

def _class_ratio(y: np.ndarray) -> float:
    pos = int(np.sum(y)); neg = len(y) - pos
    return (neg / max(pos, 1))

def make_model(mk: str, y_train: np.ndarray):
    mk = mk.lower()
    if mk == "lr":
        return LogisticRegression(solver="liblinear", penalty="l2",
                                  max_iter=500, class_weight="balanced", random_state=42)
    if mk == "rf":
        return RandomForestClassifier(n_estimators=600, min_samples_leaf=5,
                                     class_weight="balanced", n_jobs=-1, random_state=42)
    if mk == "extra_trees":
        return ExtraTreesClassifier(n_estimators=700, min_samples_leaf=3,
                                    class_weight="balanced", n_jobs=-1, random_state=42)
    if mk == "gb":
        return GradientBoostingClassifier(random_state=42)
    if mk == "adaboost":
        return AdaBoostClassifier(n_estimators=600, learning_rate=0.05, algorithm="SAMME", random_state=42)
    if mk == "lgb":
        if _HAS_LGBM:
            return LGBMClassifier(n_estimators=700, learning_rate=0.05, max_depth=-1,
                                  subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
                                  objective="binary", random_state=42, class_weight="balanced")
        return GradientBoostingClassifier(random_state=42)
    if mk == "xgb":
        if _HAS_XGB:
            return XGBClassifier(n_estimators=800, learning_rate=0.05, max_depth=5,
                                 subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
                                 objective="binary:logistic", random_state=42, n_jobs=-1,
                                 scale_pos_weight=_class_ratio(y_train), verbosity=0)
        return GradientBoostingClassifier(random_state=42)
    if mk == "catboost":
        if _HAS_CAT:
            w1 = _class_ratio(y_train)
            return CatBoostClassifier(iterations=800, learning_rate=0.05, depth=6,
                                      loss_function="Logloss", eval_metric="AUC",
                                      class_weights=[1.0, w1], random_seed=42, verbose=False)
        return None
    raise ValueError(mk)

def load_s7_pipeline(model_key: str):
    """Load calibrated/best pipeline saved by step 14 if available."""
    try:
        name = model_key.upper()
        cal_p = S7_MODELS_DIR / f"{name}_calibrated_isotonic.joblib"
        best_p = S7_MODELS_DIR / f"{name}_best_pipeline.joblib"
        if cal_p.exists():
            return joblib.load(cal_p)
        if best_p.exists():
            return joblib.load(best_p)
    except Exception:
        return None
    return None

def _as_proba(model_or_pipe, X):
    if hasattr(model_or_pipe, "predict_proba"):
        return model_or_pipe.predict_proba(X)[:,1]
    if hasattr(model_or_pipe, "decision_function"):
        z = model_or_pipe.decision_function(X)
        return 1.0 / (1.0 + np.exp(-z))
    return model_or_pipe.predict(X).astype(float)

# ---------- util: feature list from S4 & (optional) best params ----------
def pick_features_for_model(model_key: str, all_feats: List[str]) -> List[str]:
    f_best = FEATURE_SWEEP_DIR / f"S4_RFE_{model_key}_best_features.csv"
    if f_best.exists():
        feats = pd.read_csv(f_best)["feature"].dropna().astype(str).tolist()
        feats = [f for f in feats if f in all_feats]
        if len(feats): return feats
    # 回退：用 rfe_curve 里 best k 的“前 k”
    f_curve = FEATURE_SWEEP_DIR / f"rfe_curve_{model_key}.csv"
    if f_curve.exists():
        dfc = pd.read_csv(f_curve)
        sub = dfc[dfc["scenario"]=="internal_val"]
        if not len(sub): sub = dfc[dfc["scenario"]=="internal_test"]
        if len(sub):
            sub = sub.sort_values(["AUROC","AUPRC","k"], ascending=[False,False,True])
            k_best = int(sub.iloc[0]["k"])
            return all_feats[:k_best]
    return list(all_feats)

def load_best_params() -> Dict[str, Dict]:
    bp = {}
    f = TUNING_DIR / "best_params.json"
    if f.exists():
        try:
            bp = json.loads(Path(f).read_text(encoding="utf-8"))
        except Exception:
            bp = {}
    return bp

# ---------- HL + bootstrap ----------
def _seed_for(*parts) -> int:
    s = "|".join(map(str, parts))
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)

def bin_edges_by_quantiles(p: np.ndarray, n_bins: int) -> np.ndarray:
    qs = np.linspace(0, 1, n_bins+1)
    edges = np.quantile(p, qs)
    edges[0] = 0.0
    edges[-1] = 1.0
    edges = np.unique(edges)
    if len(edges) < 3:  # 全部概率相同等极端情况 -> 人工等距
        edges = np.linspace(0, 1, min(n_bins, 5)+1)
    return edges

def binned_stats(y: np.ndarray, p: np.ndarray, edges: np.ndarray) -> pd.DataFrame:
    idx = np.digitize(p, edges[1:-1], right=True)
    df = pd.DataFrame({"bin": idx, "y": y, "p": p})
    grp = df.groupby("bin", as_index=False).agg(
        n=("y","size"),
        mean_pred=("p","mean"),
        obs_rate=("y","mean"),
        exp=("p","sum"),
        obs=("y","sum")
    )
    all_bins = pd.DataFrame({"bin": np.arange(len(edges)-1)})
    out = all_bins.merge(grp, on="bin", how="left")
    for c in ["n","mean_pred","obs_rate","exp","obs"]:
        out[c] = out[c].fillna(0.0)
    return out

def hosmer_lemeshow_pvalue(table: pd.DataFrame) -> float:
    O = table["obs"].values
    E = table["exp"].values
    n = table["n"].values
    with np.errstate(divide="ignore", invalid="ignore"):
        term1 = (O - E)**2 / np.clip(E, 1e-12, None)
        term2 = ((n - O) - (n - E))**2 / np.clip(n - E, 1e-12, None)
        hl = np.nansum(term1 + term2)
    df = max(len(table) - 2, 1)
    try:
        from scipy.stats import chi2
        p = float(chi2.sf(hl, df))
    except Exception:
        p = np.nan
    return p

def optimism_corrected_curve(y: np.ndarray, p: np.ndarray, edges: np.ndarray,
                             n_boot: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    app = binned_stats(y, p, edges)["obs_rate"].values
    optims = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        in_boot = np.zeros(n, dtype=bool); in_boot[idx] = True
        oob = ~in_boot
        if oob.sum() < max(10, len(edges)-1):
            continue
        app_b = binned_stats(y[idx], p[idx], edges)["obs_rate"].values
        test_b = binned_stats(y[oob], p[oob], edges)["obs_rate"].values
        optims.append(app_b - test_b)
    if len(optims) == 0:
        return app
    optimism = np.nanmean(np.vstack(optims), axis=0)
    return app - optimism

# ---------- plotting ----------
import matplotlib.pyplot as plt

def plot_hl_panels(results: Dict[str, Dict[str, Dict[str, np.ndarray]]], out_png: Path):
    """
    results[model][split] = {"edges": edges, "mean_pred": x, "apparent": y_app, "corrected": y_cor, "p": pval, "mae": mae}
    split in {"train","test"}
    """
    models = list(results.keys())
    n_models = len(models)
    if n_models == 0:
        print("[warn] no models to plot.")
        return
    ncols = 2
    nrows = n_models
    fig = plt.figure(figsize=(10, 2.2*nrows))
    for i, m in enumerate(models):
        for j, split in enumerate(["train","test"]):
            ax = fig.add_subplot(nrows, ncols, i*ncols+j+1)
            r = results[m].get(split, None)
            if r is None:
                ax.axis("off"); continue
            x = r["mean_pred"]; y_app = r["apparent"]; y_cor = r["corrected"]
            ax.plot([0,1],[0,1],'--', color='gray', linewidth=1, label="Ideal")
            ax.plot(x, y_app, marker="o", linewidth=1.2, label="Apparent")
            ax.plot(x, y_cor, marker="o", linewidth=1.2, label="Bias-corrected")
            ax.set_xlim(0,1); ax.set_ylim(0,1)
            ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed probability")
            title = f"{m.upper()} — {'training set' if split=='train' else 'test set'}"
            ax.set_title(title, fontsize=9)
            ptxt = "NA" if (r["p"] is None or (isinstance(r["p"], float) and np.isnan(r["p"]))) else f"{r['p']:.3g}"
            ax.text(0.02, 0.05, f"HL p = {ptxt}\nMAE = {r['mae']:.3f}",
                    transform=ax.transAxes, fontsize=8,
                    bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.6))
            if i==0 and j==0:
                ax.legend(loc="upper left", fontsize=8, frameon=True)
            ax.grid(True, linestyle="--", alpha=0.25)
    fig.suptitle("Hosmer–Lemeshow calibration (train & test) — apparent vs bias-corrected", y=0.995)
    fig.tight_layout(rect=[0,0,1,0.97])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220); fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)

# ---------- main ----------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(MEAN_MODE_XY)
    assert Y_NAME in df.columns, f"{Y_NAME} not in {MEAN_MODE_XY}"

    tr_idx = load_idx_csv(TRAIN_IDX); va_idx = load_idx_csv(VAL_IDX); te_idx = load_idx_csv(TEST_IDX)

    # variable types from YAML + auto
    try:
        init_num, _bin_unused, init_mul = yaml_variable_types_from_cfg()
    except Exception:
        init_num, init_mul = [], []
    all_num, all_bin, all_mul = infer_feature_types(df, Y_NAME, init_num, init_mul, ID_COLS)
    all_feats = all_num + all_bin + all_mul

    # splits
    X_train_all = df.loc[tr_idx, all_feats]; y_train = df.loc[tr_idx, Y_NAME].astype(int).values
    X_val_all   = df.loc[va_idx, all_feats] if va_idx is not None else None
    y_val       = df.loc[va_idx, Y_NAME].astype(int).values if va_idx is not None else None
    X_test_all  = df.loc[te_idx, all_feats] if te_idx is not None else None
    y_test      = df.loc[te_idx, Y_NAME].astype(int).values if te_idx is not None else None

    best_params_map = load_best_params()  # optional

    results = {}

    for model_key in MODEL_LIST:
        print(f"[model] {model_key}")
        # Prefer loading saved pipelines from step 14 to avoid retraining
        cal_pipe = load_s7_pipeline(model_key)
        if cal_pipe is None:
            # fallback: local train (kept for robustness)
            selected = pick_features_for_model(model_key, all_feats)
            pre = build_preprocessor(all_num, all_bin, all_mul, selected=selected)
            clf = make_model(model_key, y_train)
            if clf is None:
                print(f"  [skip] {model_key} unavailable"); continue

            # inject best params if available
            try:
                bp = (best_params_map.get(model_key, {}) or {}).get("params", {}) or {}
                if bp:
                    clf.set_params(**bp)
            except Exception:
                pass

            pipe = Pipeline([("pre", pre), ("clf", clf)])
            pipe.fit(X_train_all, y_train)

            # isotonic calibration on val if present
            if X_val_all is not None and len(X_val_all):
                pre_fit = pipe.named_steps["pre"]; clf_fit = pipe.named_steps["clf"]
                try:
                    cal = CalibratedClassifierCV(estimator=clf_fit, method="isotonic", cv="prefit")
                except TypeError:
                    cal = CalibratedClassifierCV(base_estimator=clf_fit, method="isotonic", cv="prefit")
                X_val_t = pre_fit.transform(X_val_all); cal.fit(X_val_t, y_val)
                cal_pipe = Pipeline([( "pre", pre_fit), ("cal", cal)])
            else:
                cal_pipe = pipe

        # predictions
        # Use full feature frame; the loaded pipeline encodes internally by column names
        p_tr = _as_proba(cal_pipe, X_train_all)
        p_te = _as_proba(cal_pipe, X_test_all) if X_test_all is not None else None

        # per split
        res_model = {}
        for split_name, y_vec, p_vec in [("train", y_train, p_tr),
                                         ("test",  y_test,  p_te)]:
            if p_vec is None or y_vec is None:
                continue
            edges = bin_edges_by_quantiles(p_vec, N_BINS)
            tab = binned_stats(y_vec, p_vec, edges)
            pval = hosmer_lemeshow_pvalue(tab)
            app_y = tab["obs_rate"].values
            x_mean = tab["mean_pred"].values
            corr_y = optimism_corrected_curve(
                y_vec, p_vec, edges, N_BOOT,
                _seed_for(SEED, model_key, split_name)
            )
            res_model[split_name] = {
                "edges": edges, "mean_pred": x_mean,
                "apparent": app_y, "corrected": corr_y,
                "p": pval, "mae": float(np.mean(np.abs(p_vec - y_vec)))
            }

            if SAVE_BIN_TABLES:
                out_csv = OUT_DIR / "bin_tables" / f"{model_key}_{split_name}_bins.csv"
                tab.to_csv(out_csv, index=False, encoding="utf-8-sig")

        results[model_key] = res_model

    # plot big panel
    plot_hl_panels(results, OUT_DIR / "HL_panels.png")
    print(f"[save] {OUT_DIR / 'HL_panels.png'}")
    print(f"[save] {OUT_DIR / 'HL_panels.pdf'}")
    print("[done] HL panels finished.")

if __name__ == "__main__":
    main()
