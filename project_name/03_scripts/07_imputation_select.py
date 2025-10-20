# -*- coding: utf-8 -*-
r"""
07_imputation_select.py — 多插补 × 多模型评估（含 CHARLS 训练/验证/测试与外部集）
指标：AUROC, AUPRC, Accuracy, Precision, Recall, F1_Score, Brier_Score, Specificity, NPV

新增/改动要点（与全链口径对齐）：
- 只在 VAL 上选阈值（Youden）：eval.use_val_thresholds=true（默认）
  - 选出的阈值用于 VAL/TEST/EXTERNAL（严禁在 test 上重新挑阈值）
  - 输出 val_thresholds.csv 记录 (method, model, thr, tpr, fpr)
- 自动发现 02_processed_data/<VER_IN>/imputed 下的 *_Xy.csv（优先 charls_all_*）
  同时包含 baseline：frozen/charls_mean_mode_Xy.csv
- 结果落盘：10_experiments/<VER_OUT>/multi_model_eval
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings, re, yaml, numpy as np, pandas as pd
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------- 统一配置头 ----------------
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

CFG = yaml.safe_load(open(repo_root() / "07_config" / "config.yaml", "r", encoding="utf-8"))

PROJECT_ROOT = Path(CFG["paths"]["project_root"])
VER_IN  = CFG.get("run_id_in",  CFG.get("run_id"))
VER_OUT = CFG.get("run_id_out", CFG.get("run_id"))

DATA_DIR   = PROJECT_ROOT / "02_processed_data" / VER_IN
FROZEN_DIR = DATA_DIR / "frozen"
IMP_DIR    = DATA_DIR / "imputed"
SPLIT_DIR  = DATA_DIR / "splits"
OUT_DIR    = PROJECT_ROOT / "10_experiments" / VER_OUT / "multi_model_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 目标与辅助列
Y_NAME  = (CFG.get("outcome", {}) or {}).get("name", "depression_bin")
IDS     = CFG.get("ids", ["ID", "householdID"])
SEED    = int(CFG.get("seed", CFG.get("random_state", 42)))

# 变量类型 seeds（来自 config.yaml，可为空）
VT = (CFG.get("preprocess", {}) or {}).get("variable_types", {}) or {}
FORCE_NUM = list(VT.get("continuous")  or [])
FORCE_CAT = list(VT.get("categorical") or [])

# Eval 配置
EVAL_CFG = (CFG.get("eval", {}) or {})
MODELS = EVAL_CFG.get("models", ["lr","rf","extra_trees","gb","adaboost","lgb","xgb","catboost"])
CLASS_THRESHOLD = float(EVAL_CFG.get("class_threshold", 0.5))  # 当不使用 val 阈值时的固定阈值
USE_VAL_TH = bool(EVAL_CFG.get("use_val_thresholds", True))    # 默认：True

# 规范化方法名，去除历史后缀，避免重复（例如 mice_pmm_rep1 -> mice_pmm）
_PMM_REP_RE = re.compile(r"^mice_pmm_rep\d+$", re.IGNORECASE)
def normalize_method_name(method: str) -> str:
    m = method.strip().lower()
    if _PMM_REP_RE.match(m):
        return "mice_pmm"
    return m

# ---------- 兼容 OneHotEncoder ----------
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, AdaBoostClassifier
)
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve
)

def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # sklearn >= 1.4
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)         # sklearn < 1.4

# ---------------- 实用函数 ----------------
def _read_idx(name: str) -> np.ndarray:
    p = SPLIT_DIR / f"charls_{name}_idx.csv"
    return pd.read_csv(p).iloc[:, 0].astype(int).values

def _ensure_int_cats(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns and not pd.api.types.is_integer_dtype(out[c]):
            le = LabelEncoder()
            out[c] = le.fit_transform(out[c].astype("category").astype(str))
    return out

def _infer_feature_types(
    df: pd.DataFrame,
    y_name: str,
    force_num: Optional[List[str]] = None,
    force_cat: Optional[List[str]] = None,
    id_cols: Optional[List[str]] = None,
) -> Tuple[List[str], List[str], List[str]]:
    force_num = set(force_num or []); force_cat = set(force_cat or [])
    id_cols   = set(id_cols or [])
    exclude = {y_name} | id_cols
    feats = [c for c in df.columns if c not in exclude]

    num_cols = [c for c in feats if c in force_num and c in df.columns]
    mul_cols = [c for c in feats if c in force_cat and c in df.columns]
    assigned = set(num_cols) | set(mul_cols)

    bin_cols: List[str] = []
    for c in feats:
        if c in assigned: 
            continue
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            nunq = s.dropna().nunique()
            if nunq == 2:
                bin_cols.append(c)
            elif pd.api.types.is_integer_dtype(s) and 3 <= nunq <= 10:
                mul_cols.append(c)
            else:
                num_cols.append(c)
        else:
            mul_cols.append(c)
    return num_cols, bin_cols, mul_cols

def _class_ratio(y: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    pos = y.sum(); neg = len(y) - pos
    return (neg / max(pos, 1)) if pos > 0 else 1.0

def _make_model(model_key: str, y_train: np.ndarray):
    mk = model_key.lower()
    if mk == "lr":
        return LogisticRegression(solver="liblinear", penalty="l2", max_iter=1000,
                                  class_weight="balanced", random_state=SEED)
    if mk == "rf":
        return RandomForestClassifier(n_estimators=600, min_samples_leaf=5,
                                     class_weight="balanced", n_jobs=-1, random_state=SEED)
    if mk == "extra_trees":
        return ExtraTreesClassifier(n_estimators=700, min_samples_leaf=3,
                                    class_weight="balanced", n_jobs=-1, random_state=SEED)
    if mk == "gb":
        return GradientBoostingClassifier(random_state=SEED)
    if mk == "adaboost":
        return AdaBoostClassifier(n_estimators=600, learning_rate=0.05,
                                  algorithm="SAMME", random_state=SEED)
    if mk == "lgb":
        try:
            from lightgbm import LGBMClassifier
            return LGBMClassifier(n_estimators=700, learning_rate=0.05, max_depth=-1,
                                  subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
                                  objective="binary", random_state=SEED, class_weight="balanced")
        except Exception:
            return GradientBoostingClassifier(random_state=SEED)
    if mk == "xgb":
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(n_estimators=800, learning_rate=0.05, max_depth=5,
                                 subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
                                 objective="binary:logistic", random_state=SEED, n_jobs=-1,
                                 scale_pos_weight=_class_ratio(y_train))
        except Exception:
            return GradientBoostingClassifier(random_state=SEED)
    if mk == "catboost":
        try:
            from catboost import CatBoostClassifier
            w1 = _class_ratio(y_train)
            return CatBoostClassifier(iterations=800, learning_rate=0.05, depth=6,
                                      loss_function="Logloss", eval_metric="AUC",
                                      class_weights=[1.0, w1], random_seed=SEED, verbose=False)
        except Exception:
            return None
    raise ValueError(f"Unsupported model: {model_key}")

def _evaluate_metrics(y_true: np.ndarray, p: np.ndarray, threshold: float) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p).clip(1e-8, 1 - 1e-8)

    auroc = roc_auc_score(y_true, p)
    auprc = average_precision_score(y_true, p)
    brier = brier_score_loss(y_true, p)

    y_pred = (p >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1  = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv         = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return {
        "AUROC": auroc, "AUPRC": auprc, "Accuracy": acc, "Precision": pre, "Recall": rec,
        "F1_Score": f1, "Brier_Score": brier, "Specificity": specificity, "NPV": npv
    }

def _predict_pipe(pipe: Pipeline, X_df: pd.DataFrame) -> np.ndarray:
    if hasattr(pipe[-1], "predict_proba"):
        return pipe.predict_proba(X_df)[:, 1]
    if hasattr(pipe[-1], "decision_function"):
        z = pipe.decision_function(X_df)
        return 1 / (1 + np.exp(-z))
    return pipe.predict(X_df).astype(float)

def _map_indices_to_local(df_len: int,
                          tr_idx: Optional[np.ndarray],
                          va_idx: Optional[np.ndarray],
                          te_idx: Optional[np.ndarray],
                          imp_name: str,
                          imp_path: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    根据插补文件的行数/文件名，决定该文件应映射哪些“本地索引”。
    - 若 df_len == 全量样本数（由 train/val/test 最大索引 + 1 估计），则返回原 train/val/test 索引。
    - 若 df_len == len(train) 或文件名含 'train'，仅返回 local train = 0..df_len-1。
    - 若 df_len == len(val)   或文件名含 'val'，  仅返回 local val   = 0..df_len-1。
    - 若 df_len == len(test)  或文件名含 'test'， 仅返回 local test  = 0..df_len-1。
    - 否则回退：当作 train-only。
    """
    name = str(imp_path).lower()
    full_len_est = None
    for arr in (tr_idx, va_idx, te_idx):
        if arr is not None and len(arr) > 0:
            full_len_est = max(full_len_est or -1, int(np.max(arr)))
    full_len_est = None if full_len_est is None else (full_len_est + 1)

    local_tr = local_va = local_te = None
    if full_len_est is not None and df_len == full_len_est:
        return tr_idx, va_idx, te_idx

    if (tr_idx is not None and df_len == len(tr_idx)) or ("train" in name):
        local_tr = np.arange(df_len)
    if (va_idx is not None and df_len == len(va_idx)) or ("val" in name):
        local_va = np.arange(df_len)
    if (te_idx is not None and df_len == len(te_idx)) or ("test" in name):
        local_te = np.arange(df_len)

    if local_tr is None and local_va is None and local_te is None:
        print(f"[hint] {imp_name}: 文件长度={df_len} 无法匹配全量/拆分长度；默认当作 train-only。")
        local_tr = np.arange(df_len)
    return local_tr, local_va, local_te

# ---------------- 画图 ----------------
def _make_barplots(summary_csv: Path, out_dir: Path):
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    df = pd.read_csv(summary_csv)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    metrics = ["AUROC", "AUPRC", "Accuracy", "Precision", "Recall",
               "F1_Score", "Brier_Score", "Specificity", "NPV"]

    for scen in sorted(df["scenario"].unique()):
        sub = df[df["scenario"] == scen].copy()
        methods = sorted(sub["method"].unique().tolist())
        models  = sorted(sub["model"].unique().tolist())

        table = (sub.set_index(["method", "model"])[metrics]
                   .unstack("model").reindex(index=methods))

        # 莫兰迪配色（用户指定HEX）
        morandi = [
            "#6E8D8A",
            "#7D94B6",
            "#E0B67A",
            "#E5A79A",
            "#C16E71",
            "#ABC8E5",
            "#D8A0C1",
            "#FFCEBC",
            "#C6BF94",
            "#D0D08A",
        ]
        morandi_cycle = (morandi * ((len(models) // len(morandi)) + 1))[:len(models)]

        for m in metrics:
            if m not in table.columns.get_level_values(0):
                continue
            values = table[m]  # index=method, columns=model

            x = np.arange(len(methods))
            n_groups = len(models)
            width = 0.8 / max(n_groups, 1)

            fig = plt.figure(figsize=(max(7, 1.3 * len(methods)), 5.0))
            ax = fig.add_subplot(111)

            # 背景与网格（莫兰迪风格：低饱和、轻网格）
            ax.set_facecolor("#F5F5F2")
            fig.patch.set_facecolor("#F5F5F2")

            for i, mdl in enumerate(models):
                y = values[mdl].values if mdl in values.columns else np.zeros(len(methods))
                color = morandi_cycle[i]
                edge = to_rgba("#6E6E6E", 0.55)
                ax.bar(
                    x + i * width - 0.4 + width / 2, y, width=width,
                    label=mdl, color=color, edgecolor=edge, linewidth=0.8
                )

            ax.set_title(f"{scen} — {m}")
            ax.set_xlabel("Imputation method")
            ax.set_ylabel(m)
            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=30, ha="right")
            # 图例：两列，右下角，背景白色（放在坐标轴内，不再溢出）
            lg = ax.legend(
                loc="lower right", ncol=2,
                frameon=True, facecolor="#FFFFFF", edgecolor="#DDDDDD", framealpha=0.98,
                fontsize=10, title="Model", title_fontsize=11, handlelength=1.4,
                columnspacing=1.2, handletextpad=0.6
            )
            ax.grid(axis="y", linestyle="--", alpha=0.25, color="#9AA1A6")

            fig.tight_layout()
            fig.savefig(plots_dir / f"{scen}_{m}.png", dpi=200, bbox_inches="tight", pad_inches=0.25)
            fig.savefig(plots_dir / f"{scen}_{m}.pdf", bbox_inches="tight", pad_inches=0.25)
            plt.close(fig)

# ---------------- 主流程 ----------------
def main():
    # 1) 索引与外部数据
    tr_idx = _read_idx("train")
    va_idx = _read_idx("val")
    te_idx = _read_idx("test")

    # 外部集（如不存在将跳过）
    EXT_MAIN     = FROZEN_DIR / "klosa_main_Xy.csv"
    EXT_TRANSFER = FROZEN_DIR / "klosa_transfer_Xy.csv"
    ext_main = pd.read_csv(EXT_MAIN) if EXT_MAIN.exists() else None
    ext_tran = pd.read_csv(EXT_TRANSFER) if EXT_TRANSFER.exists() else None
    for ext, tag in [(ext_main, "external_main"), (ext_tran, "external_transfer")]:
        if ext is not None and Y_NAME not in ext.columns:
            raise KeyError(f"[{tag}] 缺少标签列 {Y_NAME}")

    # 2) 组装要评估的“插补输入”
    inputs: Dict[str, Path] = {}

    # 2.1 baseline：mean_mode（来自 frozen 全量 Xy）
    base_mean_mode = FROZEN_DIR / "charls_mean_mode_Xy.csv"
    if base_mean_mode.exists():
        inputs["mean_mode"] = base_mean_mode
    else:
        print(f"[WARN] baseline mean_mode 缺失：{base_mean_mode}")

    # 2.2 自动发现 imputed 下的 *_Xy.csv（优先 charls_all_*）
    if not IMP_DIR.exists():
        print(f"[WARN] 插补目录不存在：{IMP_DIR}")
    else:
        # 收集所有候选，再按优先级填入 inputs
        candidates: Dict[str, Dict[str, Path]] = {}
        for p in sorted(IMP_DIR.glob("charls_*_Xy.csv")):
            name = p.name.lower()
            m_all = re.match(r"charls_all_(.+)_xy\.csv", name)
            if m_all:
                method = normalize_method_name(m_all.group(1))
                candidates.setdefault(method, {})["all"] = p
                continue
            m_part = re.match(r"charls_(train|val|test)_(.+)_xy\.csv", name)
            if m_part:
                sp, method = m_part.group(1), m_part.group(2)
                method = normalize_method_name(method)
                entry = candidates.setdefault(method, {})
                entry[sp] = p

        for method, entry in candidates.items():
            if "all" in entry:
                inputs[method] = entry["all"]
            else:
                # 没有 all，就任选一个分片文件（后续用索引映射）
                for key in ["train","val","test"]:
                    if key in entry:
                        inputs[method] = entry[key]
                        break

    if not inputs:
        raise SystemExit("[FAIL] 未发现任何可评估的 *_Xy.csv；请先运行 06/07_add_label_to_imputed.py 生成。")

    print("[inputs] 将评估的方法：", ", ".join(sorted(inputs.keys())))

    collectors = {k: [] for k in ["internal_train","internal_val","internal_test","external_main","external_transfer"]}
    thr_rows = []  # 记录 val 阈值

    # 3) 逐方法评估
    for imp_name, imp_path in inputs.items():
        if not imp_path.exists():
            print(f"[WARN] skip {imp_name}: file not found -> {imp_path}")
            continue

        df_imp = pd.read_csv(imp_path)
        if Y_NAME not in df_imp.columns:
            print(f"[WARN] {imp_name}: 缺少标签列 {Y_NAME}，跳过")
            continue
        if df_imp[Y_NAME].dropna().nunique() > 2:
            print(f"[WARN] {imp_name}: y 非二分类（nunique={df_imp[Y_NAME].nunique()}），跳过")
            continue

        # 根据文件行数/文件名映射本地索引（train/val/test/all）
        lt, lv, ls = _map_indices_to_local(len(df_imp), tr_idx, va_idx, te_idx, imp_name, imp_path)

        # 特征分型（优先 YAML seeds）
        num_cols, bin_cols, mul_cols = _infer_feature_types(
            df_imp, y_name=Y_NAME, force_num=FORCE_NUM, force_cat=FORCE_CAT, id_cols=IDS
        )

        X_all = df_imp[num_cols + bin_cols + mul_cols].copy()
        y_all = df_imp[Y_NAME].astype(int).values
        X_all = _ensure_int_cats(X_all, mul_cols)

        pre = ColumnTransformer(
            [("num", StandardScaler(with_mean=False), num_cols),
             ("bin", "passthrough", bin_cols),
             ("mul", make_ohe(), mul_cols)],
            remainder="drop"
        )

        for model_key in MODELS:
            try:
                if lt is None or len(lt) == 0:
                    print(f"[WARN] {imp_name} × {model_key}: 无 train 索引，跳过训练")
                    continue

                clf = _make_model(model_key, y_all[lt])
                if clf is None:
                    print(f"[WARN] {imp_name} × {model_key}: 无可用模型（缺依赖？），跳过")
                    continue

                pipe = Pipeline([("pre", pre), ("clf", clf)])
                pipe.fit(X_all.iloc[lt], y_all[lt])

                # ---- 阈值选择：只在 VAL 上（可关闭）----
                if USE_VAL_TH and (lv is not None) and (len(lv) > 0):
                    p_val = _predict_pipe(pipe, X_all.iloc[lv])
                    fpr, tpr, thr = roc_curve(y_all[lv], p_val)
                    j = tpr - fpr
                    k = int(np.argmax(j))
                    thr_used = float(thr[k])
                    thr_rows.append({
                        "method": imp_name, "model": model_key,
                        "threshold": thr_used, "tpr": float(tpr[k]), "fpr": float(fpr[k])
                    })
                else:
                    thr_used = CLASS_THRESHOLD  # 固定阈值

                # 依次评估 5 个场景（阈值相关指标统一用 thr_used）
                def _eval_slice(idx, tag):
                    if idx is None or len(idx) == 0:
                        return
                    p = _predict_pipe(pipe, X_all.iloc[idx])
                    metrics = _evaluate_metrics(y_all[idx], p, threshold=thr_used)
                    row = {"method": imp_name, "model": model_key}; row.update(metrics)
                    collectors[tag].append(row)

                _eval_slice(lt, "internal_train")
                _eval_slice(lv, "internal_val")
                _eval_slice(ls, "internal_test")

                # 外部集：保证列一致
                def _ext_eval(df_ext: Optional[pd.DataFrame], tag: str):
                    if df_ext is None: return
                    need_cols = num_cols + bin_cols + mul_cols + [Y_NAME]
                    miss = [c for c in need_cols if c not in df_ext.columns]
                    if miss:
                        print(f"[WARN] {tag} 缺少列 {len(miss)}，跳过")
                        return
                    X_ext = _ensure_int_cats(df_ext[num_cols + bin_cols + mul_cols].copy(), mul_cols)
                    y_ext = df_ext[Y_NAME].astype(int).values
                    p = _predict_pipe(pipe, X_ext)
                    metrics = _evaluate_metrics(y_ext, p, threshold=thr_used)
                    row = {"method": imp_name, "model": model_key}; row.update(metrics)
                    collectors[tag].append(row)

                _ext_eval(ext_main, "external_main")
                _ext_eval(ext_tran, "external_transfer")

            except Exception as e:
                print(f"[WARN] {imp_name} × {model_key} 评估失败：{e}")

    # 4) 保存阈值（来自 VAL）
    if thr_rows:
        thr_df = pd.DataFrame(thr_rows).sort_values(["method","model"]).reset_index(drop=True)
        thr_csv = OUT_DIR / "val_thresholds.csv"
        thr_df.to_csv(thr_csv, index=False, encoding="utf-8-sig")
        print(f"[save] thresholds from val -> {thr_csv}")

    # 5) 保存汇总与作图
    metrics_order = ["AUROC", "AUPRC", "Accuracy", "Precision", "Recall",
                     "F1_Score", "Brier_Score", "Specificity", "NPV"]
    summary_frames = []
    for scen, rows in collectors.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        cols = ["method", "model"] + [m for m in metrics_order if m in df.columns]
        df = df[cols]
        out_csv = OUT_DIR / f"impute_model_selection_{scen}.csv"
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"[save] {scen} -> {out_csv}")
        df["scenario"] = scen
        summary_frames.append(df)

    if summary_frames:
        summary = pd.concat(summary_frames, ignore_index=True)
        out_sum = OUT_DIR / "impute_model_selection_summary.csv"
        summary.to_csv(out_sum, index=False, encoding="utf-8-sig")
        print(f"[save] summary -> {out_sum}")

        # 敏感性（模型内跨插补方法的 AUROC/AUPRC 波动）
        agg = (summary.groupby(["scenario", "model"])
               .agg(AUROC_spread=("AUROC", lambda x: float(np.max(x) - np.min(x))),
                    AUPRC_spread=("AUPRC", lambda x: float(np.max(x) - np.min(x)))))
        out_agg = OUT_DIR / "impute_sensitivity_by_model.csv"
        agg.to_csv(out_agg, encoding="utf-8-sig")
        print(f"[save] sensitivity_by_model -> {out_agg}")

        # 作图（可在 config.yaml 里 eval.make_plots: true/false 控制）
        MAKE_PLOTS = bool(EVAL_CFG.get("make_plots", True))
        if MAKE_PLOTS:
            _make_barplots(out_sum, OUT_DIR)

    print("[done] all evaluations finished.")

if __name__ == "__main__":
    main()
