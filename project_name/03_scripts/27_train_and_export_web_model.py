#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
27_train_and_export_web_model.py
- 从 CSV 训练随机森林管线并导出 Web 所需工件
- 兼容两种情况：1) CSV 已含 y/label 等 ；2) 只有 CESD 分数，按阈值生成 y
- 自动对齐 train/val/test 的同名特征；导出 schema/threshold/bg 样本等
"""

import argparse, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
import joblib

# --------- 你可以把这里替换为你调参出来的最佳参数 ----------
RF_BEST_PARAMS = dict(
    n_estimators=600,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
# ---------------------------------------------------------------

# ★ 常见 CESD 与 标签列名（大小写不敏感）
CESD_CANDIDATE_COLS = ["cesd", "cesd_total", "cesd_score", "cesd10", "cesd_sum"]
LABEL_CANDIDATE_COLS = ["depression_bin", "y", "label", "target", "outcome", "is_depressed"]

def read_csv_smart(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV 不存在: {path}")
    df = pd.read_csv(p)
    if df.empty:
        raise ValueError(f"CSV 为空: {path}")
    return df

def _find_col_case_insensitive(df: pd.DataFrame, name: str) -> str | None:
    # 返回与 name 大小写匹配的真实列名
    low = {c.lower(): c for c in df.columns}
    return low.get(name.lower())

def _find_any_candidate(df: pd.DataFrame, candidates: list[str]) -> str | None:
    low = {c.lower(): c for c in df.columns}
    for k in candidates:
        if k.lower() in low:
            return low[k.lower()]
    return None

def ensure_target(df: pd.DataFrame, target_col: str | None, derive_from: str | None, threshold: float) -> tuple[pd.DataFrame, str]:
    """
    返回 (包含标签列的 DataFrame, 标签列名)，规则：
    1) 若 target_col 指向的列命中（大小写不敏感），就用它；
    2) 否则在 LABEL_CANDIDATE_COLS 里自动找一个现成标签列；
    3) 若还没有，则从 derive_from（或自动猜测 CESD 列）按阈值生成新列 'y'。
    - 最终保证返回的标签列名为真实存在于 df.columns 的名字。
    """
    df = df.copy()

    # 1) 显式 target 参数优先（大小写无关）
    if target_col:
        real = _find_col_case_insensitive(df, target_col)
        if real is not None:
            # 统一到 {0,1}
            if set(pd.Series(df[real]).dropna().unique()) - {0, 1}:
                if pd.api.types.is_bool_dtype(df[real]):
                    df[real] = df[real].astype(int)
                else:
                    df[real] = (pd.to_numeric(df[real], errors="coerce").fillna(0) >= 1).astype(int)
            return df, real

    # 2) 自动识别已有标签列
    auto_lab = _find_any_candidate(df, LABEL_CANDIDATE_COLS)
    if auto_lab is not None:
        if set(pd.Series(df[auto_lab]).dropna().unique()) - {0, 1}:
            if pd.api.types.is_bool_dtype(df[auto_lab]):
                df[auto_lab] = df[auto_lab].astype(int)
            else:
                df[auto_lab] = (pd.to_numeric(df[auto_lab], errors="coerce").fillna(0) >= 1).astype(int)
        return df, auto_lab

    # 3) 无现成标签：从 CESD 生成
    cand = None
    if derive_from:
        cand = _find_col_case_insensitive(df, derive_from)
    if cand is None:
        cand = _find_any_candidate(df, CESD_CANDIDATE_COLS)
    if cand is None:
        raise AssertionError(
            f"找不到标签列（候选 {LABEL_CANDIDATE_COLS}），也未发现可用于派生的 CESD 列（候选 {CESD_CANDIDATE_COLS}）。"
        )

    score = pd.to_numeric(df[cand], errors="coerce")
    if score.isna().all():
        raise ValueError(f"列 {cand} 中无有效数值，无法按阈值生成标签")
    df["y"] = (score >= threshold).astype(int)
    return df, "y"

def detect_feature_types(df: pd.DataFrame, exclude: list[str]) -> tuple[list[str], list[str]]:
    """
    简单类型判定：
    - 对象/类别 -> 分类
    - 其余数值列中，若唯一值数很少（<=15），也当作分类（常见0/1/编码）
    """
    cat_cols, num_cols = [], []
    for c in df.columns:
        if c in exclude:
            continue
        s = df[c]
        if s.dtype.name in ("object", "category"):
            cat_cols.append(c)
        else:
            nunique = s.nunique(dropna=True)
            if nunique <= 15:
                cat_cols.append(c)
            else:
                num_cols.append(c)
    return cat_cols, num_cols

def align_columns(df_tr: pd.DataFrame, df_va: pd.DataFrame | None, df_te: pd.DataFrame | None, target_col: str):
    """取交集列（含 target_col），并保证相同顺序"""
    dfs = [d for d in [df_tr, df_va, df_te] if d is not None]
    inter = set(dfs[0].columns)
    for d in dfs[1:]:
        inter &= set(d.columns)
    if target_col not in inter:
        raise AssertionError(f"对齐后仍缺少目标列 {target_col}，请检查输入文件是否一致。")
    inter = [c for c in dfs[0].columns if c in inter]  # 按训练列顺序
    return df_tr[inter], (df_va[inter] if df_va is not None else None), (df_te[inter] if df_te is not None else None)


def youden_threshold(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    k = int(np.argmax(j))
    return float(thr[k]), float(tpr[k]), float(fpr[k])

def build_pipeline(cat_cols, num_cols, rf_params, model_type="rf"):
    """构建预处理+模型管道，根据模型类型使用不同的预处理策略：
    - 树模型（rf等）：直接转float，不做标准化和One-Hot，仅做缺失值填充
    - 线性模型（lr等）：标准化连续变量，One-Hot分类变量
    """
    # 树模型使用简单预处理
    if model_type.lower() in ["rf", "extra_trees", "gb", "adaboost", "lgb", "xgb", "catboost"]:
        pre = ColumnTransformer(
            transformers=[
                ("num", Pipeline(steps=[
                    ("imp", SimpleImputer(strategy="median")),
                    # 不做标准化，直接保持原始尺度
                ]), num_cols),
                ("cat", Pipeline(steps=[
                    ("imp", SimpleImputer(strategy="most_frequent")),
                    # 不做One-Hot，直接保持原始编码
                ]), cat_cols),
            ],
            remainder="drop"
        )
    # 线性模型使用标准预处理
    else:
        pre = ColumnTransformer(
            transformers=[
                ("num", Pipeline(steps=[
                    ("imp", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=False)),
                ]), num_cols),
                ("cat", Pipeline(steps=[
                    ("imp", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False)),
                ]), cat_cols),
            ],
            remainder="drop"
        )
    
    # 根据模型类型创建不同的模型
    if model_type.lower() == "rf":
        model = RandomForestClassifier(**rf_params)
    else:
        # 可以扩展支持其他模型类型
        model = RandomForestClassifier(**rf_params)
    
    pipe = Pipeline(steps=[("pre", pre), ("clf", model)])
    return pipe

def main():
    ap = argparse.ArgumentParser()
    # ★ 默认 target 改为 None：允许自动识别 depression_bin / y 等
    ap.add_argument("--train", required=True, help="训练 CSV 路径")
    ap.add_argument("--val", required=True, help="验证 CSV 路径")
    ap.add_argument("--test", default=None, help="测试 CSV 路径（可选）")
    ap.add_argument("--target", default=None, help="目标列名；留空则自动识别（depression_bin/y/label/target/outcome/…）")
    ap.add_argument("--derive_from", default=None, help="当缺标签时，从此列派生（如 cesd/cesd_total/cesd10）")
    ap.add_argument("--cesd_threshold", type=float, default=10.0, help="CESD 阈值，默认 10")
    ap.add_argument("--outdir", required=True, help="输出目录")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[load] train={args.train}")
    df_tr = read_csv_smart(args.train)
    print(f"[load] val  ={args.val}")
    df_va = read_csv_smart(args.val)

    df_te = None
    if args.test:
        print(f"[load] test ={args.test}")
        df_te = read_csv_smart(args.test)

    # ★ 确保存在且确定标签列名（自动识别 depression_bin/y/...）
    df_tr, target_tr = ensure_target(df_tr, args.target, args.derive_from, args.cesd_threshold)
    df_va, target_va = ensure_target(df_va, args.target or target_tr, args.derive_from, args.cesd_threshold)
    if df_te is not None:
        df_te, target_te = ensure_target(df_te, args.target or target_tr, args.derive_from, args.cesd_threshold)
    # ★ 用训练集识别到的真实标签名为准
    target_col = target_tr

    # 对齐列
    df_tr, df_va, df_te = align_columns(df_tr, df_va, df_te, target_col=target_col)

    # 特征/标签
    X_tr, y_tr = df_tr.drop(columns=[target_col]), df_tr[target_col].astype(int).values
    X_va, y_va = df_va.drop(columns=[target_col]), df_va[target_col].astype(int).values

    # 特征类型
    cat_cols, num_cols = detect_feature_types(pd.concat([X_tr, X_va], axis=0), exclude=[])
    print(f"[schema] target='{target_col}' | cat={len(cat_cols)} num={len(num_cols)}")

    # 管线 & 训练
    # 默认使用RF模型，不做标准化和One-Hot
    pipe = build_pipeline(cat_cols, num_cols, RF_BEST_PARAMS, model_type="rf")
    pipe.fit(X_tr, y_tr)

    # 评估（train/val，若有 test 也评）
    prob_tr = pipe.predict_proba(X_tr)[:, 1]
    prob_va = pipe.predict_proba(X_va)[:, 1]
    auc_tr = roc_auc_score(y_tr, prob_tr)
    auc_va = roc_auc_score(y_va, prob_va)
    thr, tpr, fpr = youden_threshold(y_va, prob_va)
    y_pred_thr = (prob_va >= thr).astype(int)
    acc_va = accuracy_score(y_va, y_pred_thr)

    print(f"[perf] AUC(train)={auc_tr:.3f} | AUC(val)={auc_va:.3f} | thr*={thr:.3f} | TPR={tpr:.3f} | FPR={fpr:.3f} | ACC(val)={acc_va:.3f}")

    test_metrics = None
    if df_te is not None:
        X_te, y_te = df_te.drop(columns=[target_col]), df_te[target_col].astype(int).values
        prob_te = pipe.predict_proba(X_te)[:, 1]
        auc_te = roc_auc_score(y_te, prob_te)
        acc_te = accuracy_score(y_te, (prob_te >= thr).astype(int))
        print(f"[perf] AUC(test)={auc_te:.3f} | ACC(test@thr*)={acc_te:.3f}")
        test_metrics = dict(auc_test=float(auc_te), acc_test=float(acc_te), n_test=int(len(X_te)))

    # 背景样本（原始特征空间）
    bg = X_tr.sample(min(500, len(X_tr)), random_state=42)
    np.save(outdir / "X_bg_raw.npy", bg.to_numpy())

    # 导出模型
    joblib.dump(pipe, outdir / "final_rf_pipeline.joblib")

    # 导出 schema
    schema = {
        "ver_out": "rf_web_export_v1",
        "target": target_col,  # ★ 记录真实标签名
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "cat_values": {c: np.sort(pd.Series(X_tr[c]).dropna().unique()).tolist() for c in cat_cols},
    }
    Path(outdir / "schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")

    # 导出阈值与性能
    thr_json = {
        "threshold": thr,
        "metric": "Youden-J on val",
        "rf_params": RF_BEST_PARAMS,
        "n_train": int(len(X_tr)),
        "n_val": int(len(X_va)),
        "auc_train": float(auc_tr),
        "auc_val": float(auc_va),
        "acc_val": float(acc_va),
    }
    if test_metrics:
        thr_json.update(test_metrics)
    Path(outdir / "threshold.json").write_text(json.dumps(thr_json, ensure_ascii=False, indent=2), encoding="utf-8")

    # 导出训练列名（原始空间）
    Path(outdir / "train_columns.json").write_text(json.dumps(list(X_tr.columns), ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] artifacts saved to: {outdir.resolve()}")
    print("       - final_rf_pipeline.joblib")
    print("       - schema.json")
    print("       - threshold.json")
    print("       - train_columns.json")
    print("       - X_bg_raw.npy")

if __name__ == "__main__":
    main()
