# -*- coding: utf-8 -*-
"""
28_rf_shap_standalone_rewrite.py — 独立脚本：从指定CSV训练随机森林并输出SHAP图

特性：
- 不依赖 config.yaml 与索引文件；默认使用用户给定的冻结CSV路径
- 直接用原始数值特征训练RF（不做 ColumnTransformer/OneHot/标准化 预处理）
- 计算SHAP（兼容新旧接口），保存4类图：bar / beeswarm / waterfall / dependence
- 参数：--csv --y-name --out-dir --test-size --seed --drop-cols
- 新增：--topk --sample --formats --na --n-estimators --min-samples-leaf --max-depth

用法示例：
python "project_name/03_scripts/28_rf_shap_standalone_rewrite.py" \
  --csv "C:/Users/lenovo/Desktop/20250906 charls  klosa/project_name/02_processed_data/v2025-10-03/frozen/charls_mean_mode_Xy.csv"
"""

import os
os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path
from typing import List, Optional, Tuple
import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


DEFAULT_CSV = (
    "C:/Users/lenovo/Desktop/20250906 charls  klosa/project_name/02_processed_data/"
    "v2025-10-03/frozen/charls_mean_mode_Xy.csv"
)


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all columns are numeric for tree models (no preprocessing)."""
    out = df.copy()
    for c in out.columns:
        if not np.issubdtype(out[c].dtype, np.number):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def handle_na(df: pd.DataFrame) -> pd.DataFrame:
    # 占位，稍后依据 args.na 替换调用
    return df


def robust_tree_shap(clf, Xt, debug: bool = False):
    import shap
    Xt = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)
    # 优先“概率”空间；若版本不支持则回退
    try:
        explainer = shap.TreeExplainer(
            clf, feature_perturbation="interventional", model_output="probability"
        )
    except Exception:
        try:
            explainer = shap.TreeExplainer(clf, model_output="raw")
        except Exception:
            explainer = shap.TreeExplainer(clf)

    # 新接口优先
    try:
        exp = explainer(Xt, check_additivity=False)
        values = getattr(exp, "values", None)
        base_values = getattr(exp, "base_values", None)
        if values is not None:
            # 3D: [n, classes, features]
            if values.ndim == 3:
                base_value = float(np.mean(base_values[:, 1])) if np.ndim(base_values) == 2 else float(base_values)
                return explainer, values[:, 1, :], base_value
            # 2D: could be [n, features] OR [n, classes] (buggy API variants)
            if values.ndim == 2:
                if values.shape[1] == Xt.shape[1]:
                    base_value = float(np.mean(base_values)) if np.ndim(base_values) else float(base_values)
                    return explainer, values, base_value
                else:
                    # 形状与特征数不一致，回退旧接口
                    if debug:
                        print(f"[DEBUG] new API returned shape {values.shape}, expected features={Xt.shape[1]} -> fallback to old API")
                    raise ValueError("values shape mismatch")
    except Exception:
        pass

    # 旧接口回退
    sv = explainer.shap_values(Xt)
    if isinstance(sv, list):
        shap_values = np.asarray(sv[1]) if len(sv) > 1 else np.asarray(sv[0])
        ev = explainer.expected_value
        base = ev[1] if isinstance(ev, (list, np.ndarray)) and len(ev) > 1 else ev
    else:
        shap_values = np.asarray(sv)
        base = explainer.expected_value
    base = float(np.mean(base)) if np.ndim(base) else float(base)
    return explainer, shap_values, base


def guess_out_dir(csv_path: Path, explicit_out_dir: Optional[Path]) -> Path:
    if explicit_out_dir is not None:
        return explicit_out_dir
    # 若目录结构形如: .../02_processed_data/<ver>/frozen/charls_mean_mode_Xy.csv
    # 则输出目录设为:   .../10_experiments/<ver>/figs
    parts = list(csv_path.parts)
    try:
        idx = parts.index("02_processed_data")
        ver = parts[idx + 1]
        project_root = Path(*parts[:idx])
        out_dir = project_root / "10_experiments" / ver / "figs"
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir
    except Exception:
        out_dir = csv_path.parent / "shap_outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir


def main():
    parser = argparse.ArgumentParser(
        description="Train RF on CSV and export SHAP plots (standalone)"
    )
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV,
                        help="Path to dataset CSV (包含特征和'y'列)")
    parser.add_argument("--y-name", type=str, default="depression_bin",
                        help="目标变量列名")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="输出目录（默认自动推断）")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="测试集比例 (0,1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--drop-cols", type=str, nargs="*", default=["cesd"],
                        help="从特征集中丢弃的列，避免潜在泄露")
    parser.add_argument("--topk", type=int, default=10,
                        help="可视化展示的Top-K特征数")
    parser.add_argument("--sample", type=int, default=8000,
                        help="beeswarm下采样上限(行)")
    parser.add_argument("--formats", type=str, nargs="*", default=["png", "pdf"],
                        help="输出图像格式列表，例如: png pdf svg")
    parser.add_argument("--na", type=str, default="keep",
                        choices=["keep", "drop", "zero", "median"],
                        help="缺失值处理策略")
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--debug", action="store_true", help="打印与保存调试信息")
    # 以下参数在本脚本中仅定义一次，防止冲突
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到数据: {csv_path}")

    out_dir = guess_out_dir(csv_path, Path(args.out_dir) if args.out_dir else None)

    df = pd.read_csv(csv_path)
    if args.y_name not in df.columns:
        raise KeyError(f"'{args.y_name}' 不在 {csv_path}")

    # 丢弃潜在泄露列
    drop_cols = [c for c in (args.drop_cols or []) if c in df.columns]
    if drop_cols:
        keep_cols = [c for c in df.columns if c not in drop_cols]
        df = df[keep_cols].copy()

    # 切分（并保证全部为数值列）
    X = df.drop(columns=[args.y_name])
    y = df[args.y_name].astype(int).values
    Xtr, Xte, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )
    # 缺失值策略
    if args.na == "drop":
        tmp = pd.concat([Xtr, pd.Series(y_tr, name="__y")], axis=1).dropna(axis=0)
        y_tr = tmp.pop("__y").astype(int).values
        Xtr = tmp
        tmp = pd.concat([Xte, pd.Series(y_te, name="__y")], axis=1).dropna(axis=0)
        y_te = tmp.pop("__y").astype(int).values
        Xte = tmp
    elif args.na == "zero":
        Xtr = Xtr.fillna(0)
        Xte = Xte.fillna(0)
    elif args.na == "median":
        med = Xtr.median(numeric_only=True)
        Xtr = Xtr.fillna(med)
        Xte = Xte.fillna(med)

    # 强制转数值
    Xtr = coerce_numeric(Xtr)
    Xte = coerce_numeric(Xte)
    feat_names = list(Xtr.columns)

    rf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        max_depth=args.max_depth,
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=min(4, max((os.cpu_count() or 4) - 1, 1)),
    )
    rf.fit(Xtr, y_tr)

    if args.debug:
        print(f"[DEBUG] Xtr shape={Xtr.shape}, Xte shape={Xte.shape}")
        print(f"[DEBUG] n_features_in_={getattr(rf, 'n_features_in_', None)}")
        print(f"[DEBUG] feat_names count={len(feat_names)}")

    # 简要性能回显
    try:
        p_te = rf.predict_proba(Xte)[:, 1]
        uniq = np.unique(y_te)
        if len(uniq) > 1:
            auc = roc_auc_score(y_te, p_te)
            print(f"[AUC test] RF = {auc:.3f}")
        else:
            print("[AUC test] 单一类别，跳过AUC计算")
    except Exception as ex:
        print(f"[AUC test] 计算失败: {ex}")

    # SHAP 计算
    clf = rf
    explainer, shap_values, base = robust_tree_shap(clf, Xtr.values, debug=args.debug)
    Xtr_td = np.asarray(Xtr.values)

    if args.debug:
        print(f"[DEBUG] shap_values shape={getattr(shap_values, 'shape', None)}")

    # TopK选择
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    order = np.argsort(mean_abs)[::-1]
    k = int(min(args.topk, Xtr_td.shape[1]))
    idx_topk = order[:k]
    Sv = shap_values[:, idx_topk]
    Xk = Xtr_td[:, idx_topk]
    feat_topk = [feat_names[i] for i in idx_topk]

    if args.debug:
        top_pairs = [(feat_topk[i], float(mean_abs[idx_topk[i]])) for i in range(len(idx_topk))]
        print(f"[DEBUG] topk={k}, features={top_pairs[:10]}")

    import shap  # 放在此处以避免与后端冲突

    # A: bar (TopK)
    plt.figure(figsize=(9, 6), dpi=240)
    shap.summary_plot(Sv, features=Xk, feature_names=feat_topk,
                      plot_type="bar", show=False, max_display=k)
    plt.title("RF SHAP — mean(|SHAP|)")
    plt.tight_layout()
    for fmt in args.formats:
        plt.savefig(out_dir / f"RF_SHAP_bar.{fmt}", dpi=300)
    plt.close()

    # B: beeswarm (TopK, 可下采样)
    plt.figure(figsize=(9, 6), dpi=240)
    if Xk.shape[0] > args.sample:
        rng = np.random.default_rng(args.seed)
        sel = rng.choice(Xk.shape[0], size=args.sample, replace=False)
        X_vis = Xk[sel]
        Sv_vis = Sv[sel]
    else:
        X_vis = Xk
        Sv_vis = Sv
    shap.summary_plot(Sv_vis, features=X_vis, feature_names=feat_topk,
                      show=False, max_display=k)
    plt.title("RF SHAP — beeswarm (train)")
    plt.tight_layout()
    for fmt in args.formats:
        plt.savefig(out_dir / f"RF_SHAP_beeswarm.{fmt}", dpi=300)
    plt.close()

    # C: waterfall（取训练集中预测概率最高样本）
    try:
        p_tr = rf.predict_proba(Xtr)[:, 1]
        idx_top = int(np.argmax(p_tr))
        e = shap.Explanation(values=Sv[idx_top],
                             base_values=base,
                             data=Xk[idx_top],
                             feature_names=feat_topk)
        plt.figure(figsize=(8, 6), dpi=240)
        shap.plots.waterfall(e, max_display=k, show=False)
        plt.tight_layout()
        for fmt in args.formats:
            plt.savefig(out_dir / f"RF_SHAP_waterfall.{fmt}", dpi=300)
        plt.close()
    except Exception as ex:
        print("[WARN] waterfall 绘制失败:", ex)

    # D: dependence（Top-9）
    try:
        order9 = np.argsort(np.mean(np.abs(Sv), axis=0))[::-1][:9]
        rows, cols = 3, 3
        fig, axes = plt.subplots(rows, cols, figsize=(10, 8), dpi=240)
        for ax, j in zip(axes.ravel(), order9):
            shap.dependence_plot(j, Sv, Xk, feature_names=feat_topk, show=False, ax=ax)
        fig.suptitle("RF SHAP — dependence (top features)")
        fig.tight_layout()
        for fmt in args.formats:
            plt.savefig(out_dir / f"RF_SHAP_dependence.{fmt}", dpi=300)
        plt.close(fig)
    except Exception as ex:
        print("[WARN] dependence 绘制失败:", ex)

    # 导出CSV与metrics
    try:
        import json
        # 对齐长度（防御性）：
        names_all = list(feat_names)
        mean_abs_all = list(mean_abs)
        if len(names_all) != len(mean_abs_all):
            d_align = int(min(len(names_all), len(mean_abs_all)))
            names_all = names_all[:d_align]
            mean_abs_all = mean_abs_all[:d_align]
        df_imp = pd.DataFrame({"feature": names_all, "mean_abs_shap": mean_abs_all})
        df_imp.sort_values("mean_abs_shap", ascending=False, inplace=True)
        df_imp.to_csv(out_dir / "shap_mean_abs.csv", index=False)
        pd.DataFrame(Sv, columns=feat_topk).to_csv(out_dir / "shap_values_topk.csv", index=False)
        if hasattr(rf, "feature_importances_"):
            imp = list(rf.feature_importances_)
            if len(names_all) != len(imp):
                d_align2 = int(min(len(names_all), len(imp)))
                names_imp = names_all[:d_align2]
                imp = imp[:d_align2]
            else:
                names_imp = names_all
            df_rf = pd.DataFrame({"feature": names_imp, "rf_importance": imp})
            df_rf.sort_values("rf_importance", ascending=False, inplace=True)
            df_rf.to_csv(out_dir / "rf_importance.csv", index=False)
        uniq = np.unique(y_te)
        auc_val = float(roc_auc_score(y_te, p_te)) if len(uniq) > 1 else None
        summary = {
            "n_train": int(Xtr.shape[0]),
            "n_test": int(Xte.shape[0]),
            "pos_rate_train": float(np.mean(y_tr)),
            "pos_rate_test": float(np.mean(y_te)),
            "auc_test": auc_val,
            "topk": int(k)
        }
        (out_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.debug:
            debug_info = {
                "Xtr_shape": list(Xtr.shape),
                "Xte_shape": list(Xte.shape),
                "n_features_in": getattr(rf, 'n_features_in_', None),
                "feat_names_count": len(feat_names),
                "shap_values_shape": list(shap_values.shape) if hasattr(shap_values, 'shape') else None,
                "topk": int(k),
                "top_features": top_pairs[:k]
            }
            (out_dir / "debug_info.json").write_text(json.dumps(debug_info, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as ex:
        print("[WARN] 导出数据失败:", ex)

    print("[save]", out_dir)


if __name__ == "__main__":
    main()



