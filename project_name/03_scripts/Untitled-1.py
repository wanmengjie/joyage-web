# -*- coding: utf-8 -*-
"""
RF + SHAP (sklearn Iris, multiclass)
------------------------------------
A complete, self-contained script that trains a RandomForestClassifier on the
built-in Iris dataset and generates SHAP explanations and evaluation artifacts.

Outputs (under ./outputs/):
- metrics.txt: Accuracy, macro Precision/Recall/F1, macro ROC-AUC(OVR)
- cmatrix.png: Confusion matrix
- rf_feature_importance.png: RandomForest feature importances
- shap_summary_beeswarm_class{i}_<label>.png: SHAP beeswarm per class
- shap_summary_bar_class{i}_<label>.png: SHAP bar per class
- shap_dependence_<topfeat>_class{i}_<label>.png: SHAP dependence for top feat per class

Dependencies: scikit-learn, shap, matplotlib, numpy, pandas
Run:  python rf_shap_iris.py
"""

import os
os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
import shap


# -------------------------- io utils --------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def savefig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


# -------------------------- data ------------------------------
def load_iris_dataset():
    data = load_iris(as_frame=True)
    X = data.frame.drop(columns=["target"])  # 4 numeric features
    y = data.frame["target"]                 # 3 classes: 0,1,2
    target_names = list(data.target_names)
    feature_names = list(X.columns)
    return X, y, feature_names, target_names, "iris"


# ------------------------ pipeline ----------------------------
def build_pipeline(n_estimators=300, max_depth=None, random_state=42):
    scaler = StandardScaler()
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,
        random_state=random_state,
    )
    pipe = Pipeline([("scaler", scaler), ("rf", rf)])
    return pipe


# --------------------- plots & metrics ------------------------
def plot_confusion_matrix(cm: np.ndarray, labels: list, outpath: Path):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    # text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    savefig(outpath)


def plot_rf_importance(model: RandomForestClassifier, feat_names: list, outpath: Path):
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(idx)), importances[idx])
    ax.set_xticks(range(len(idx)))
    ax.set_xticklabels([feat_names[i] for i in idx], rotation=30, ha="right")
    ax.set_ylabel("Feature importance")
    ax.set_title("RandomForest — Gini importances")
    savefig(outpath)


# -------------------------- SHAP ------------------------------
def compute_and_plot_shap(pipe: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame,
                          class_names: list, outdir: Path, dep_topk: int = 1):
    """Compute SHAP with TreeExplainer and create per-class plots.
    Robust to SHAP version differences and shape mismatches.
    """
    ensure_dir(outdir)

    rf = pipe.named_steps["rf"]
    scaler = pipe.named_steps["scaler"]

    # Use the model's actual numeric input to SHAP
    X_test_t = scaler.transform(X_test)
    X_df_full = pd.DataFrame(X_test_t, columns=X_test.columns)

    explainer = shap.TreeExplainer(rf)
    shap_vals_raw = explainer.shap_values(X_test_t)

    # Normalize to list-of-arrays per class
    if not isinstance(shap_vals_raw, (list, tuple)):
        shap_vals_list = [np.asarray(shap_vals_raw)]
    else:
        shap_vals_list = [np.asarray(v) for v in shap_vals_raw]

    # per-class beeswarm / bar / dependence (with shape guards)
    for i, cname in enumerate(class_names):
        # Some SHAP versions may return fewer entries than classes
        if i >= len(shap_vals_list):
            continue
        vals_i = np.asarray(shap_vals_list[i])
        # Align feature dimensions if SHAP returns truncated features
        n_feat = vals_i.shape[1]
        X_df = X_df_full.iloc[:, :n_feat]

        # beeswarm
        plt.figure(figsize=(9, 6))
        shap.summary_plot(vals_i, X_df, show=False)
        savefig(outdir / f"shap_summary_beeswarm_class{i}_{cname}.png")

        # bar (mean |SHAP|)
        plt.figure(figsize=(9, 6))
        shap.summary_plot(vals_i, X_df, plot_type="bar", show=False)
        savefig(outdir / f"shap_summary_bar_class{i}_{cname}.png")

        # dependence for top-k features
                # dependence for top-k features (robust index handling)
        mean_abs = np.mean(np.abs(vals_i), axis=0)
        mean_abs = np.array(mean_abs).reshape(-1)  # ensure 1D
        top_idx_order = np.argsort(mean_abs)[::-1]
        top_idx_order = np.array(top_idx_order).reshape(-1)[:max(1, dep_topk)]
        for j in range(len(top_idx_order)):
            idx_j = int(top_idx_order[j])  # guaranteed scalar
            top_feat = X_df.columns[idx_j]
            color_feat = None
            if j + 1 < len(top_idx_order):
                color_feat = X_df.columns[int(top_idx_order[j + 1])]
            plt.figure(figsize=(7, 5))
            shap.dependence_plot(top_feat, vals_i, X_df, interaction_index=color_feat, show=False)
            savefig(outdir / f"shap_dependence_{top_feat}_class{i}_{cname}.png")


# --------------------------- main ----------------------------
def main():
    outdir = Path("outputs")
    ensure_dir(outdir)

    # data
    X, y, feat_names, class_names, dname = load_iris_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # model
    pipe = build_pipeline(n_estimators=300, max_depth=None, random_state=42)
    pipe.fit(X_train, y_train)

    # predictions
    y_pred = pipe.predict(X_test)
    # probs for ROC-AUC (OVR)
    y_proba = pipe.predict_proba(X_test)

    # metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")
    # macro ROC-AUC (OVR). If a class has a single label in y_test, roc_auc_score may error; guard it.
    try:
        auc_macro_ovr = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
    except Exception:
        auc_macro_ovr = float("nan")

    with open(outdir / "metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"Dataset: {dname}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision(macro): {prec:.4f}\n")
        f.write(f"Recall(macro): {rec:.4f}\n")
        f.write(f"F1(macro): {f1:.4f}\n")
        f.write(f"ROC-AUC(macro, OVR): {auc_macro_ovr:.4f}\n")
    print("[SAVE] metrics.txt written.")

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, class_names, outdir / "cmatrix.png")

    # RF feature importances (Gini)
    rf = pipe.named_steps["rf"]
    plot_rf_importance(rf, feat_names, outdir / "rf_feature_importance.png")

    # SHAP explanations (per class)
    compute_and_plot_shap(pipe, X_train, X_test, class_names, outdir, dep_topk=1)
    print("[SAVE] SHAP plots saved to:", outdir)


if __name__ == "__main__":
    main()
