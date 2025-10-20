# -*- coding: utf-8 -*-
"""
29_make_adaboost_threshold.py — 为网页导出的 AdaBoost 管线计算验证集阈值(Youden)

输入：
- project_name/joyage-depression/saved_models/ADABOOST_pipeline.joblib
- 冻结数据与 splits（来自 config.yaml 对应 run_id_in）

输出：
- project_name/joyage-depression/saved_models/threshold.json
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_curve


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def youden_threshold(y, p) -> float:
    fpr, tpr, th = roc_curve(y.astype(int), p.astype(float))
    j = tpr - fpr
    k = int(np.nanargmax(j))
    return float(th[k])


def main():
    CFG = yaml.safe_load(open(repo_root() / "07_config" / "config.yaml", "r", encoding="utf-8"))

    PROJECT_ROOT = Path(CFG["paths"]["project_root"])
    VER_IN = CFG.get("run_id_in", CFG.get("run_id"))

    FROZEN = PROJECT_ROOT / "02_processed_data" / VER_IN / "frozen" / "charls_mean_mode_Xy.csv"
    VAL_IDX = PROJECT_ROOT / "02_processed_data" / VER_IN / "splits" / "charls_val_idx.csv"

    JOY_SAVED = PROJECT_ROOT / "joyage-depression" / "saved_models"
    PIPE_PATH = JOY_SAVED / "ADABOOST_pipeline.joblib"

    assert PIPE_PATH.exists(), f"未找到导出的 AdaBoost 管线：{PIPE_PATH}"
    assert FROZEN.exists(), f"未找到冻结数据：{FROZEN}"
    assert VAL_IDX.exists(), f"未找到验证索引：{VAL_IDX}"

    pipe = joblib.load(PIPE_PATH)
    df = pd.read_csv(FROZEN)
    y_name = (CFG.get("outcome", {}) or {}).get("name", "depression_bin")
    assert y_name in df.columns, f"{y_name} 不在 {FROZEN}"

    va_idx = pd.read_csv(VAL_IDX).iloc[:, 0].astype(int).values

    # 从 ColumnTransformer 提取其期望的原始列集
    pre = pipe.named_steps.get("pre")
    cols = []
    if hasattr(pre, "transformers"):
        for _, _, cols_spec in pre.transformers:
            if cols_spec is not None and cols_spec != "drop" and cols_spec != []:
                cols.extend(list(cols_spec))
    # 去重并保持顺序
    seen = set(); sel_cols = []
    for c in cols:
        if c not in seen and c in df.columns:
            sel_cols.append(c); seen.add(c)
    if not sel_cols:
        sel_cols = [c for c in df.columns if c != y_name]

    Xva = df.loc[va_idx, sel_cols].copy()
    yva = df.loc[va_idx, y_name].astype(int).values

    # 预测概率
    p_va = pipe.predict_proba(Xva)[:, 1]
    thr = youden_threshold(yva, p_va)

    out = JOY_SAVED / "threshold.json"
    out.write_text(json.dumps({"threshold": float(thr), "source": "Youden on val"}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] threshold.json -> {out} (thr={thr:.4f})")


if __name__ == "__main__":
    main()


