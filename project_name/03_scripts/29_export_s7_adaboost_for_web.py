# -*- coding: utf-8 -*-
"""
29_export_s7_adaboost_for_web.py — 导出第14步的 AdaBoost 管线到 JoyAge 网页工程

导出内容：
- 复制模型到 joyage-depression/saved_models/ADABOOST_pipeline.joblib（若存在校准版优先）
- 从 S7_thresholds_from_val.json 提取 adaboost 阈值，写入 threshold.json（可选）

注意：网页端加载逻辑已扩展可识别 ADABOOST_* 命名；也可直接重命名为 cesd_model_best_latest.joblib。
"""

from pathlib import Path
import json
import shutil
import yaml
import numpy as np
import pandas as pd
import joblib


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


CFG = yaml.safe_load(open(repo_root() / "07_config" / "config.yaml", "r", encoding="utf-8"))

PROJECT_ROOT = Path(CFG["paths"]["project_root"])
VER_OUT = CFG.get("run_id_out", CFG.get("run_id"))

S7_DIR = PROJECT_ROOT / "10_experiments" / VER_OUT / "final_eval_s7"
MODELS_DIR = S7_DIR / "models"
THR_JSON = S7_DIR / "S7_thresholds_from_val.json"
VER_IN = CFG.get("run_id_in", CFG.get("run_id"))
FROZEN = PROJECT_ROOT / "02_processed_data" / VER_IN / "frozen" / "charls_mean_mode_Xy.csv"
TRAIN_IDX = PROJECT_ROOT / "02_processed_data" / VER_IN / "splits" / "charls_train_idx.csv"

JOYAGE_DIR = PROJECT_ROOT / "joyage-depression"
JOY_SAVED = JOYAGE_DIR / "saved_models"


def pick_adaboost_model(models_dir: Path) -> Path | None:
    cand = [
        models_dir / "ADABOOST_calibrated_isotonic.joblib",
        models_dir / "ADABOOST_best_pipeline.joblib",
    ]
    for p in cand:
        if p.exists():
            return p
    return None


def read_thr(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        # 兼容 dict 或 {"adaboost": thr}
        if isinstance(data, dict):
            if "adaboost" in data:
                return float(data["adaboost"])
            # 直接就是阈值
            if "threshold" in data:
                return float(data["threshold"])
        return None
    except Exception:
        return None


def main():
    assert MODELS_DIR.exists(), f"未找到 S7 模型目录：{MODELS_DIR}"
    JOY_SAVED.mkdir(parents=True, exist_ok=True)

    src = pick_adaboost_model(MODELS_DIR)
    assert src is not None, "S7 未找到 AdaBoost 模型文件"

    dst = JOY_SAVED / "ADABOOST_pipeline.joblib"
    shutil.copy2(src, dst)
    print(f"[copy] {src} -> {dst}")

    thr = read_thr(THR_JSON)
    if thr is not None:
        thr_out = JOY_SAVED / "threshold.json"
        thr_out.write_text(json.dumps({"threshold": float(thr), "source": "Youden on validation"}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[save] threshold.json -> {thr_out}")
    else:
        print("[warn] 未找到阈值信息（可选）")

    # 导出背景样本（供 KernelExplainer 使用）
    try:
        assert FROZEN.exists() and TRAIN_IDX.exists(), "缺少冻结数据或训练索引，无法生成背景样本"
        df = pd.read_csv(FROZEN)
        y_name = (CFG.get("outcome", {}) or {}).get("name", "depression_bin")
        tr_idx = pd.read_csv(TRAIN_IDX).iloc[:, 0].astype(int).values
        # 从管线提取原始列
        pipe = joblib.load(dst)  # type: ignore
        pre = pipe.named_steps.get("pre")  # type: ignore
        cols = []
        if hasattr(pre, "transformers"):
            for _, _, cols_spec in pre.transformers:  # type: ignore
                if cols_spec is not None and cols_spec != "drop" and cols_spec != []:
                    cols.extend(list(cols_spec))
        sel = [c for c in cols if c in df.columns and c != y_name]
        if not sel:
            sel = [c for c in df.columns if c != y_name]
        Xtr = df.loc[tr_idx, sel]
        # 应用与训练一致的预处理，保存“变换后的背景矩阵”，确保与 SHAP 输入一致
        Xtr_t = pre.transform(Xtr)  # type: ignore
        if isinstance(Xtr_t, np.ndarray):
            bg = Xtr_t
        else:
            bg = np.asarray(Xtr_t)
        # 下采样最多500行
        if bg.shape[0] > 500:
            rng = np.random.default_rng(42)
            idx = rng.choice(bg.shape[0], size=500, replace=False)
            bg = bg[idx]
        np.save(JOY_SAVED / "X_bg_raw.npy", bg)
        print(f"[save] X_bg_raw.npy -> {JOY_SAVED / 'X_bg_raw.npy'}  (shape={bg.shape})")
    except Exception as e:
        print(f"[warn] 生成背景样本失败：{e}")

    print("[done] Exported AdaBoost pipeline for web.")


if __name__ == "__main__":
    main()


