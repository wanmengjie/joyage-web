# -*- coding: utf-8 -*-
"""
02_utils_thresholds.py

功能：
- 计算验证集的 Youden 阈值（J = TPR - FPR）
- 保存/读取阈值 JSON（强制来源为 val，避免泄漏）
- 兼容两种运行方式：
  1) 包方式：python -m 03_scripts.02_utils_thresholds
  2) 脚本直跑：python 03_scripts/02_utils_thresholds.py

项目结构（假定）：
project_root/
  03_scripts/
    __init__.py            <-- 建议添加（空文件即可）
    config_io.py           <-- 包含 dump_json/load_json
    02_utils_thresholds.py <-- 本文件
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

# ========== 兼容导入层 ==========
# 优先包内相对导入；失败则把当前目录(03_scripts)加入 sys.path 再绝对导入
try:
    # 当以包方式(import 或 python -m)运行时
    from .config_io import dump_json, load_json  # type: ignore
except Exception:
    # 当脚本直跑时（没有父包），把当前脚本所在目录加入 sys.path
    THIS_FILE = Path(__file__).resolve()
    THIS_DIR = THIS_FILE.parent  # .../03_scripts
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    from config_io import dump_json, load_json  # type: ignore


def youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """
    基于 Youden J 统计量选择最优阈值：J = TPR - FPR

    Parameters
    ----------
    y_true : np.ndarray
        真实标签（0/1）
    y_prob : np.ndarray
        预测为正类的概率（0~1）

    Returns
    -------
    thr : float
        使 J 最大的概率阈值
    meta : Dict[str, float]
        对应该阈值处的 tpr 与 fpr
    """
    from sklearn.metrics import roc_curve

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if y_true.ndim != 1 or y_prob.ndim != 1 or y_true.shape[0] != y_prob.shape[0]:
        raise ValueError("y_true 与 y_prob 需为等长一维数组")

    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    k = int(np.argmax(j))
    return float(thr[k]), {"tpr": float(tpr[k]), "fpr": float(fpr[k])}


def save_threshold_from_val(path: Path, ver: str, thr: float, method: str = "youden") -> None:
    """
    保存阈值 JSON（来源固定为 val）
    JSON 字段：
      - source_split: "val"
      - method: 计算方法（默认 youden）
      - threshold: 阈值
      - ver: 版本号/运行标识
    """
    data = {
        "source_split": "val",
        "method": method,
        "threshold": float(thr),
        "ver": ver,
    }
    dump_json(data, path)


def load_threshold_from_val(path: Path) -> float:
    """
    读取阈值 JSON，并强制校验其来源为 val
    """
    cfg = load_json(path)
    if cfg.get("source_split") != "val":
        raise ValueError("阈值来源必须是 val（禁止 test/external 重新挑阈值）")
    return float(cfg["threshold"])


# -------------------------
# 自检：脚本直跑时的最小单元测试
# -------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(2025)
    y_true_demo = rng.integers(0, 2, size=256)
    y_prob_demo = rng.random(256)
    thr_demo, meta_demo = youden_threshold(y_true_demo, y_prob_demo)
    print("[02_utils_thresholds] self-check OK")
    print(f"  youden thr = {thr_demo:.4f}, TPR={meta_demo['tpr']:.3f}, FPR={meta_demo['fpr']:.3f}")
