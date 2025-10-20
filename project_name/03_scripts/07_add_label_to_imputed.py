# -*- coding: utf-8 -*-
"""
07_add_label_to_imputed.py
作用：
- 确保 02_processed_data/<VER>/imputed/charls_{split}_{method}_Xy.csv 都包含标签列 Y_NAME。
- 若某 method 的某 split 仍是“无 y”的历史文件（如 charls_train_knn5.csv / charls_val_mice_pmm_rep1.csv），
  则从 splits/charls_{split}.csv 读取标签列补上，保存为 *_Xy.csv。
- 若 train/val/test 三件套齐全，则拼接出 charls_all_{method}_Xy.csv（可选）。
不再依赖 frozen/charls_mean_mode_Xy.csv。
"""

from __future__ import annotations
import argparse
from pathlib import Path
import re
import sys
from collections import defaultdict

import pandas as pd
import yaml

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

def load_cfg():
    with open(repo_root() / "07_config" / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

ENC = "utf-8-sig"
XY_RE = re.compile(r"^charls_(train|val|test)_(.+)_Xy\.csv$", re.IGNORECASE)
LEGACY_PATTERNS = {
    "knn5"      : "charls_{split}_knn5.csv",
    "mice_bayes": "charls_{split}_mice_bayes.csv",
    "mice_cart" : "charls_{split}_mice_cart.csv",
    "mean_mode" : "charls_{split}_mean_mode.csv",
}
PMM_GLOB = "charls_*_mice_pmm_rep*.csv"

def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", type=str, default="mice_pmm_rep1",
                    help="与 06 保持一致的后缀（用于确认 *_Xy 是否存在；不是必需）")
    args = ap.parse_args()

    CFG = load_cfg()
    PROJECT_ROOT = Path(CFG["paths"]["project_root"])
    VER = CFG.get("run_id_out", CFG.get("run_id"))
    Y_NAME = (CFG.get("outcome", {}) or {}).get("name", "depression_bin")

    DATA_DIR = PROJECT_ROOT / "02_processed_data" / VER
    SPLITS = DATA_DIR / "splits"
    IMPUTED = DATA_DIR / "imputed"

    # 基础检查
    need = [SPLITS / f"charls_{sp}.csv" for sp in ("train", "val", "test")]
    for p in need:
        if not p.exists():
            print(f"[ERR] 缺少 splits：{p}"); sys.exit(1)
    if not IMPUTED.exists():
        print(f"[ERR] 缺少 imputed 目录：{IMPUTED}"); sys.exit(1)

    # 读 splits 的标签（权威来源）
    split_y = {}
    for sp in ("train", "val", "test"):
        df = pd.read_csv(SPLITS / f"charls_{sp}.csv")
        if Y_NAME not in df.columns:
            print(f"[ERR] splits/{sp} 不含标签列 {Y_NAME}"); sys.exit(1)
        split_y[sp] = df[Y_NAME].reset_index(drop=True)

    # 收集已带 y 的文件 & 仍无 y 的历史文件
    ready_xy = defaultdict(dict)     # ready_xy[method][split] -> Path
    legacy_no_y = defaultdict(dict)  # legacy_no_y[method][split] -> Path

    # 1) 已带 y 的 *_Xy.csv（优先使用）
    for p in IMPUTED.glob("charls_*_Xy.csv"):
        m = XY_RE.match(p.name)
        if not m: 
            continue
        sp, method = m.group(1).lower(), m.group(2)
        ready_xy[method][sp] = p

    # 2) 历史无 y：固定命名
    for method, pat in LEGACY_PATTERNS.items():
        for sp in ("train", "val", "test"):
            p = IMPUTED / pat.format(split=sp)
            if p.exists() and sp not in ready_xy[method]:
                legacy_no_y[method][sp] = p

    # 3) 历史无 y：PMM 系（带 rep）
    for p in IMPUTED.glob(PMM_GLOB):
        name = p.name.lower()
        m_split = re.search(r"charls_(train|val|test)_", name)
        m_rep   = re.search(r"rep(\d+)", name)
        if not m_split or not m_rep:
            continue
        sp = m_split.group(1)
        method = f"mice_pmm_rep{int(m_rep.group(1))}"
        if sp not in ready_xy[method]:
            legacy_no_y[method][sp] = p

    # 4) 给历史无 y 的文件补标签，生成 *_Xy.csv
    fixed_any = False
    for method, parts in sorted(legacy_no_y.items()):
        print(f"\n[proc] add labels for legacy method: {method}")
        for sp, p in sorted(parts.items()):
            df_x = pd.read_csv(p)
            if Y_NAME in df_x.columns and not df_x[Y_NAME].isna().all():
                # 已经有 y（某些手工产物），直接认定为 ready
                ready_xy[method][sp] = p
                print(f"  [skip] {p.name} 已含 {Y_NAME}")
                continue
            # 对齐长度（假定与 splits 同顺序；若你的数据有 ID，可在此改为按 ID merge）
            y = split_y[sp]
            if len(y) != len(df_x):
                n = min(len(y), len(df_x))
                print(f"  [warn] 长度不一致：y={len(y)} vs X={len(df_x)}；按 {n} 对齐。")
                y = y.iloc[:n].reset_index(drop=True)
                df_x = df_x.iloc[:n].reset_index(drop=True)
            df_xy = df_x.copy()
            df_xy[Y_NAME] = y
            out = p.with_name(p.stem + "_Xy.csv")
            ensure_dir(out)
            df_xy.to_csv(out, index=False, encoding=ENC)
            ready_xy[method][sp] = out
            fixed_any = True
            print(f"  [fix] {out.name} 已补标签")

    if fixed_any:
        print("\n[done] 历史无 y 的插补文件已全部补齐。")

    # 5) 三件套齐的 method 生成全量 charls_all_{method}_Xy.csv（可选）
    made_full = False
    for method, parts in sorted(ready_xy.items()):
        has = set(parts.keys())
        need = {"train", "val", "test"}
        if not need.issubset(has):
            missing = sorted(need - has)
            print(f"[info] method={method} 缺 {missing}，跳过 charls_all_{method}_Xy.csv")
            continue
        print(f"\n[stitch] method={method}")
        tr = pd.read_csv(parts["train"])
        va = pd.read_csv(parts["val"])
        te = pd.read_csv(parts["test"])

        # 统一列顺序（以出现最多列的文件为参考），再拼接
        cols = tr.columns
        if len(va.columns) > len(cols): cols = va.columns
        if len(te.columns) > len(cols): cols = te.columns
        def _align(df):
            x = df.copy()
            for c in cols:
                if c not in x.columns: x[c] = pd.NA
            return x[cols]
        full = pd.concat([_align(tr), _align(va), _align(te)], axis=0, ignore_index=True)
        out_full = IMPUTED / f"charls_all_{method}_Xy.csv"
        full.to_csv(out_full, index=False, encoding=ENC)
        print(f"  [save] {out_full.name}")
        made_full = True

    if not made_full:
        print("\n[info] 没有发现可拼全量的三件套；如果你的流程只用 train/val，这条提示可以忽略。")

    print("\n[done] 07 完成。输出目录：", IMPUTED)

if __name__ == "__main__":
    main()
