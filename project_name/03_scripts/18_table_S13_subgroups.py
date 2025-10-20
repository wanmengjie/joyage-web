# -*- coding: utf-8 -*-
"""
18_table_S13_subgroups.py — 训练集分层(S13)
- 基于 VER_IN 的 mean_mode 全量表 + 训练索引，保证与 S7 概率严格对齐
- 自动统一列名小写，容错 drinkev/smokev 等不同拼写
- 阈值优先用 S7_thresholds_from_val.json，缺失回退 0.5
"""

from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import yaml

from sklearn.metrics import (
    accuracy_score, roc_auc_score, brier_score_loss, f1_score, confusion_matrix
)

# ================== 配置与路径 ==================
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

CFG = yaml.safe_load(open(repo_root() / "07_config" / "config.yaml", "r", encoding="utf-8"))

PROJECT_ROOT = Path(CFG.get("paths", {}).get("project_root", repo_root()))
VER_IN  = CFG.get("run_id_in", CFG.get("run_id", "v2025-10-01"))
VER_OUT = CFG.get("run_id_out", CFG.get("run_id", "v2025-10-01"))
Y_NAME  = (CFG.get("outcome", {}) or {}).get("name", "y")

# 数据（保证与 S7 概率一致的来源/顺序）
MEAN_MODE_XY = PROJECT_ROOT / "02_processed_data" / VER_IN / "frozen" / "charls_mean_mode_Xy.csv"
TRAIN_IDX    = PROJECT_ROOT / "02_processed_data" / VER_IN / "splits" / "charls_train_idx.csv"

DIR_S7 = PROJECT_ROOT / "10_experiments" / VER_OUT / "final_eval_s7"
PROBS_TRAIN = DIR_S7 / "S7_probs_internal_train.csv"
THRESHOLDS_JSON = DIR_S7 / "S7_thresholds_from_val.json"

OUT_DIR = PROJECT_ROOT / "10_experiments" / VER_OUT / "subgroup_S13"
OUT_PATH = OUT_DIR / "S13_subgroup_internal_train.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ================== 小工具 ==================
def load_thresholds(path: Path):
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return {str(k).lower(): float(v) for k, v in raw.items()}
        except Exception:
            return {}
    return {}

def sanitize_proba(p):
    p = np.asarray(p, dtype=float)
    p = np.where(np.isfinite(p), p, np.nan)
    if np.isnan(p).any():
        med = np.nanmedian(p)
        p = np.where(np.isnan(p), med, p)
    return np.clip(p, 1e-12, 1-1e-12)

def safe_auc(y_true, p_hat):
    try:
        return float(roc_auc_score(y_true, p_hat))
    except Exception:
        return np.nan

def bin_metrics(y_true, p_hat, thr=0.5):
    y_true = np.asarray(y_true).astype(int)
    p_hat  = np.asarray(p_hat, dtype=float)
    y_pred = (p_hat >= float(thr)).astype(int)

    acc   = accuracy_score(y_true, y_pred)
    auc   = safe_auc(y_true, p_hat)
    brier = brier_score_loss(y_true, p_hat)

    # labels=[0,1] 确保 2x2 即便单类缺失
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv  = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    f1   = f1_score(y_true, y_pred) if (tp + fp > 0 and tp + fn > 0) else np.nan

    return dict(
        AUC=auc, Brier=brier, Accuracy=acc,
        Sensitivity=sens, Specificity=spec, Precision=prec, NPV=npv, F1=f1
    )

def fmt(x, k=4):
    if pd.isna(x):
        return ""
    try:
        return f"{float(x):.{k}f}"
    except Exception:
        return str(x)

# ================== 读概率（保留 y） ==================
def load_probs_train(probs_csv: Path) -> pd.DataFrame:
    if not probs_csv.exists():
        raise FileNotFoundError(f"未找到训练集概率：{probs_csv}\n请先运行 14_final_eval_s7.py 生成 S7 概率。")
    df = pd.read_csv(probs_csv)

    # 统一 y 列名（或配置里的 outcome.name）
    y_col = None
    for c in [Y_NAME, "y", "y_x", "y_y", "label", "target"]:
        if c in df.columns:
            y_col = c; break
    if y_col is None:
        raise KeyError(f"{probs_csv} 中找不到标签列（尝试了 {Y_NAME}/y/label/target 等）")

    if y_col != "y":
        df = df.rename(columns={y_col: "y"})

    p_cols = [c for c in df.columns if c.startswith("p_")]
    if not p_cols:
        raise RuntimeError(f"{probs_csv} 中未发现概率列（以 'p_' 开头）")

    for c in p_cols:
        df[c] = sanitize_proba(df[c].values)
    return df[["y"] + p_cols].reset_index(drop=True)

# ================== 分层变量（基于 VER_IN 的训练集原始行） ==================
def load_train_rows() -> pd.DataFrame:
    if not MEAN_MODE_XY.exists():
        raise FileNotFoundError(f"找不到 mean_mode 数据：{MEAN_MODE_XY}")
    if not TRAIN_IDX.exists():
        raise FileNotFoundError(f"找不到训练索引：{TRAIN_IDX}")
    df_all = pd.read_csv(MEAN_MODE_XY)
    tr_idx = pd.read_csv(TRAIN_IDX).iloc[:,0].astype(int).values
    df_tr = df_all.iloc[tr_idx].copy()
    # 统一小写列名，避免 drinkev 等拼写大小写差异
    df_tr.columns = [c.lower() for c in df_tr.columns]
    return df_tr

def build_subgroup_columns(meta_tr: pd.DataFrame) -> pd.DataFrame:
    m = meta_tr.copy()

    # --- 性别 ---
    if "ragender" in m.columns:
        v = pd.to_numeric(m["ragender"], errors="coerce")
        m["ragender"] = v.map({1.0: "Male", 2.0: "Female"})

    # --- 年龄分箱 ---
    if "agey" in m.columns:
        bins   = [-1e9, 60, 70, 80, 1e9]
        labels = ["<60", "60–69", "70–79", "≥80"]
        m["age_group"] = pd.cut(pd.to_numeric(m["agey"], errors="coerce"), bins=bins, labels=labels, right=False)

    # --- 城乡 ---
    if "rural" in m.columns:
        r = pd.to_numeric(m["rural"], errors="coerce")
        m["rural"] = r.map({0.0: "Urban", 1.0: "Rural"})

    # --- 教育（1/2/3 视为 Low/Middle/High） ---
    if "raeducl" in m.columns:
        raw = pd.to_numeric(m["raeducl"], errors="coerce")
        if set(pd.Series(raw.dropna().astype(int).unique())).issubset({1,2,3}):
            m["raeducl"] = raw.astype("Int64").map({1: "Low", 2: "Middle", 3: "High"})
        else:
            m["raeducl"] = m["raeducl"].astype(str)

    # --- ADL ---
    if "adlfive" in m.columns:
        v = pd.to_numeric(m["adlfive"], errors="coerce")
        m["adl_status"] = np.where(v.fillna(0) > 0, "Limited", "Not limited")

    # --- 自评健康/活动受限（保持原分档为字符串） ---
    if "shlta" in m.columns:
        m["shlta"] = m["shlta"].astype(str)
    if "hlthlm" in m.columns:
        m["hlthlm"] = m["hlthlm"].astype(str)

    # --- 合并症计数 ---
    disease_cols = [c for c in ["hibpe","diabe","cancre","lunge","hearte","stroke","arthre","livere"] if c in m.columns]
    if disease_cols:
        mm = m[disease_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        comorb_cnt = mm.sum(axis=1)
        m["comorb_cnt"] = comorb_cnt
        m["comorb_cat"] = pd.cut(comorb_cnt, bins=[-1e9, 0.5, 1.5, 1e9], labels=["0", "1", "≥2"], right=True)

    # --- 吸烟/饮酒 Ever ---
    if "smokev" in m.columns:
        v = pd.to_numeric(m["smokev"], errors="coerce")
        m["smoke_ever"] = np.where(v.fillna(0) > 0, "Ever", "Never")
    # 兼容 drinkev 拼写
    drink_col = None
    for cand in ["drinkev", "drinkeV".lower(), "drinKev".lower()]:
        if cand in m.columns:
            drink_col = cand; break
    if drink_col is not None:
        v = pd.to_numeric(m[drink_col], errors="coerce")
        m["drink_ever"] = np.where(v.fillna(0) > 0, "Ever", "Never")

    return m

# ================== 主流程 ==================
def main():
    # 1) 概率（含 y）
    df_probs = load_probs_train(PROBS_TRAIN)
    thr_map = load_thresholds(THRESHOLDS_JSON)  # 可能为空 -> 回退 0.5
    p_cols = [c for c in df_probs.columns if c.startswith("p_")]
    model_keys = [c.replace("p_", "").lower() for c in p_cols]

    # 2) 训练集原始行（VER_IN，保证顺序一致）
    df_tr = load_train_rows()
    if Y_NAME.lower() not in df_tr.columns:
        raise KeyError(f"训练集表中找不到标签列 {Y_NAME!r}（小写后匹配为 {Y_NAME.lower()!r}）")
    # 校验 y 一致性（数量级检查）
    if len(df_probs) != len(df_tr):
        raise RuntimeError(f"S7 概率与训练集行数不一致：probs={len(df_probs)}, train={len(df_tr)}")
    y_s7  = df_probs["y"].astype(int).values
    y_tru = pd.to_numeric(df_tr[Y_NAME.lower()], errors="coerce").fillna(0).astype(int).values
    if int(y_s7.sum()) != int(y_tru.sum()):
        print(f"[warn] y(概率文件) 与 y(训练集) 的阳性数不一致：{y_s7.sum()} vs {y_tru.sum()}（可能来源不同插补/过滤；将以概率文件 y 为准）")

    # 3) 构造分层列
    meta = build_subgroup_columns(df_tr)

    # 4) 拼接（严格同序）
    df_all = pd.concat([df_probs, meta.reset_index(drop=True)], axis=1)
    assert "y" in df_all.columns

    # 5) 分层列（存在才用）
    group_vars = [
        ("ragender", "Sex"),
        ("age_group", "Age"),
        ("rural", "Residence"),
        ("raeducl", "Education"),
        ("adl_status", "ADL"),
        ("shlta", "SelfRatedHealth"),
        ("hlthlm", "ActivityLimit"),
        ("comorb_cat", "Comorbidity"),
        ("smoke_ever", "Smoking"),
        ("drink_ever", "Drinking"),
    ]
    group_vars = [(c, lab) for c, lab in group_vars if c in df_all.columns]

    # 6) 逐组逐模型计算
    rows = []
    for gcol, glabel in group_vars:
        # 固定顺序：Categorical 用自身类别顺序；否则按值排序稳定输出
        col = df_all[gcol]
        if pd.api.types.is_categorical_dtype(col):
            levels = list(col.cat.categories)
        else:
            levels = sorted(pd.Series(col).dropna().unique().tolist(), key=lambda x: str(x))

        for lv in levels:
            sub = df_all[col == lv]
            if len(sub) == 0:
                continue
            y = sub["y"].values.astype(int)
            n_sub = int(len(sub))
            ev    = int(np.nansum(y))

            for pcol, mkey in zip(p_cols, model_keys):
                p = sub[pcol].values.astype(float)
                thr = float(thr_map.get(mkey, 0.5))
                met = bin_metrics(y, p, thr=thr)

                rows.append({
                    "Dataset": "Training set",
                    "GroupVar": glabel,
                    "GroupLevel": str(lv),
                    "N": n_sub,
                    "Events": ev,
                    "Model": mkey.upper(),
                    "AUC": met["AUC"],
                    "Brier": met["Brier"],
                    "Accuracy": met["Accuracy"],
                    "Sensitivity": met["Sensitivity"],
                    "Specificity": met["Specificity"],
                    "Precision": met["Precision"],
                    "NPV": met["NPV"],
                    "F1": met["F1"],
                    "Threshold": thr
                })

    df_out = pd.DataFrame(rows)

    # 7) 美化 & 导出
    num_cols = ["AUC","Brier","Accuracy","Sensitivity","Specificity","Precision","NPV","F1","Threshold"]
    for c in num_cols:
        if c in df_out.columns:
            df_out[c] = df_out[c].map(lambda x: fmt(x, 4))

    df_out = df_out.sort_values(["GroupVar","GroupLevel","Model"]).reset_index(drop=True)
    df_out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print("[save]", OUT_PATH)
    print("[done] S13 subgroup table ready.")

if __name__ == "__main__":
    main()
