# -*- coding: utf-8 -*-
# 24_quick_shap_adaboost.py — SHAP 分析（仅 RFE 特征；Tree 优先 + Kernel 兜底）
# 要点：
# - 只使用 RFE 选出的特征（特征文件缺失→退出，不回退全特征）
# - 在“预处理后空间”做 SHAP，避免重复预处理错配
# - 优先 TreeExplainer；若分类器不受支持，则尝试拆出校准器底层树；仍失败→ KernelExplainer
# - 强校验：模型训练期望列 vs 当前 RFE 特征列一致；不一致→退出
# - KLOSA 缺列按类型补齐（数值=0，分类="__UNK__"）

import os; os.environ.setdefault("MPLBACKEND", "Agg")
import sys
from pathlib import Path
import time
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib

from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

# ============== 配置 ==============
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

PROJECT_ROOT = repo_root()
try:
    with open(PROJECT_ROOT / "07_config" / "config.yaml", "r", encoding="utf-8") as f:
        CFG = yaml.safe_load(f) or {}
    VER_IN = CFG.get("run_id_in", "v2025-10-01")
    VER_OUT = CFG.get("run_id_out", "v2025-10-03")
    Y_NAME = (CFG.get("outcome", {}) or {}).get("name", "depression_bin")
except Exception:
    VER_IN = "v2025-10-01"; VER_OUT = "v2025-10-03"; Y_NAME = "depression_bin"

# 数据路径
MEAN_MODE_XY = PROJECT_ROOT / "02_processed_data" / VER_IN / "frozen" / "charls_mean_mode_Xy.csv"
TRAIN_IDX    = PROJECT_ROOT / "02_processed_data" / VER_IN / "splits" / "charls_train_idx.csv"
VAL_IDX      = PROJECT_ROOT / "02_processed_data" / VER_IN / "splits" / "charls_val_idx.csv"
TEST_IDX     = PROJECT_ROOT / "02_processed_data" / VER_IN / "splits" / "charls_test_idx.csv"
# 更新KLOSA数据路径
KLOSA_MEAN_MODE_XY = PROJECT_ROOT / "02_processed_data" / VER_OUT / "frozen" / "klosa_transfer_Xy.csv"
# 备用路径，如果上面的路径不存在
KLOSA_BACKUP_PATH = PROJECT_ROOT / "02_processed_data" / VER_IN / "frozen" / "klosa_mean_mode_Xy.csv"

# 特征文件（必须存在）
S4_DIR = PROJECT_ROOT / "10_experiments" / VER_OUT / "feature_sweep"
ADABOOST_FEATURES_FILE = S4_DIR / "S4_RFE_adaboost_best_features.csv"

# 模型目录（必须能加载到）
MODELS_DIR = PROJECT_ROOT / "10_experiments" / VER_OUT / "final_eval_s7" / "models"
CAND_MODEL_FILES = [
    "ADABOOST_best_pipeline.joblib",
    "ADABOOST_calibrated_isotonic.joblib",
    "AdaBoost_best_pipeline.joblib",
    "AdaBoost_pipeline.joblib",
    "adaboost_model.joblib",
    "adaboost_pipeline.joblib",
    "RF_best_pipeline.joblib",
    "rf_pipeline.joblib",
]

# 输出目录 - 优化目录结构以区分不同数据集的结果
OUT_DIR = PROJECT_ROOT / "10_experiments" / VER_OUT / "figs" / "shap_adaboost_rfe_tree_kernel"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 创建子目录用于不同数据集的结果
TEST_OUT_DIR = OUT_DIR / "test_charls"  # 测试集结果目录
KLOSA_OUT_DIR = OUT_DIR / "klosa"       # KLOSA数据结果目录
COMP_OUT_DIR = OUT_DIR / "comparison"   # 比较结果目录
TEST_OUT_DIR.mkdir(parents=True, exist_ok=True)
KLOSA_OUT_DIR.mkdir(parents=True, exist_ok=True)
COMP_OUT_DIR.mkdir(parents=True, exist_ok=True)

# 运行参数
MAX_SAMPLES = 100000     # SHAP 计算的上限样本数（设置为很大的数，实际上使用全部样本）
KLOSA_MAX_SAMPLES = 10000   # KLOSA数据的SHAP计算上限样本数（保持限制以提高性能）
TOPK = 10                # 图上显示的Top-K特征
KMEANS_BACKGROUND = 30   # Kernel 背景聚类中心数
AUC_DECIMALS = 4

# ============== 工具函数 ==============
def to_dense(X):
    return X.toarray() if sparse.issparse(X) else np.asarray(X)

def guess_steps(pipe: Pipeline):
    """稳健识别 Pipeline 中的预处理器与分类器 step 名称。"""
    assert isinstance(pipe, Pipeline)
    pre_name = None
    clf_name = None
    for name, step in pipe.named_steps.items():
        if hasattr(step, "transform") and pre_name is None:
            pre_name = name
        if hasattr(step, "predict_proba") and clf_name is None:
            clf_name = name
    if pre_name is None or clf_name is None:
        raise RuntimeError("无法在 Pipeline 中识别预处理器或分类器 step 名称")
    return pre_name, clf_name

def get_feature_names(pre: ColumnTransformer, fallback_cols):
    try:
        names = pre.get_feature_names_out()
        return list(names), list(names)
    except Exception:
        return list(fallback_cols), list(fallback_cols)

def clean_display_names(model_space_names):
    return [n.split("__", 1)[1] if "__" in n else n for n in model_space_names]

def downsample(X, max_n):
    n = len(X)
    if n <= max_n: return X, np.arange(n)
    idx = np.random.choice(n, max_n, replace=False)
    if isinstance(X, pd.DataFrame): return X.iloc[idx], idx
    return X[idx], idx

def ensure_klosa_columns(df_klosa: pd.DataFrame, feats, num_cols, cat_cols):
    out = pd.DataFrame(index=df_klosa.index, columns=feats)
    for c in feats:
        out[c] = df_klosa[c] if c in df_klosa.columns else ("__UNK__" if c in cat_cols else 0)
    return out

def as_1d(arr):
    arr = np.asarray(arr)
    return arr.ravel() if arr.ndim > 1 else arr

def scalar_base_value(ev, class_index=1):
    ev = np.asarray(ev)
    if ev.ndim == 0: return float(ev)
    if ev.ndim == 1:
        if len(ev) == 1: return float(ev[0])
        idx = class_index if class_index < len(ev) else -1
        return float(ev[idx])
    return float(ev.reshape(-1)[0])

def expected_input_columns(pre: ColumnTransformer):
    cols = []
    for name, trans, cols_sel in pre.transformers_:
        if cols_sel in ("drop", "remainder"): continue
        if hasattr(cols_sel, "__iter__"): cols.extend(list(cols_sel))
    seen=set(); uniq=[]
    for c in cols:
        if c not in seen: uniq.append(c); seen.add(c)
    return uniq

def unwrap_base_tree_if_calibrated(clf):
    """
    若是 CalibratedClassifierCV，尽量取出底层树模型；否则返回 None。
    可能字段：estimator 或 base_estimator（不同 sklearn 版本）
    """
    if isinstance(clf, CalibratedClassifierCV):
        base = None
        try:
            base = clf.calibrated_classifiers_[0].estimator
        except Exception:
            try:
                base = clf.calibrated_classifiers_[0].base_estimator
            except Exception:
                base = None
        return base
    return None

# ============== 主流程 ==============
def main():
    print("="*70)
    print("AdaBoost/RF SHAP（RFE特征；Tree优先 + Kernel兜底）")
    print("="*70)
    t0 = time.time()

    # 0) 检查特征文件
    if not ADABOOST_FEATURES_FILE.exists():
        print(f"[ERROR] 未找到 AdaBoost 特征选择文件：{ADABOOST_FEATURES_FILE}")
        print("        按要求不使用全特征，已停止。")
        sys.exit(1)

    # 1) 读取数据
    print("[INFO] 读取 CHARLS 数据与索引 ...")
    try:
        df_all = pd.read_csv(MEAN_MODE_XY)
        tr = pd.read_csv(TRAIN_IDX).iloc[:, 0].astype(int).values
        te = pd.read_csv(TEST_IDX).iloc[:, 0].astype(int).values
        val = pd.read_csv(VAL_IDX).iloc[:, 0].astype(int).values if VAL_IDX.exists() else None
    except Exception as e:
        print(f"[ERROR] 数据读取失败: {e}")
        sys.exit(1)

    # 2) 读取 RFE 特征
    print(f"[INFO] 读取 RFE 特征文件：{ADABOOST_FEATURES_FILE.name}")
    feat_df = pd.read_csv(ADABOOST_FEATURES_FILE)
    feat_col = "feature" if "feature" in feat_df.columns else feat_df.columns[0]
    selected_features = feat_df[feat_col].astype(str).tolist()
    feats = [f for f in selected_features if f in df_all.columns and f != Y_NAME]
    if len(feats) == 0:
        print("[ERROR] RFE 特征列表在当前数据中均不存在，或只包含目标列。")
        sys.exit(1)
    print(f"[INFO] 使用 RFE 选出的特征：{len(feats)} 个")

    # 3) 拆分
    Xtr = df_all.loc[tr, feats].copy()
    ytr = df_all.loc[tr, Y_NAME].astype(int).values
    Xte = df_all.loc[te, feats].copy()
    yte = df_all.loc[te, Y_NAME].astype(int).values

    # 4) 推断特征类型（仅用于 KLOSA 缺列补）
    num_cols = [c for c in feats if pd.api.types.is_numeric_dtype(df_all[c].dtype) and df_all[c].nunique() > 5]
    cat_cols = [c for c in feats if c not in num_cols]
    print(f"[INFO] 数值特征: {len(num_cols)}; 分类特征: {len(cat_cols)}")

    # 5) 加载已保存模型（不回退训练）
    pipe = None
    if MODELS_DIR.exists():
        print(f"[INFO] 尝试从 {MODELS_DIR} 加载既有模型 ...")
        for fn in CAND_MODEL_FILES:
            path = MODELS_DIR / fn
            if path.exists():
                try:
                    tmp = joblib.load(path)
                    if isinstance(tmp, Pipeline):
                        pre_name, clf_name = guess_steps(tmp)
                        _ = tmp.named_steps[pre_name], tmp.named_steps[clf_name]
                        pipe = tmp
                        print(f"[SUCCESS] 已加载模型: {fn}")
                        break
                except Exception as e:
                    print(f"[WARN] 加载失败 {fn}: {e}")
    if pipe is None:
        print("[ERROR] 未找到可用的第14步模型，已停止。请先完成模型训练并保存到 models 目录。")
        sys.exit(1)

    # 5.1) 期望列一致性强校验
    pre_name, clf_name = guess_steps(pipe)
    exp_cols = expected_input_columns(pipe.named_steps[pre_name])
    miss = [c for c in exp_cols if c not in Xte.columns]
    extra = [c for c in Xte.columns if c not in exp_cols]
    if miss or extra:
        print("[ERROR] 模型训练期望的输入列与当前 RFE 特征不一致。")
        if miss:  print("  缺失列(示例):", miss[:30], "..." if len(miss) > 30 else "")
        if extra: print("  多余列(示例):", extra[:30], "..." if len(extra) > 30 else "")
        sys.exit(1)

    # 6) 简单评估（用完整管道的概率 —— 可能是“已校准”的）
    try:
        Xte_t_full = pipe.named_steps[pre_name].transform(Xte)
        p_full = pipe.named_steps[clf_name].predict_proba(Xte_t_full)[:, 1]
        auc = roc_auc_score(yte, p_full)
        print(f"[RESULT] 测试集 AUC（管道概率） = {auc:.{AUC_DECIMALS}f}")
    except Exception as e:
        print(f"[ERROR] 简单评估失败：{e}")
        sys.exit(1)

    # ========== SHAP 准备（Tree 优先 + Kernel 兜底） ==========
    model_pre = pipe.named_steps[pre_name]
    model_clf = pipe.named_steps[clf_name]

    # 特征名（模型空间）与展示名
    model_space_names, _ = get_feature_names(model_pre, feats)
    display_names = clean_display_names(model_space_names)

    # 预处理测试集（模型空间）
    Xte_t = model_pre.transform(Xte)
    Xte_dense = to_dense(Xte_t)
    Xte_df_model = pd.DataFrame(Xte_dense, columns=model_space_names)

    # 不再下采样，使用全部测试集样本
    # 由于MAX_SAMPLES已经设置得很大，这里实际上会使用全部样本
    Xte_for_shap, idx_te = downsample(Xte_df_model, MAX_SAMPLES)
    print(f"[INFO] SHAP 使用测试集样本数：{len(Xte_for_shap)} / {len(Xte_df_model)}")
    print(f"[INFO] 注意: CHARLS测试集使用全部样本")

    # KLOSA（可选）- 增强调试信息以确保KLOSA数据处理正确
    Xk_for_shap = None
    print(f"[DEBUG] 检查KLOSA数据文件路径: {KLOSA_MEAN_MODE_XY}")
    print(f"[DEBUG] KLOSA数据文件是否存在: {KLOSA_MEAN_MODE_XY.exists()}")
    
    # 如果主路径不存在，尝试备用路径
    klosa_path = KLOSA_MEAN_MODE_XY
    if not klosa_path.exists() and KLOSA_BACKUP_PATH.exists():
        print(f"[INFO] 主路径不存在，使用备用路径: {KLOSA_BACKUP_PATH}")
        klosa_path = KLOSA_BACKUP_PATH
    
    try:
        if klosa_path.exists():
            print(f"[INFO] 读取 KLOSA 数据：{klosa_path.name}")
            df_k = pd.read_csv(klosa_path)
            print(f"[DEBUG] KLOSA数据列名: {df_k.columns.tolist()[:10]}... 共{len(df_k.columns)}列")
            print(f"[DEBUG] KLOSA样本数: {len(df_k)}")
            
            if Y_NAME in df_k.columns:
                print(f"[DEBUG] 目标变量{Y_NAME}分布: {df_k[Y_NAME].value_counts().to_dict()}")
                df_k_feats = ensure_klosa_columns(df_k, feats, num_cols, cat_cols)
                print(f"[DEBUG] 确保列完整后的列数: {len(df_k_feats.columns)}")
                
                Xk_t = model_pre.transform(df_k_feats)
                Xk_dense = to_dense(Xk_t)
                Xk_df_model = pd.DataFrame(Xk_dense, columns=model_space_names)
                print(f"[DEBUG] 变换后的KLOSA数据形状: {Xk_df_model.shape}")
                
                Xk_for_shap, klosa_idx = downsample(Xk_df_model, KLOSA_MAX_SAMPLES)  # 使用KLOSA专用的样本数限制
                print(f"[INFO] SHAP 使用 KLOSA 样本数：{len(Xk_for_shap)} / {len(df_k)}")
                print(f"[INFO] 注意: KLOSA数据使用上限{KLOSA_MAX_SAMPLES}个样本，而CHARLS使用全部样本")
            else:
                print(f"[WARN] KLOSA 缺少目标列 {Y_NAME}，跳过其 SHAP。")
                print(f"[DEBUG] 可用列: {df_k.columns.tolist()[:10]}...")
        else:
            print(f"[WARN] KLOSA数据文件不存在: {klosa_path}")
    except Exception as e:
        import traceback
        print(f"[WARN] KLOSA 处理失败，跳过其 SHAP：{e}")
        traceback.print_exc()
        Xk_for_shap = None

    # 预处理后空间的预测函数（供 Kernel 使用）
    def model_predict_on_preprocessed(X_pre_df_or_nd):
        X_arr = X_pre_df_or_nd.values if isinstance(X_pre_df_or_nd, pd.DataFrame) else np.asarray(X_pre_df_or_nd)
        return model_clf.predict_proba(X_arr)[:, 1]

    # ========== 计算 SHAP ==========
    Sv_test = None
    Sv_klosa = None
    expected_value = None
    title_suffix = ""  # 若解释的是未校准底层树模型，会补充说明

    t_shap0 = time.time()
    print("[INFO] 优先使用 TreeExplainer ...")
    used_explainer = "tree"

    # 先对分类器本体尝试 TreeExplainer
    try:
        tree_explainer = shap.TreeExplainer(model_clf, feature_perturbation="interventional", model_output="probability")
        sv = tree_explainer.shap_values(Xte_for_shap.values if isinstance(Xte_for_shap, pd.DataFrame) else Xte_for_shap)
        Sv_test = np.asarray(sv[1]) if isinstance(sv, list) else np.asarray(sv)
        expected_value = tree_explainer.expected_value
    except Exception as e_tree_direct:
        # 若本体是 CalibratedClassifierCV，试抽底层树模型
        base = unwrap_base_tree_if_calibrated(model_clf)
        if base is not None:
            try:
                tree_explainer = shap.TreeExplainer(base, feature_perturbation="interventional", model_output="probability")
                sv = tree_explainer.shap_values(Xte_for_shap.values if isinstance(Xte_for_shap, pd.DataFrame) else Xte_for_shap)
                Sv_test = np.asarray(sv[1]) if isinstance(sv, list) else np.asarray(sv)
                expected_value = tree_explainer.expected_value
                title_suffix = " (explaining uncalibrated base model)"
            except Exception as e_tree_base:
                used_explainer = "kernel"
        else:
            used_explainer = "kernel"

    # 如果 Tree 两次都失败 → Kernel
    if used_explainer == "kernel":
        print("[WARN] TreeExplainer 不可用，改用 KernelExplainer（会较慢）")
        bg_k = min(KMEANS_BACKGROUND, len(Xte_for_shap))
        background = shap.kmeans(Xte_for_shap, bg_k)
        kernel_explainer = shap.KernelExplainer(model_predict_on_preprocessed, background)
        Sv_test = np.asarray(kernel_explainer.shap_values(Xte_for_shap))
        expected_value = kernel_explainer.expected_value

    # KLOSA 的 SHAP
    if Xk_for_shap is not None:
        if used_explainer == "kernel":
            Sv_klosa = np.asarray(kernel_explainer.shap_values(Xk_for_shap))
        else:
            svk = tree_explainer.shap_values(Xk_for_shap.values if isinstance(Xk_for_shap, pd.DataFrame) else Xk_for_shap)
            Sv_klosa = np.asarray(svk[1]) if isinstance(svk, list) else np.asarray(svk)

    print(f"[INFO] SHAP 计算耗时 {time.time() - t_shap0:.2f}s，方式：{used_explainer}{title_suffix}")
    print(f"[DEBUG] 测试集 Sv_test 形状: {Sv_test.shape}")
    if Sv_klosa is not None:
        print(f"[DEBUG] KLOSA   Sv_klosa 形状: {Sv_klosa.shape}")

    # ========== 导出 & 图表 ==========
    # 重要度（测试集）
    mean_abs_shap_test = np.mean(np.abs(Sv_test), axis=0)
    imp_test_df = pd.DataFrame({
        "feature_model_space": model_space_names[:len(mean_abs_shap_test)],
        "feature": clean_display_names(model_space_names[:len(mean_abs_shap_test)]),
        "importance": mean_abs_shap_test
    }).sort_values("importance", ascending=False)
    imp_test_df.to_csv(OUT_DIR / "test_feature_importance.csv", index=False)
    pd.DataFrame(Sv_test, columns=model_space_names[:Sv_test.shape[1]]).to_csv(OUT_DIR / "test_shap_values.csv", index=False)
    
    # 提取AdaBoost模型的原生特征重要性（如果可用）
    print("[INFO] 提取AdaBoost原生特征重要性...")
    adaboost_model = None
    try:
        # 尝试提取AdaBoost模型
        if isinstance(model_clf, AdaBoostClassifier):
            adaboost_model = model_clf
        elif isinstance(model_clf, CalibratedClassifierCV):
            base = unwrap_base_tree_if_calibrated(model_clf)
            if isinstance(base, AdaBoostClassifier):
                adaboost_model = base
        
        if adaboost_model is not None:
            # 提取特征重要性
            if hasattr(adaboost_model, 'feature_importances_'):
                ada_importances = adaboost_model.feature_importances_
                ada_imp_df = pd.DataFrame({
                    "feature_model_space": model_space_names[:len(ada_importances)],
                    "feature": clean_display_names(model_space_names[:len(ada_importances)]),
                    "importance": ada_importances
                }).sort_values("importance", ascending=False)
                
                # 保存到CSV
                ada_imp_df.to_csv(OUT_DIR / "adaboost_native_importance.csv", index=False)
                print(f"[INFO] 已保存AdaBoost原生特征重要性到 {OUT_DIR / 'adaboost_native_importance.csv'}")
                
                # 生成AdaBoost原生特征重要性排序图
                plt.figure(figsize=(10, 8), dpi=240)
                top_n = min(TOPK, len(ada_imp_df))
                ada_top = ada_imp_df.head(top_n)
                plt.barh(np.arange(top_n), ada_top['importance'], align='center')
                plt.yticks(np.arange(top_n), ada_top['feature'])
                plt.gca().invert_yaxis()  # 使最重要的特征在顶部
                plt.title(f"AdaBoost Native Feature Importance (top {top_n})")
                plt.xlabel('Importance')
                plt.tight_layout()
                plt.savefig(OUT_DIR / "adaboost_native_importance.png", dpi=300)
                plt.savefig(TEST_OUT_DIR / "adaboost_native_importance.png", dpi=300)
                plt.close()
                
                # 如果同时有SHAP和AdaBoost原生特征重要性，生成比较图
                plt.figure(figsize=(12, 10), dpi=240)
                # 合并两种重要性分数
                top_feats_both = list(imp_test_df['feature'].head(15).tolist()) + list(ada_imp_df['feature'].head(15).tolist())
                top_feats_both = list(dict.fromkeys(top_feats_both))[:15]  # 去重并限制到前15个
                
                comp_both = pd.DataFrame({
                    'feature': top_feats_both,
                    'SHAP': [imp_test_df.set_index('feature').reindex([f]).fillna(0)['importance'].values[0] for f in top_feats_both],
                    'AdaBoost': [ada_imp_df.set_index('feature').reindex([f]).fillna(0)['importance'].values[0] for f in top_feats_both]
                })
                
                # 标准化两种重要性分数，使其可比较
                comp_both['SHAP'] = comp_both['SHAP'] / comp_both['SHAP'].max()
                comp_both['AdaBoost'] = comp_both['AdaBoost'] / comp_both['AdaBoost'].max()
                
                # 按SHAP重要性排序
                comp_both = comp_both.sort_values('SHAP', ascending=False)
                
                # 绘制比较图
                x = np.arange(len(comp_both))
                width = 0.35
                fig, ax = plt.subplots(figsize=(12, 8), dpi=240)
                ax.bar(x - width/2, comp_both['SHAP'], width, label='SHAP')
                ax.bar(x + width/2, comp_both['AdaBoost'], width, label='AdaBoost Native')
                ax.set_xticks(x)
                ax.set_xticklabels(comp_both['feature'], rotation=45, ha='right')
                ax.legend()
                plt.title('Feature Importance: SHAP vs AdaBoost Native (normalized)')
                plt.tight_layout()
                plt.savefig(COMP_OUT_DIR / "importance_shap_vs_adaboost.png", dpi=300)
                plt.savefig(OUT_DIR / "importance_shap_vs_adaboost.png", dpi=300)  # 为了兼容性也保存到主目录
                plt.close()
            else:
                print("[WARN] AdaBoost模型没有feature_importances_属性")
        else:
            print("[WARN] 无法提取AdaBoost模型")
    except Exception as e:
        print(f"[WARN] 提取AdaBoost原生特征重要性失败: {e}")
        import traceback
        traceback.print_exc()

    # KLOSA（可选）
    if Sv_klosa is not None:
        mean_abs_shap_klosa = np.mean(np.abs(Sv_klosa), axis=0)
        imp_klosa_df = pd.DataFrame({
            "feature_model_space": model_space_names[:len(mean_abs_shap_klosa)],
            "feature": clean_display_names(model_space_names[:len(mean_abs_shap_klosa)]),
            "importance": mean_abs_shap_klosa
        }).sort_values("importance", ascending=False)
        imp_klosa_df.to_csv(OUT_DIR / "klosa_feature_importance.csv", index=False)
        imp_klosa_df.to_csv(KLOSA_OUT_DIR / "klosa_feature_importance.csv", index=False)
        pd.DataFrame(Sv_klosa, columns=model_space_names[:Sv_klosa.shape[1]]).to_csv(OUT_DIR / "klosa_shap_values.csv", index=False)
        pd.DataFrame(Sv_klosa, columns=model_space_names[:Sv_klosa.shape[1]]).to_csv(KLOSA_OUT_DIR / "klosa_shap_values.csv", index=False)
        
        # 为KLOSA数据生成图表
        print("[INFO] 生成KLOSA数据的SHAP图表...")
        Xk_for_plot = Xk_for_shap if isinstance(Xk_for_shap, pd.DataFrame) else pd.DataFrame(Xk_for_shap, columns=model_space_names)
        
        # KLOSA Bar图
        plt.figure(figsize=(10, 8), dpi=240)
        shap.summary_plot(Sv_klosa, features=Xk_for_plot, feature_names=clean_display_names(model_space_names),
                          plot_type="bar", show=False, max_display=TOPK)
        plt.title(f"SHAP — mean(|SHAP|) (KLOSA, top {TOPK}){title_suffix}")
        plt.tight_layout(); plt.savefig(KLOSA_OUT_DIR / "shap_bar.png", dpi=300); plt.close()
        
        # KLOSA Beeswarm图
        plt.figure(figsize=(10, 8), dpi=240)
        shap.summary_plot(Sv_klosa, features=Xk_for_plot, feature_names=clean_display_names(model_space_names),
                          show=False, max_display=TOPK)
        plt.title(f"SHAP — beeswarm (KLOSA, top {TOPK}){title_suffix}")
        plt.tight_layout(); plt.savefig(KLOSA_OUT_DIR / "shap_beeswarm.png", dpi=300); plt.close()
        
        # KLOSA数据的单样本图表
        try:
            # 选择概率最高和最低的样本
            if used_explainer == "kernel":
                p_klosa = model_predict_on_preprocessed(Xk_for_plot)
            else:
                if isinstance(model_clf, CalibratedClassifierCV) and title_suffix:
                    base = unwrap_base_tree_if_calibrated(model_clf)
                    p_klosa = base.predict_proba(Xk_for_plot.values)[:, 1]
                else:
                    p_klosa = model_clf.predict_proba(Xk_for_plot.values)[:, 1]
            
            idx_k_top = int(np.argmax(p_klosa)); idx_k_bot = int(np.argmin(p_klosa))
            x_k_top = Xk_for_plot.iloc[idx_k_top:idx_k_top+1]; x_k_bot = Xk_for_plot.iloc[idx_k_bot:idx_k_bot+1]
            prob_k_top = float(p_klosa[idx_k_top]); prob_k_bot = float(p_klosa[idx_k_bot])
            
            # 计算SHAP值
            if used_explainer == "kernel":
                sv_k_top = kernel_explainer.shap_values(x_k_top)
                sv_k_bot = kernel_explainer.shap_values(x_k_bot)
                ev_k = kernel_explainer.expected_value
            else:
                sv_k_top = tree_explainer.shap_values(x_k_top.values)
                sv_k_bot = tree_explainer.shap_values(x_k_bot.values)
                ev_k = tree_explainer.expected_value
            
            if isinstance(sv_k_top, list): sv_k_top = sv_k_top[1]
            if isinstance(sv_k_bot, list): sv_k_bot = sv_k_bot[1]
            base_val_k = scalar_base_value(ev_k, class_index=1)
            
            # KLOSA最高概率样本的瀑布图
            plt.figure(figsize=(10, 12), dpi=240)
            shap.plots.waterfall(
                shap.Explanation(values=as_1d(sv_k_top), base_values=base_val_k,
                               data=x_k_top.values[0], feature_names=clean_display_names(model_space_names)),
                max_display=20, show=False
            )
            plt.title(f"SHAP — waterfall (KLOSA highest prob={prob_k_top:.4f}){title_suffix}")
            plt.tight_layout(); plt.savefig(KLOSA_OUT_DIR / "shap_waterfall_high.png", dpi=300); plt.close()
            
            # KLOSA最低概率样本的瀑布图
            plt.figure(figsize=(10, 12), dpi=240)
            shap.plots.waterfall(
                shap.Explanation(values=as_1d(sv_k_bot), base_values=base_val_k,
                               data=x_k_bot.values[0], feature_names=clean_display_names(model_space_names)),
                max_display=20, show=False
            )
            plt.title(f"SHAP — waterfall (KLOSA lowest prob={prob_k_bot:.4f}){title_suffix}")
            plt.tight_layout(); plt.savefig(KLOSA_OUT_DIR / "shap_waterfall_low.png", dpi=300); plt.close()
            
            # KLOSA最高概率样本的力量图
            plt.figure(figsize=(20, 3), dpi=240)
            shap.plots.force(base_value=base_val_k, shap_values=as_1d(sv_k_top),
                           features=x_k_top.iloc[0], feature_names=clean_display_names(model_space_names),
                           matplotlib=True, show=False)
            plt.title(f"SHAP — force (KLOSA highest prob={prob_k_top:.4f}){title_suffix}")
            plt.tight_layout(); plt.savefig(KLOSA_OUT_DIR / "shap_force_high.png", dpi=300, bbox_inches='tight'); plt.close()
            
            # KLOSA最低概率样本的力量图
            plt.figure(figsize=(20, 3), dpi=240)
            shap.plots.force(base_value=base_val_k, shap_values=as_1d(sv_k_bot),
                           features=x_k_bot.iloc[0], feature_names=clean_display_names(model_space_names),
                           matplotlib=True, show=False)
            plt.title(f"SHAP — force (KLOSA lowest prob={prob_k_bot:.4f}){title_suffix}")
            plt.tight_layout(); plt.savefig(KLOSA_OUT_DIR / "shap_force_low.png", dpi=300, bbox_inches='tight'); plt.close()
        except Exception as e:
            print(f"[WARN] KLOSA数据的单样本图生成失败: {e}")
        
        # KLOSA依赖图
        try:
            top_idx_klosa = np.argsort(mean_abs_shap_klosa)[::-1][:9]
            fig, axes = plt.subplots(3, 3, figsize=(15, 15), dpi=240)
            for i, idx in enumerate(top_idx_klosa):
                ax = axes[i // 3, i % 3]
                shap.dependence_plot(ind=idx, shap_values=Sv_klosa, features=Xk_for_plot,
                                   feature_names=clean_display_names(model_space_names), show=False, ax=ax)
            fig.suptitle(f"SHAP — dependence (KLOSA top 9 features){title_suffix}")
            fig.tight_layout(); fig.savefig(KLOSA_OUT_DIR / "shap_dependence.png", dpi=300); plt.close(fig)
            
            # 单独的依赖图
            klosa_dep_dir = KLOSA_OUT_DIR / "dependence_plots"; klosa_dep_dir.mkdir(exist_ok=True)
            for rank, idx in enumerate(top_idx_klosa, start=1):
                plt.figure(figsize=(8, 6), dpi=240)
                shap.dependence_plot(ind=idx, shap_values=Sv_klosa, features=Xk_for_plot,
                                   feature_names=clean_display_names(model_space_names), show=False)
                plt.title(f"SHAP dependence (KLOSA): {clean_display_names(model_space_names)[idx]} (#{rank}){title_suffix}")
                plt.tight_layout(); plt.savefig(klosa_dep_dir / f"{rank:02d}_{clean_display_names(model_space_names)[idx]}_dependence.png", dpi=300); plt.close()
        except Exception as e:
            print(f"[WARN] KLOSA数据的依赖图生成失败: {e}")
            import traceback
            traceback.print_exc()

        # 对比图（matplotlib）
        top_feats = list(imp_test_df["feature"].head(20)) + list(imp_klosa_df["feature"].head(20))
        top_feats = list(dict.fromkeys(top_feats))
        comp = pd.DataFrame({
            "feature": top_feats,
            "CHARLS": [imp_test_df.set_index("feature").reindex([f]).fillna(0)["importance"].values[0] for f in top_feats],
            "KLOSA":  [imp_klosa_df.set_index("feature").reindex([f]).fillna(0)["importance"].values[0] for f in top_feats],
        })
        comp_top = comp.sort_values("CHARLS", ascending=False).head(15)
        plt.figure(figsize=(12, 8), dpi=240)
        x = np.arange(len(comp_top)); w = 0.38
        plt.bar(x - w/2, comp_top["CHARLS"].values, width=w, label="CHARLS")
        plt.bar(x + w/2, comp_top["KLOSA"].values, width=w, label="KLOSA")
        plt.xticks(x, comp_top["feature"].values, rotation=45, ha="right")
        plt.title("Feature Importance Comparison (mean |SHAP|)")
        plt.legend(); plt.tight_layout()
        plt.savefig(COMP_OUT_DIR / "feature_importance_comparison.png", dpi=300)
        plt.savefig(OUT_DIR / "feature_importance_comparison.png", dpi=300)  # 为了兼容性也保存到主目录
        plt.close()

    # summary bar / beeswarm（测试集）
    Xte_for_plot = Xte_for_shap if isinstance(Xte_for_shap, pd.DataFrame) else pd.DataFrame(Xte_for_shap, columns=model_space_names)

    plt.figure(figsize=(10, 8), dpi=240)
    shap.summary_plot(Sv_test, features=Xte_for_plot, feature_names=clean_display_names(model_space_names),
                      plot_type="bar", show=False, max_display=TOPK)
    plt.title(f"SHAP — mean(|SHAP|) (CHARLS Test, top {TOPK}){title_suffix}")
    plt.tight_layout(); plt.savefig(TEST_OUT_DIR / "shap_bar.png", dpi=300)
    plt.savefig(OUT_DIR / "shap_bar.png", dpi=300); plt.close()

    plt.figure(figsize=(10, 8), dpi=240)
    shap.summary_plot(Sv_test, features=Xte_for_plot, feature_names=clean_display_names(model_space_names),
                      show=False, max_display=TOPK)
    plt.title(f"SHAP — beeswarm (CHARLS Test, top {TOPK}){title_suffix}")
    plt.tight_layout(); plt.savefig(TEST_OUT_DIR / "shap_beeswarm.png", dpi=300)
    plt.savefig(OUT_DIR / "shap_beeswarm.png", dpi=300); plt.close()

    # 单样本 waterfall/force（选择样本时，用“用于解释的概率”更一致）
    print("[INFO] 生成单样本 waterfall/force 图 ...")
    try:
        if used_explainer == "kernel":
            # 用完整管道概率来选样（Kernel 解释的也是管道概率）
            p_for_pick = model_predict_on_preprocessed(Xte_for_plot)
        else:
            # 若解释的是底层树，则用树概率选样
            if isinstance(model_clf, CalibratedClassifierCV) and title_suffix:
                # 解释底层树
                base = unwrap_base_tree_if_calibrated(model_clf)
                p_for_pick = base.predict_proba(Xte_for_plot.values)[:, 1]
            else:
                p_for_pick = model_clf.predict_proba(Xte_for_plot.values)[:, 1]

        idx_top = int(np.argmax(p_for_pick)); idx_bot = int(np.argmin(p_for_pick))
        x_top = Xte_for_plot.iloc[idx_top:idx_top+1]; x_bot = Xte_for_plot.iloc[idx_bot:idx_bot+1]
        prob_top = float(p_for_pick[idx_top]); prob_bot = float(p_for_pick[idx_bot])

        if used_explainer == "kernel":
            sv_top = kernel_explainer.shap_values(x_top)
            sv_bot = kernel_explainer.shap_values(x_bot)
            ev = kernel_explainer.expected_value
        else:
            sv_top = tree_explainer.shap_values(x_top.values)
            sv_bot = tree_explainer.shap_values(x_bot.values)
            ev = tree_explainer.expected_value

        if isinstance(sv_top, list): sv_top = sv_top[1]
        if isinstance(sv_bot, list): sv_bot = sv_bot[1]
        base_val = scalar_base_value(ev, class_index=1)

        plt.figure(figsize=(10, 12), dpi=240)
        shap.plots.waterfall(
            shap.Explanation(values=as_1d(sv_top), base_values=base_val,
                             data=x_top.values[0], feature_names=clean_display_names(model_space_names)),
            max_display=20, show=False
        )
        plt.title(f"SHAP — waterfall (highest prob={prob_top:.4f}){title_suffix}")
        plt.tight_layout(); plt.savefig(OUT_DIR / "shap_waterfall_high.png", dpi=300); plt.close()

        plt.figure(figsize=(10, 12), dpi=240)
        shap.plots.waterfall(
            shap.Explanation(values=as_1d(sv_bot), base_values=base_val,
                             data=x_bot.values[0], feature_names=clean_display_names(model_space_names)),
            max_display=20, show=False
        )
        plt.title(f"SHAP — waterfall (lowest prob={prob_bot:.4f}){title_suffix}")
        plt.tight_layout(); plt.savefig(OUT_DIR / "shap_waterfall_low.png", dpi=300); plt.close()

        plt.figure(figsize=(20, 3), dpi=240)
        shap.plots.force(base_value=base_val, shap_values=as_1d(sv_top),
                         features=x_top.iloc[0], feature_names=clean_display_names(model_space_names),
                         matplotlib=True, show=False)
        plt.title(f"SHAP — force (highest prob={prob_top:.4f}){title_suffix}")
        plt.tight_layout(); plt.savefig(OUT_DIR / "shap_force_high.png", dpi=300, bbox_inches='tight'); plt.close()

        plt.figure(figsize=(20, 3), dpi=240)
        shap.plots.force(base_value=base_val, shap_values=as_1d(sv_bot),
                         features=x_bot.iloc[0], feature_names=clean_display_names(model_space_names),
                         matplotlib=True, show=False)
        plt.title(f"SHAP — force (lowest prob={prob_bot:.4f}){title_suffix}")
        plt.tight_layout(); plt.savefig(OUT_DIR / "shap_force_low.png", dpi=300, bbox_inches='tight'); plt.close()
    except Exception as e:
        print(f"[WARN] 单样本图生成失败：{e}")

    # 依赖图（Top-9）
    print("[INFO] 生成 Top-9 依赖图 ...")
    try:
        top_idx = np.argsort(mean_abs_shap_test)[::-1][:9]
        fig, axes = plt.subplots(3, 3, figsize=(15, 15), dpi=240)
        for i, idx in enumerate(top_idx):
            ax = axes[i // 3, i % 3]
            shap.dependence_plot(ind=idx, shap_values=Sv_test, features=Xte_for_plot,
                                 feature_names=clean_display_names(model_space_names), show=False, ax=ax)
        fig.suptitle(f"SHAP — dependence (top 9 features){title_suffix}")
        fig.tight_layout(); fig.savefig(OUT_DIR / "shap_dependence.png", dpi=300); plt.close(fig)

        # 生成单个特征依赖图
        dep_dir = OUT_DIR / "dependence_plots"; dep_dir.mkdir(exist_ok=True)
        single_dep_dir = OUT_DIR / "single_feature_dependence"; single_dep_dir.mkdir(exist_ok=True)
        
        # 为前9个最重要特征生成单独的依赖图
        for rank, idx in enumerate(top_idx, start=1):
            # 普通依赖图
            plt.figure(figsize=(8, 6), dpi=240)
            shap.dependence_plot(ind=idx, shap_values=Sv_test, features=Xte_for_plot,
                                 feature_names=clean_display_names(model_space_names), show=False)
            plt.title(f"SHAP dependence: {clean_display_names(model_space_names)[idx]} (#{rank}){title_suffix}")
            plt.tight_layout(); plt.savefig(dep_dir / f"{rank:02d}_{clean_display_names(model_space_names)[idx]}_dependence.png", dpi=300); plt.close()
            
            # 单个特征依赖图 - 每个特征与前20个重要特征的交互依赖图
            feature_name = clean_display_names(model_space_names)[idx]
            print(f"[INFO] 生成特征 '{feature_name}' 的交互依赖图...")
            
            # 为当前特征创建一个子目录
            feature_dir = single_dep_dir / f"{rank:02d}_{feature_name}"
            feature_dir.mkdir(exist_ok=True)
            
            # 获取前20个重要特征的索引（不包括当前特征）
            top20_idx = np.argsort(mean_abs_shap_test)[::-1][:20]
            interact_idx = [i for i in top20_idx if i != idx][:10]  # 最多取前10个不同的特征
            
            # 生成当前特征与其他重要特征的交互依赖图
            for j, interact_feature_idx in enumerate(interact_idx):
                interact_feature_name = clean_display_names(model_space_names)[interact_feature_idx]
                plt.figure(figsize=(10, 8), dpi=240)
                shap.dependence_plot(
                    ind=idx,  # 主特征索引
                    shap_values=Sv_test, 
                    features=Xte_for_plot,
                    feature_names=clean_display_names(model_space_names),
                    interaction_index=interact_feature_idx,  # 交互特征索引
                    show=False
                )
                plt.title(f"SHAP interaction: {feature_name} vs {interact_feature_name}{title_suffix}")
                plt.tight_layout()
                plt.savefig(feature_dir / f"interaction_{j+1:02d}_{interact_feature_name}.png", dpi=300)
                plt.close()
    except Exception as e:
        print(f"[WARN] 依赖图生成失败：{e}")

    # shap_importance.csv
    shap_importance = pd.DataFrame({
        "feature_model_space": model_space_names[:len(mean_abs_shap_test)],
        "feature": clean_display_names(model_space_names[:len(mean_abs_shap_test)]),
        "shap_importance": mean_abs_shap_test
    }).sort_values("shap_importance", ascending=False)
    shap_importance.to_csv(OUT_DIR / "shap_importance.csv", index=False)

    print(f"[INFO] 所有输出已保存到：{OUT_DIR}")
    print(f"[INFO] 总耗时：{time.time() - t0:.2f}s")
    print("="*70)
    print("[完成] AdaBoost/RF SHAP 分析（Tree 优先 + Kernel 兜底，RFE 特征）")
    print("="*70)

if __name__ == "__main__":
    main()
