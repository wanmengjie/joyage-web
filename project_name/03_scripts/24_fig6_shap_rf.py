# -*- coding: utf-8 -*-
# 24_fig6_shap_rf.py — SHAP for RF: bar / beeswarm / waterfall / dependence (Fig.6 style)
# - 读取 config.yaml 自动定位路径与标签名
# - 特征来源优先 S5，再回退 S4，最后全特征
# - 兼容 sklearn/SHAP 新旧接口；稀疏矩阵转密；特征名对齐

import os; os.environ["MPLBACKEND"] = "Agg"

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

import shap
from scipy import sparse

# ============== 配置 ==============
def repo_root() -> Path:
    # 本脚本位于 03_scripts/；父级为项目根
    return Path(__file__).resolve().parents[1]

with open(repo_root() / "07_config" / "config.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f) or {}
_paths = CFG.get("paths") or {}
project_root_cfg = _paths.get("project_root") or str(repo_root())
PROJECT_ROOT = Path(project_root_cfg)
VER_IN  = CFG.get("run_id_in",  CFG.get("run_id", "v2025-10-01"))
VER_OUT = CFG.get("run_id_out", CFG.get("run_id", "v2025-10-03"))
Y_NAME  = (CFG.get("outcome", {}) or {}).get("name", "depression_bin")

# 使用冻结版全量(Xy) + 索引划分
MEAN_MODE_XY = PROJECT_ROOT / "02_processed_data" / VER_IN  / "frozen" / "charls_mean_mode_Xy.csv"
TRAIN_IDX    = PROJECT_ROOT / "02_processed_data" / VER_IN  / "splits" / "charls_train_idx.csv"
TEST_IDX     = PROJECT_ROOT / "02_processed_data" / VER_IN  / "splits" / "charls_test_idx.csv"

# S5/S4 产物（先 S5，再回退 S4）
S5_RF_IMP = PROJECT_ROOT / "10_experiments" / VER_OUT / "feature_importance" / "rf_importance.csv"
S4_RF_BEST= PROJECT_ROOT / "10_experiments" / VER_OUT / "feature_sweep" / "S4_RFE_rf_best_features.csv"

OUT_DIR = PROJECT_ROOT / "10_experiments" / VER_OUT / "figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 第14步保存的模型目录（若存在优先加载RF管线）
S7_MODELS_DIR = PROJECT_ROOT / "10_experiments" / VER_OUT / "final_eval_s7" / "models"

TOPK = 5
USE_ALL_FEATURES = True  # 强制使用全量原始特征
RANDOM_STATE = 42
MAX_BEESWARM_N = 8000   # beeswarm 太大时下采样，防止渲染过慢/内存大

# ============== 工具函数 ==============
def _load_idx(p: Path) -> np.ndarray:
    if not p.exists():
        raise FileNotFoundError(f"缺少索引文件：{p}")
    return pd.read_csv(p).iloc[:, 0].astype(int).values

def infer_types(df: pd.DataFrame, y_name: str | None = None):
    feats = [c for c in df.columns if c != y_name] if y_name else list(df.columns)
    num, cat = [], []
    for c in feats:
        s = df[c]
        if np.issubdtype(s.dtype, np.number) and pd.Series(s).dropna().nunique() > 5:
            num.append(c)
        else:
            cat.append(c)
    return num, cat


def pick_rf_features(df: pd.DataFrame) -> list[str]:
    # 1) S5 importance（带 importance 排序，优先）
    if S5_RF_IMP.exists():
        df_imp = pd.read_csv(S5_RF_IMP)
        feat_col = "feature" if "feature" in df_imp.columns else df_imp.columns[0]
        if "importance" in df_imp.columns:
            feats = (df_imp[[feat_col, "importance"]]
                     .sort_values("importance", ascending=False)[feat_col]
                     .astype(str).tolist())
        else:
            feats = df_imp[feat_col].astype(str).tolist()
        feats = [f for f in feats if f in df.columns]
        if feats:
            return feats
    # 2) S4 RFE best
    if S4_RF_BEST.exists():
        s4 = pd.read_csv(S4_RF_BEST)
        col = "feature" if "feature" in s4.columns else s4.columns[0]
        feats = s4[col].astype(str).tolist()
        feats = [f for f in feats if f in df.columns]
        if feats:
            return feats
    # 3) 全特征兜底
    return [c for c in df.columns if c != Y_NAME]

def to_dense(X):
    return X.toarray() if sparse.issparse(X) else np.asarray(X)

def robust_tree_shap(clf, Xt):
    """
    兼容 SHAP 的旧/新接口：
    - 新：explainer(X) -> Explanation（优先）
    - 旧：explainer.shap_values(X) -> list per class / ndarray
    返回：explainer, shap_values(正类, shape [n, d]), base_value(float)
    """
    Xt = to_dense(Xt)
    # 优先“概率”空间；若版本不支持则回退到安全的默认(raw)
    try:
        explainer = shap.TreeExplainer(clf, model_output="probability")
    except Exception:
        # 兼容老版本/不同实现
        try:
            explainer = shap.TreeExplainer(clf, model_output="raw")
        except Exception:
            explainer = shap.TreeExplainer(clf)

    # 新接口
    try:
        exp = explainer(Xt, check_additivity=False)
        if hasattr(exp, "values"):
            vals = exp.values
            base = exp.base_values
            if vals.ndim == 2:
                return explainer, vals, float(np.mean(base)) if np.ndim(base) else float(base)
            if vals.ndim == 3:
                return explainer, vals[:, 1, :], float(np.mean(base[:, 1]))
    except Exception:
        pass

    # 旧接口
    sv = explainer.shap_values(Xt)
    if isinstance(sv, list):
        shap_vals = np.asarray(sv[1])
        ev = explainer.expected_value
        base = ev[1] if isinstance(ev, (list, np.ndarray)) else ev
    else:
        shap_vals = np.asarray(sv)
        base = explainer.expected_value
    base = float(np.mean(base)) if np.ndim(base) else float(base)
    return explainer, shap_vals, base

def _load_rf_pipeline_from_s7():
    """尝试从第14步加载RF最佳管线。优先未校准的 *_best_pipeline；找不到则返回 None。"""
    try:
        p_best = S7_MODELS_DIR / "RF_best_pipeline.joblib"
        if p_best.exists():
            pipe = joblib.load(p_best)
            # 期望结构：Pipeline(pre=ColumnTransformer, clf=RandomForestClassifier)
            if hasattr(pipe, "named_steps") and "pre" in pipe.named_steps and "clf" in pipe.named_steps:
                return pipe
    except Exception:
        return None
    return None

# ============== 主流程 ==============
def main():
    # 读取冻结版全量数据与索引
    if not MEAN_MODE_XY.exists():
        raise FileNotFoundError(f"未找到数据：{MEAN_MODE_XY}")
    if not TRAIN_IDX.exists() or not TEST_IDX.exists():
        raise FileNotFoundError(f"未找到索引：\n- {TRAIN_IDX}\n- {TEST_IDX}")
    df_all = pd.read_csv(MEAN_MODE_XY)
    if Y_NAME not in df_all.columns:
        raise KeyError(f"'{Y_NAME}' 不在 {MEAN_MODE_XY}")
    tr = _load_idx(TRAIN_IDX); te = _load_idx(TEST_IDX)
    y_tr = df_all.loc[tr, Y_NAME].astype(int).values
    y_te = df_all.loc[te, Y_NAME].astype(int).values

    # 若强制全特征，则不加载 S7 管线，走本地训练
    pipe = None if USE_ALL_FEATURES else _load_rf_pipeline_from_s7()
    if pipe is not None:
        pre = pipe.named_steps["pre"]
        # 从 ColumnTransformer 提取其期望的原始列集
        try:
            # transformers: List[ (name, transformer, columns) ]
            cols = []
            for name, _, cols_spec in pre.transformers:
                if cols_spec is not None and cols_spec != "drop" and cols_spec != []:
                    cols.extend(list(cols_spec))
            # 去重并保持顺序
            seen = set(); sel_cols = []
            for c in cols:
                if c not in seen and c in df_all.columns:
                    sel_cols.append(c); seen.add(c)
            if not sel_cols:
                sel_cols = [c for c in df_all.columns if c != Y_NAME]
        except Exception:
            sel_cols = [c for c in df_all.columns if c != Y_NAME]
        Xtr = df_all.loc[tr, sel_cols].copy()
        Xte = df_all.loc[te, sel_cols].copy()
        # 无需再 fit；直接使用已保存的 pipe
    else:
        # 使用全量原始特征（去除目标列）；如需限制前K，可把 USE_ALL_FEATURES 置 False
        feats = [c for c in df_all.columns if c != Y_NAME]
        if not feats:
            raise RuntimeError("没有可用特征用于 SHAP。")

        Xtr = df_all.loc[tr, feats].copy()
        Xte = df_all.loc[te, feats].copy()

        # 列类型推断 + 预处理（树模型口径：数值直通，分类 Ordinal 编码；不标准化、不 One-Hot）
        num, cat = infer_types(pd.concat([Xtr, Xte], axis=0), y_name=None)
        pre = ColumnTransformer([
            ("num", "passthrough", num),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat),
        ], remainder="drop")

        rf = RandomForestClassifier(
            n_estimators=600, min_samples_leaf=5,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        )
        pipe = Pipeline([("pre", pre), ("clf", rf)])
        pipe.fit(Xtr, y_tr)

    # 简要性能回显（test） - 显式变换后再预测，避免对象列导致的转换问题
    Xte_t_perf = pipe.named_steps["pre"].transform(Xte)
    p_te = pipe.named_steps["clf"].predict_proba(Xte_t_perf)[:, 1]
    uniq = np.unique(y_te)
    cnts = [(y_te == u).sum() for u in uniq]
    print(f"[test y] classes={uniq.tolist()}, counts={cnts}")
    if len(uniq) > 1:
        try:
            auc = roc_auc_score(y_te, p_te)
            print(f"[AUC test] RF on {Xte_t_perf.shape[1]} features = {auc:.3f}")
        except Exception:
            print("[AUC test] 计算失败（数据异常）")
    else:
        print("[AUC test] 单一类别，跳过AUC计算")

    # 变换后的矩阵与特征名
    Xtr_t = pipe.named_steps["pre"].transform(Xtr)
    try:
        feat_names = pipe.named_steps["pre"].get_feature_names_out()
    except Exception:
        # 兼容旧版：退回原始列名（OrdinalEncoder 不扩增列数）
        feat_names = np.array(list(Xtr.columns))
    clf = pipe.named_steps["clf"]
    
    # 去除特征名前缀（num__, bin__, mul__）
    clean_names = []
    for name in feat_names:
        if "__" in name:
            parts = name.split("__", 1)
            clean_names.append(parts[1])
        else:
            clean_names.append(name)
    
    # 打印并导出特征信息
    print(f"\n[DEBUG] 特征总数: {len(clean_names)}")
    print(f"前10个特征: {clean_names[:10]}")
    print(f"后10个特征: {clean_names[-10:]}")
    
    # 导出所有特征名到CSV（包含原始名和清洁名）
    pd.DataFrame({
        "original_name": feat_names,
        "clean_name": clean_names
    }).to_csv(OUT_DIR / "feature_names.csv", index=True)
    print(f"[save] 特征名已保存到 {OUT_DIR / 'feature_names.csv'}")
    
    # 打印原始特征数量
    print(f"原始特征数量（减去目标列）: {len([c for c in df_all.columns if c != Y_NAME])}")
    print(f"转换后特征数量: {Xtr_t.shape[1]}")

    # SHAP（基于树）——先下采样再计算，降低计算与内存成本
    Xtr_t_dense = to_dense(Xtr_t)
    n_tr = Xtr_t_dense.shape[0]
    if n_tr > MAX_BEESWARM_N:
        rng = np.random.default_rng(RANDOM_STATE)
        idx_shap = rng.choice(n_tr, size=MAX_BEESWARM_N, replace=False)
        X_for_shap = Xtr_t_dense[idx_shap]
    else:
        idx_shap = None
        X_for_shap = Xtr_t_dense

    # 使用 shap.Explainer 代替 TreeExplainer，确保所有特征都被包含
    print(f"\n[INFO] 使用 shap.Explainer 计算所有特征的 SHAP 值...")
    print(f"[DEBUG] 原始特征数量: {len([c for c in df_all.columns if c != Y_NAME])}")
    print(f"[DEBUG] 转换后特征数量: {Xtr_t.shape[1]}")
    print(f"[DEBUG] 特征名数量: {len(feat_names)}")
    
    # 检查特征是否被过滤
    if hasattr(clf, 'feature_importances_'):
        n_features_in = getattr(clf, 'n_features_in_', Xtr_t.shape[1])
        print(f"[DEBUG] RF.n_features_in_: {n_features_in}")
        n_nonzero = np.sum(clf.feature_importances_ > 0)
        print(f"[DEBUG] 非零重要性特征数: {n_nonzero}")
        print(f"[DEBUG] 零重要性特征数: {len(clf.feature_importances_) - n_nonzero}")
    
    # 使用 shap.Explainer 计算 SHAP 值 - 将整个预测管道视为黑盒
    try:
        # 将数据转换为DataFrame，因为Explainer需要DataFrame格式
        X_for_shap_df = pd.DataFrame(X_for_shap, columns=clean_names)
        
        # 使用直接的随机森林模型而不是整个管道
        print("[INFO] 使用随机森林模型计算SHAP值...")
        explainer = shap.TreeExplainer(clf, feature_perturbation="tree_path_dependent")
        shap_values = explainer(X_for_shap_df)
        
        # 提取 SHAP 值
        if hasattr(shap_values, 'values'):
            if shap_values.values.ndim == 3:
                Sv = shap_values.values[:, 1, :]
                base = shap_values.base_values[:, 1].mean() if hasattr(shap_values, 'base_values') else 0.0
            else:
                Sv = shap_values.values
                base = shap_values.base_values.mean() if hasattr(shap_values, 'base_values') else 0.0
        else:
            # 旧版接口
            if isinstance(shap_values, list):
                Sv = shap_values[1]
                base = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
            else:
                Sv = shap_values
                base = explainer.expected_value
        
        # 保持变量名一致性
        Xb = X_for_shap
        
        print(f"[SUCCESS] 成功计算所有特征的 SHAP 值")
        print(f"[DEBUG] SHAP值维度: {Sv.shape}")
        print(f"[DEBUG] 输入矩阵维度: {Xb.shape}")
    except Exception as e:
        print(f"[ERROR] 新版 TreeExplainer 失败: {e}")
        print("[INFO] 回退到原始 TreeExplainer 方法...")
        # 回退到原来的方法
        explainer, Sv, base = robust_tree_shap(clf, X_for_shap)
        Xb = X_for_shap
        print(f"[DEBUG] SHAP值维度(回退方法): {Sv.shape}")

    # 防御性对齐：若列数不一致，按最小列数裁剪，避免 SHAP 版本差异导致的断言
    d_sv = Sv.shape[1]
    d_xb = Xb.shape[1]
    d_fn = len(feat_names) if hasattr(feat_names, '__len__') else d_sv
    
    if not (d_sv == d_xb == d_fn):
        print(f"\n[WARN] 特征维度不一致: SHAP={d_sv}, X={d_xb}, 特征名={d_fn}")
        
        # 尝试扩展特征名而不是裁剪 SHAP 值
        if d_sv > d_fn:
            print(f"[INFO] SHAP维度({d_sv})大于特征名数量({d_fn})，尝试扩展特征名...")
            # 扩展特征名列表
            try:
                extra_names = [f"feature_{i}" for i in range(d_fn, d_sv)]
                feat_names = list(feat_names) + extra_names
                clean_names = list(clean_names) + extra_names
                print(f"[SUCCESS] 特征名已扩展至 {len(clean_names)} 个")
            except Exception as e:
                print(f"[ERROR] 特征名扩展失败: {e}")
                # 回退到裁剪方案
                d = int(min(d_sv, d_xb, d_fn))
                Sv = Sv[:, :d]
                Xb = Xb[:, :d]
                try:
                    feat_names = list(feat_names)[:d]
                    clean_names = list(clean_names)[:d]
                except Exception:
                    pass
        else:
            # 标准裁剪流程
            d = int(min(d_sv, d_xb, d_fn))
            if d <= 0:
                raise RuntimeError("SHAP 对齐失败：无有效特征列用于可视化。")
            print(f"[INFO] 按最小维度 {d} 裁剪特征")
            Sv = Sv[:, :d]
            Xb = Xb[:, :d]
            try:
                feat_names = list(feat_names)[:d]
                clean_names = list(clean_names)[:d]  # 同步裁剪清洁名
            except Exception:
                pass

    # ---------- A: bar ----------
    plt.figure(figsize=(9, 6), dpi=240)
    shap.summary_plot(Sv, features=Xb, feature_names=clean_names,  # 使用清洁后的特征名
                      plot_type="bar", show=False, max_display=TOPK)
    plt.title("SHAP — mean(|SHAP|) (top features)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "Fig6A_shap_bar.png", dpi=300)
    plt.savefig(OUT_DIR / "Fig6A_shap_bar.pdf")
    plt.close()

    # ---------- B: beeswarm ----------
    plt.figure(figsize=(9, 6), dpi=240)
    shap.summary_plot(Sv, features=Xb, feature_names=clean_names,  # 使用清洁后的特征名
                      show=False, max_display=TOPK)
    plt.title("SHAP — beeswarm (train)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "Fig6B_shap_beeswarm.png", dpi=300)
    plt.savefig(OUT_DIR / "Fig6B_shap_beeswarm.pdf")
    plt.close()

    # ---------- C: waterfall （取 train 中预测概率最高的样本） ----------
    p_tr = pipe.named_steps["clf"].predict_proba(Xtr_t)[:, 1]
    idx_top = int(np.argmax(p_tr))
    try:
        # 使用 shap.Explainer 计算单个样本的 SHAP 值
        print(f"\n[INFO] 为预测概率最高的样本计算瀑布图 SHAP 值...")
        x0 = to_dense(Xtr_t)[idx_top:idx_top+1]
        
        # 创建DataFrame格式的单个样本
        try:
            x0_df = pd.DataFrame(x0, columns=clean_names)
            
            # 使用与前面相同的 TreeExplainer
            if hasattr(explainer, 'shap_values'):
                # 使用原始 explainer 的 shap_values 方法
                sv0 = explainer.shap_values(x0_df)
                if isinstance(sv0, list):
                    vals0_row = np.asarray(sv0[1][0])
                    ev = explainer.expected_value
                    base0_scalar = float(ev[1] if isinstance(ev, (list, np.ndarray)) else ev)
                else:
                    vals0_row = np.asarray(sv0[0])
                    base0_scalar = float(getattr(explainer, 'expected_value', base))
            else:
                # 使用新版 explainer 的 __call__ 方法
                exp0 = explainer(x0_df)
                if hasattr(exp0, 'values'):
                    if exp0.values.ndim == 3:
                        vals0_row = exp0.values[0, 1, :]
                        base0_scalar = float(exp0.base_values[0, 1])
                    else:
                        vals0_row = exp0.values[0]
                        base0_scalar = float(exp0.base_values[0]) if hasattr(exp0, 'base_values') else 0.0
                else:
                    raise RuntimeError("SHAP explanation missing 'values'")
        except Exception as e:
            print(f"[WARN] DataFrame方法计算单样本 SHAP 失败: {e}")
            print("[INFO] 回退到原始方法...")
            
            # 直接使用已计算的SHAP值中的对应行
            print("[INFO] 使用已计算的SHAP值中的对应行...")
            if idx_top < len(Sv):
                vals0_row = Sv[idx_top]
                base0_scalar = base
            else:
                # 如果索引超出范围，尝试重新计算
                try:
                    temp_explainer = shap.TreeExplainer(clf)
                    sv0 = temp_explainer.shap_values(x0)
                    if isinstance(sv0, list):
                        vals0_row = np.asarray(sv0[1][0])
                        base0_scalar = float(temp_explainer.expected_value[1])
                    else:
                        vals0_row = np.asarray(sv0[0])
                        base0_scalar = float(temp_explainer.expected_value)
                except Exception:
                    # 最后的备用方案
                    vals0_row = Sv[0] if len(Sv) > 0 else np.zeros(len(clean_names))
                    base0_scalar = base

        # 确保特征维度匹配
        print(f"[DEBUG] 瀑布图 SHAP 值维度: {len(vals0_row)}, 特征名维度: {len(clean_names)}")
        if len(vals0_row) > len(clean_names):
            # 扩展特征名
            extra_names = [f"feature_{i}" for i in range(len(clean_names), len(vals0_row))]
            clean_names_subset = list(clean_names) + extra_names
            print(f"[INFO] 扩展特征名至 {len(clean_names_subset)} 个")
        elif len(vals0_row) < len(clean_names):
            clean_names_subset = clean_names[:len(vals0_row)]
            print(f"[INFO] 裁剪特征名至 {len(clean_names_subset)} 个")
        else:
            clean_names_subset = clean_names

        e = shap.Explanation(values=vals0_row,
                             base_values=base0_scalar,
                             data=to_dense(Xtr_t)[idx_top],
                             feature_names=clean_names_subset)  # 使用清洁后的特征名
        plt.figure(figsize=(8, 6), dpi=240)
        shap.plots.waterfall(e, max_display=TOPK, show=False)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "Fig6C_shap_waterfall.png", dpi=300)
        plt.savefig(OUT_DIR / "Fig6C_shap_waterfall.pdf")
        plt.close()
    except Exception as ex:
        print("[WARN] waterfall 绘制失败：", ex)

    # ---------- D: dependence（Top-9） ----------
    try:
        mean_abs = np.mean(np.abs(Sv), axis=0)
        order = np.argsort(mean_abs)[::-1][:9]
        rows, cols = 3, 3
        fig, axes = plt.subplots(rows, cols, figsize=(10, 8), dpi=240)
        for ax, j in zip(axes.ravel(), order):
            shap.dependence_plot(j, Sv, Xb, feature_names=clean_names, show=False, ax=ax)  # 使用清洁后的特征名
        fig.suptitle("SHAP — dependence (top features)")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "Fig6D_shap_dependence.png", dpi=300)
        fig.savefig(OUT_DIR / "Fig6D_shap_dependence.pdf")
        plt.close(fig)
    except Exception as ex:
        print("[WARN] dependence 绘制失败：", ex)

    print("[save]", OUT_DIR)

if __name__ == "__main__":
    main()
