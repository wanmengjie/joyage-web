import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from pathlib import Path

# 路径设置
OUT_DIR = Path("C:/Users/lenovo/Desktop/20250906 charls  klosa/project_name/10_experiments/v2025-10-03/figs/shap_adaboost_rfe_tree_kernel")
KLOSA_OUT_DIR = OUT_DIR / "klosa"

# 读取SHAP值 - 从正确的子目录中读取
klosa_shap_df = pd.read_csv(KLOSA_OUT_DIR / "klosa_shap_values.csv")
klosa_feature_imp = pd.read_csv(KLOSA_OUT_DIR / "klosa_feature_importance.csv")

# 打印文件信息以确认读取成功
print(f"\n成功读取KLOSA SHAP值文件，形状: {klosa_shap_df.shape}")
print(f"成功读取KLOSA特征重要性文件，形状: {klosa_feature_imp.shape}\n")

# 获取特征名和SHAP值
feature_names = klosa_feature_imp['feature'].tolist()
Sv_klosa = klosa_shap_df.values

# 创建一个简单的数据框作为特征值
X_klosa = pd.DataFrame(np.zeros(Sv_klosa.shape), columns=klosa_shap_df.columns)

# 生成图表
TOPK = 10

# Bar图
plt.figure(figsize=(10, 8), dpi=240)
shap.summary_plot(Sv_klosa, features=X_klosa, feature_names=feature_names,
                 plot_type="bar", show=False, max_display=TOPK)
plt.title(f"SHAP — mean(|SHAP|) (KLOSA, top {TOPK})")
plt.tight_layout()
plt.savefig(KLOSA_OUT_DIR / "shap_bar.png", dpi=300)
print(f"已保存KLOSA Bar图到 {KLOSA_OUT_DIR / 'shap_bar.png'}")
plt.close()

# Beeswarm图
plt.figure(figsize=(10, 8), dpi=240)
shap.summary_plot(Sv_klosa, features=X_klosa, feature_names=feature_names,
                 show=False, max_display=TOPK)
plt.title(f"SHAP — beeswarm (KLOSA, top {TOPK})")
plt.tight_layout()
plt.savefig(KLOSA_OUT_DIR / "shap_beeswarm.png", dpi=300)
print(f"已保存KLOSA Beeswarm图到 {KLOSA_OUT_DIR / 'shap_beeswarm.png'}")
plt.close()

# 依赖图
top_idx = np.argsort(np.mean(np.abs(Sv_klosa), axis=0))[::-1][:9]
fig, axes = plt.subplots(3, 3, figsize=(15, 15), dpi=240)
for i, idx in enumerate(top_idx):
    ax = axes[i // 3, i % 3]
    shap.dependence_plot(
        ind=idx, 
        shap_values=Sv_klosa, 
        features=X_klosa,
        feature_names=feature_names, 
        show=False, 
        ax=ax
    )
fig.suptitle(f"SHAP — dependence (KLOSA top 9 features)")
fig.tight_layout()
fig.savefig(KLOSA_OUT_DIR / "shap_dependence.png", dpi=300)
print(f"已保存KLOSA依赖图到 {KLOSA_OUT_DIR / 'shap_dependence.png'}")
plt.close(fig)

print("\n所有KLOSA图表已生成完成!")

   