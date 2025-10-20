# run_all.ps1 — 一键顺序运行全流程（Windows PowerShell）
$ErrorActionPreference = "Continue"
# 不把原生进程的 stderr 当作错误（避免 Python 打印到 stderr 时被 PowerShell 视作错误）
$PSNativeCommandUseErrorActionPreference = $false

# 项目根（脚本所在目录）
$ROOT    = $PSScriptRoot
$SCRIPTS = Join-Path $ROOT "project_name\03_scripts"
$LOGDIR  = Join-Path $ROOT "project_name\logs"
New-Item -ItemType Directory -Force -Path $LOGDIR | Out-Null

function Run-Step {
  param([string]$py, [string]$args = "")
  $name = [IO.Path]::GetFileNameWithoutExtension($py)
  $log  = Join-Path $LOGDIR ("manual_" + (Get-Date -Format "yyyyMMdd_HHmmss") + "_" + $name + ".log")
  Write-Host ">>> RUN $name" -ForegroundColor Cyan
  # 抑制警告；不中断脚本；根据退出码判断是否失败
  & python -W ignore "$py" $args 2>&1 | Tee-Object -FilePath $log
  if ($LASTEXITCODE -ne 0) {
    Write-Host ">>> FAIL $name (exit=$LASTEXITCODE) | log: $log" -ForegroundColor Red
    throw "step failed: $name"
  }
  Write-Host ">>> DONE $name | log: $log`n" -ForegroundColor Green
}

# 如需“强制重算第14步”，可取消下一行注释：删除历史 CV 结果再跑（可选）
# Get-ChildItem -Path (Join-Path $ROOT "project_name\10_experiments") -Recurse -Filter "cv_results_*.csv" | Remove-Item -Force -ErrorAction SilentlyContinue

# 0x — 基础检查（可选）
Run-Step (Join-Path $SCRIPTS "00_sanity_checks.py")

# 03 — 划分/清洗，已确保排除 cesd
Run-Step (Join-Path $SCRIPTS "03_split_charls.py")

# 04 — 缺失图（有 RuntimeWarning 属正常）
Run-Step (Join-Path $SCRIPTS "04_missing_plots.py")

# 06 — 多种插补（按你要求的7种；如要更快可只跑mice_pmm）
Run-Step (Join-Path $SCRIPTS "06_impute_charls.py") "--all"

# 07 — 给插补结果加标签
Run-Step (Join-Path $SCRIPTS "07_add_label_to_imputed.py")

# 07 — 插补方法选择与图（莫兰迪配色，图例内置）
Run-Step (Join-Path $SCRIPTS "07_imputation_select.py")

# 08 — 冻结 mean_mode（生成 frozen/charls_mean_mode_Xy.csv）
Run-Step (Join-Path $SCRIPTS "08_freeze_mean_mode.py")

# 10 — 特征数量扫描（S4 数据与图）
Run-Step (Join-Path $SCRIPTS "10_feature_count_sweep.py")

# 11 — 嵌套CV调参（树模型无标准化/OHE；已禁用 CatBoost）
Run-Step (Join-Path $SCRIPTS "11_tune_models_nestedcv.py")

# 12 — 特征重要度（S5）
Run-Step (Join-Path $SCRIPTS "12_feature_importance_rfe.py")

# 13 — 敏感性面板（S6）
Run-Step (Join-Path $SCRIPTS "13_sensitivity_panels.py")

# 14 — 最终评估（S7；含CI；禁用 CatBoost；LGB/XGB 走GPU）
Run-Step (Join-Path $SCRIPTS "14_final_eval_s7.py")

# 22 — Fig4（多面板：ROC/Calibration/Metrics/DCA）
Run-Step (Join-Path $SCRIPTS "22_fig4_multi_panels_roc_calib_dca.py")

# 15 — HL 校准面板（可选）
Run-Step (Join-Path $SCRIPTS "15_hosmer_lemeshow_panels.py")

Write-Host "All steps completed." -ForegroundColor Yellow


