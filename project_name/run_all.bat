# -*- coding: utf-8 -*-
"""
run_all.py — 自适应总控：在“项目根”和“03_scripts”双目录里找脚本来跑，
工作目录固定为仓库根（含 07_config/config.yaml）。
"""

from __future__ import annotations
import sys, subprocess, time, os
from pathlib import Path
from datetime import datetime

# ───────── 自动定位仓库根 ─────────
HERE = Path(__file__).resolve().parent
def find_repo_root(start: Path) -> Path:
    cur = start
    for _ in range(4):  # 最多向上找 4 层
        if (cur / "07_config" / "config.yaml").exists():
            return cur
        cur = cur.parent
    return start

REPO = find_repo_root(HERE)
SCRIPT_DIRS = [REPO, REPO / "03_scripts"]  # 两处都找
LOG_DIR = REPO / "logs" / datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR.mkdir(parents=True, exist_ok=True)

print(f"[runner] Repo root: {REPO}")
print(f"[runner] Logs -> {LOG_DIR}\n")

# ───────── 你的脚本顺序（兼容 S12/S13 两个命名） ─────────
SCRIPTS = [
    "00_sanity_checks.py",
    "03_split_charls.py",
    "04_missing_plots.py",
    "05_export_with_nan_xy.py",
    "06_impute_charls.py",
    "07_add_label_to_imputed.py",
    "07_imputation_select.py",
    "08_freeze_mean_mode.py",
    "10_feature_count_cv_s4.py",
    "10_feature_count_sweep.py",
    "11_table1_baseline.py",
    "11_tune_models_nestedcv.py",
    "12_feature_importance_rfe.py",
    "13_sensitivity_panels.py",
    "14_final_eval_s7.py",
    "15_hosmer_lemeshow_panels.py",
    "16_tables_S1_S4.py",
    "17_tables_pairwise_stats.py",
    "18_table_S13_subgroups.py",
    # S12/S13 可能叫 17 或 19，两个都尝试
    ("19_tables_S12_S13.py", "17_tables_S12_S13.py"),
    "20_fig2_rcs.py",
    "21_fig3_forest_internal_perf.py",
    "22_fig4_multi_panels_roc_calib_dca.py",
    # 你这份文件名是 23_fig5_rf_heatmap_pretty.py（或旧名 23_fig5_rf_heatmap.py）
    ("23_fig5_rf_heatmap_pretty.py", "23_fig5_rf_heatmap.py"),
    "24_fig6_shap_rf.py",
]

def resolve_script(item) -> Path | None:
    names = item if isinstance(item, tuple) else (item,)
    for name in names:
        for d in SCRIPT_DIRS:
            p = d / name
            if p.exists():
                return p
    return None

def run_script(pyfile: Path):
    t0 = time.time()
    logp = LOG_DIR / f"{pyfile.stem}.log"
    cmd = [sys.executable, "-X", "utf8", str(pyfile)]
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("MPLBACKEND", "Agg")
    with open(logp, "w", encoding="utf-8") as fout:
        fout.write(f"==> CMD: {' '.join(cmd)}\n")
        fout.write(f"==> CWD: {REPO}\n\n")
        try:
            r = subprocess.run(cmd, cwd=str(REPO), stdout=fout, stderr=subprocess.STDOUT, check=False, env=env)
            ok = (r.returncode == 0)
        except Exception as e:
            ok = False
            fout.write("\n[runner] Exception:\n" + repr(e) + "\n")
    return ok, time.time() - t0, logp

def main():
    results = []
    for it in SCRIPTS:
        p = resolve_script(it)
        shown = it if isinstance(it, str) else "/".join(it)
        if p is None:
            print(f"[skip] {shown} 不存在（根目录与 03_scripts 都没找到）")
            results.append((shown, "SKIP", 0, None))
            continue
        print(f"[run ] {p.name} ... ", end="", flush=True)
        ok, dt, logp = run_script(p)
        print(("OK" if ok else "FAIL") + f" ({dt:.1f}s)")
        results.append((p.name, "OK" if ok else "FAIL", dt, logp))

    # 汇总
    w = max(len(n) for n, *_ in results) if results else 18
    okc = sum(1 for _, st, *_ in results if st == "OK")
    print("\n======== SUMMARY ========")
    for n, st, dt, lp in results:
        print(f"{n:<{w}}  {st}  {dt:.1f}s  {lp if lp else ''}")
    print(f"\n[runner] Done. {okc}/{len(results)} succeeded.")
    print(f"[runner] Logs folder: {LOG_DIR}")

if __name__ == "__main__":
    main()
