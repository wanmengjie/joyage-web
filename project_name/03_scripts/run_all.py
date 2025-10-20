# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, re, subprocess, sys
from pathlib import Path
from datetime import datetime
try:
    import yaml
except Exception:
    yaml = None

ROOT = Path(__file__).resolve().parents[1]       # project_name/
SCRIPTS_DIR = ROOT / "03_scripts"
LOG_DIR = ROOT / "10_experiments" / "_runner_logs"
NUM_PREFIX = re.compile(r"^(\d+)_.*\.py$")

def find_scripts():
    files = []
    for p in sorted(SCRIPTS_DIR.glob("*.py")):
        if p.name in {"run_all.py"}:  # 自身排除
            continue
        files.append(p)
    return files

def numeric_key(p: Path):
    m = NUM_PREFIX.match(p.name)
    if m:
        try: return (0, int(m.group(1)), p.name)
        except: pass
    return (1, 10**9, p.name)

def load_order_yaml():
    yml = SCRIPTS_DIR / "scripts_order.yaml"
    if not yml.exists() or yaml is None:
        return None
    data = yaml.safe_load(yml.read_text(encoding="utf-8"))
    assert isinstance(data, list), "scripts_order.yaml 顶层必须是 list"
    return data

def plan(include: str|None, exclude: str|None, start_from: str|None, start_after: str|None):
    yml = load_order_yaml()
    if yml:
        planned = []
        for item in yml:
            script = SCRIPTS_DIR / item["script"]
            if script.exists():
                planned.append((script, item.get("args","")))
        return planned

    files = find_scripts()
    files.sort(key=numeric_key)

    def filt(lst):
        res = lst
        if include: res = [p for p in res if re.search(include, p.name)]
        if exclude: res = [p for p in res if not re.search(exclude, p.name)]
        return res
    files = filt(files)

    if start_from and start_after:
        raise ValueError("不可同时使用 --start-from 与 --start-after")
    if start_from or start_after:
        key = start_from or start_after
        idx = None
        for i,p in enumerate(files):
            if p.name.startswith(key):
                idx = i; break
        if idx is not None:
            files = files[idx:] if start_from else files[idx+1:]

    return [(p,"") for p in files]

def run_step(py: str, script: Path, extra_args: str, env: dict|None):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logf = LOG_DIR / f"{ts}__{script.name}.log"
    cmd = [py, str(script)] + ([*extra_args.split()] if extra_args else [])
    print(f"\n[run] {script.name}\n      cmd: {' '.join(cmd)}\n      log: {logf}")
    with logf.open("w", encoding="utf-8") as lf:
        lf.write(f"# CMD: {' '.join(cmd)}\n# CWD: {SCRIPTS_DIR}\n\n")
        proc = subprocess.Popen(cmd, cwd=str(SCRIPTS_DIR), stdout=lf, stderr=subprocess.STDOUT, env=env or None)
        return proc.wait()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--include")
    ap.add_argument("--exclude")
    ap.add_argument("--start-from")
    ap.add_argument("--start-after")
    ap.add_argument("--extra-args", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tasks = plan(args.include, args.exclude, args.start_from, args.start_after)
    if not tasks:
        print("[info] 没有要执行的脚本。"); return 0

    print("\n[plan] 即将执行：")
    for p,a in tasks: print(f"  - {p.name}" + (f" [args: {a}]" if a else ""))
    print()

    if args.dry_run:
        print("[dry-run] 仅预演，不执行。"); return 0

    for p,a in tasks:
        merged = " ".join(x for x in [a, args.extra_args] if x).strip()
        code = run_step(args.python, p, merged, None)
        if code != 0:
            print(f"\n[error] 失败：{p.name}（退出码 {code}）")
            return code

    print("\n[done] 全部完成 ✅"); return 0

if __name__ == "__main__":
    import sys as _s; _s.exit(main())
