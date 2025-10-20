# -*- coding: utf-8 -*-
# 自动发现所有编号脚本（00_* 到 99_*，以及单独的 26.py），按编号顺序执行
import argparse, subprocess, sys, time, re
from pathlib import Path

SEARCH_DIRS = [Path("."), Path("03_scripts")]

def norm(s: str) -> str:
    return s.lower().replace("\\", "/")

def script_num(name: str) -> int:
    """返回文件名前缀编号，如 '14_final_eval_s7.py' -> 14；'26.py' -> 26；否则 9999"""
    m = re.match(r"^(\d{1,2})(?:_|\.py$)", name)
    return int(m.group(1)) if m else 9999

def discover_steps():
    seen = {}
    for base in SEARCH_DIRS:
        if not base.exists(): 
            continue
        for p in base.rglob("*.py"):
            nm = p.name
            # 只要匹配 NN_*.py 或 NN.py
            if re.match(r"^\d{1,2}(_.+)?\.py$", nm):
                n = script_num(nm)
                # 以“最靠近根目录”的优先（如果同名在多个地方）
                key = (n, nm.lower())
                if key not in seen:
                    seen[key] = p.resolve()
    # 按编号 + 文件名排序
    items = sorted(seen.items(), key=lambda x: (x[0][0], x[0][1]))
    return [p for (_, _), p in items]

def run(py, script: Path, logs: Path, keep: bool):
    log = logs / (script.stem + ".log")
    print(f"[run ] {script} -> {log}")
    with log.open("w", encoding="utf-8") as f:
        f.write(f"== {script} == {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        p = subprocess.Popen([py, str(script)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            sys.stdout.write(line); f.write(line)
        p.wait(); ok = (p.returncode == 0)
        f.write(f"\n[exit] code={p.returncode}\n")
    print(f"[done] {script.name} (code={p.returncode})")
    if (not ok) and (not keep):
        print("[stop] 失败已停止；如需忽略错误继续，请加 --keep-going")
        return False
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", type=str, default=sys.executable)
    ap.add_argument("--only", type=str, help="只跑这些编号/文件名（逗号分隔），如: 00,01,14 或 21_fig3")
    ap.add_argument("--from", dest="frm", type=int, help="从编号起（含），如 00")
    ap.add_argument("--to",   dest="to",  type=int, help="到编号止（含），如 26")
    ap.add_argument("--keep-going", action="store_true")
    args = ap.parse_args()

    root = Path.cwd()
    logs = root / "logs" / ("run_" + time.strftime("%Y%m%d_%H%M%S"))
    logs.mkdir(parents=True, exist_ok=True)

    steps = discover_steps()
    # 过滤 --only / --from / --to
    if args.only:
        tokens = [t.strip().lower() for t in args.only.split(",") if t.strip()]
        def match(s: Path):
            nm = s.name.lower()
            return any(nm.startswith(t) or nm == f"{t}.py" for t in tokens) or any(script_num(nm) == int(t) for t in tokens if t.isdigit())
        steps = [s for s in steps if match(s)]
    if args.frm is not None:
        steps = [s for s in steps if script_num(s.name) >= args.frm]
    if args.to is not None:
        steps = [s for s in steps if script_num(s.name) <= args.to]

    print(f"[cwd ] {root}")
    print(f"[py  ] {args.python}")
    print("[plan]")
    for s in steps: print("   -", s)

    ok_all = True
    for s in steps:
        if not run(args.python, s, logs, args.keep_going):
            ok_all = False; break
    print("[ALL DONE]" if ok_all else "[DONE WITH ERRORS]")

if __name__ == "__main__":
    main()
