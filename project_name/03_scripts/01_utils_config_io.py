# utils/config_io.py
from pathlib import Path
import yaml, json, hashlib, os
from typing import Dict, Any

def repo_root() -> Path:
    # 仓库根目录：按你的结构修改
    return Path(__file__).resolve().parents[2]

def load_cfg() -> Dict[str, Any]:
    cfg_path = repo_root() / "07_config" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p

def sha256_of_file(p: Path, block=1<<20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(block)
            if not b: break
            h.update(b)
    return h.hexdigest()

def dump_json(obj: Any, path: Path):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def outdirs(cfg: Dict[str, Any], ver: str):
    od = cfg.get("outdirs", {})
    base = repo_root() / "10_experiments" / ver
    main = ensure_dir(base / (od.get("main", "main")))
    supp = ensure_dir(base / (od.get("supp", "supp")))
    tuning = ensure_dir(base / "tuning")
    cache = ensure_dir(base / "cache")
    web = ensure_dir(base / "web_model")
    return {"base": base, "main": main, "supp": supp, "tuning": tuning, "cache": cache, "web": web}

def assert_file(p: Path, msg=""):
    if not p.exists():
        raise FileNotFoundError(msg or f"Missing file: {p}")
