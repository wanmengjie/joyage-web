# -*- coding: utf-8 -*-
"""
config_io.py
通用 I/O 与路径工具集合（Windows 友好；UTF-8 编码）

主要函数（与现有脚本兼容）：
- dump_json(obj, path)
- load_json(path)

补充：
- dump_yaml(obj, path), load_yaml(path)
- write_csv_smart(df, path, index=False), read_csv_smart(path)
- write_parquet(df, path), read_parquet(path)
- write_pickle_safe(obj, path), read_pickle_safe(path)
- ensure_dir(path)，resolve_path(p, base=None)
- get_project_root(markers=("project_name", ".git", "pyproject.toml"))
- get_logger(name="io")

说明：
- 所有写盘函数自动创建父目录
- 所有文本 I/O 默认 UTF-8（读取 CSV 支持自动识别 UTF-8-SIG/BOM）
"""

from __future__ import annotations

import json
import os
import sys
import io
import pickle
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

# ---- 可选依赖：pandas / yaml / pyarrow ----
try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # 某些环境不需要表格 I/O

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

# ------------------------------------------
# 基础工具
# ------------------------------------------
PathLike = str | os.PathLike[str] | Path


def ensure_dir(path: PathLike) -> Path:
    """确保父目录存在；返回 Path 对象（可用于链式写盘）"""
    p = Path(path)
    if p.suffix:
        p.parent.mkdir(parents=True, exist_ok=True)
    else:
        p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_path(p: PathLike, base: Optional[PathLike] = None) -> Path:
    """将 p 解析为绝对路径；可指定 base 作为相对基准"""
    p = Path(p)
    if not p.is_absolute():
        base_dir = Path(base).resolve() if base else Path.cwd()
        p = (base_dir / p).resolve()
    return p


def get_project_root(
    start: Optional[PathLike] = None,
    markers: Iterable[str] = ("project_name", ".git", "pyproject.toml"),
) -> Path:
    """
    从 start（默认当前文件所在目录）向上搜索，遇到 markers 中任何一个就当作项目根。
    返回找到的目录；找不到则返回 start 的祖先（通常是磁盘根），不抛错。
    """
    here = Path(start).resolve() if start else Path(__file__).resolve().parent
    cur = here
    while True:
        for m in markers:
            if (cur / m).exists():
                return cur
        if cur.parent == cur:
            return here  # 到顶了，回退
        cur = cur.parent


# ------------------------------------------
# JSON
# ------------------------------------------
def dump_json(obj: Any, path: PathLike) -> None:
    """写 JSON（UTF-8，无 ASCII 转义，缩进=2，自动建目录）"""
    p = ensure_dir(path)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: PathLike) -> Any:
    """读 JSON（UTF-8）"""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------
# YAML（可选）
# ------------------------------------------
def dump_yaml(obj: Any, path: PathLike) -> None:
    """写 YAML（需要 pyyaml；UTF-8；自动建目录）"""
    if yaml is None:
        raise ImportError("需要安装 pyyaml 才能 dump_yaml：pip install pyyaml")
    p = ensure_dir(path)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def load_yaml(path: PathLike) -> Any:
    """读 YAML（需要 pyyaml）"""
    if yaml is None:
        raise ImportError("需要安装 pyyaml 才能 load_yaml：pip install pyyaml")
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ------------------------------------------
# CSV（可选，基于 pandas）
# ------------------------------------------
def read_csv_smart(path: PathLike, **kwargs):
    """
    读取 CSV（自动处理 UTF-8-SIG/BOM）
    依赖 pandas；若未安装将抛错。
    """
    if pd is None:
        raise ImportError("需要 pandas 才能读取 CSV：pip install pandas")
    p = Path(path)
    # 自动尝试 UTF-8 与 UTF-8-SIG
    try:
        return pd.read_csv(p, encoding="utf-8", **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(p, encoding="utf-8-sig", **kwargs)


def write_csv_smart(df, path: PathLike, index: bool = False, **kwargs) -> None:
    """写 CSV（UTF-8，自动建目录）"""
    if pd is None:
        raise ImportError("需要 pandas 才能写入 CSV：pip install pandas")
    p = ensure_dir(path)
    df.to_csv(p, index=index, encoding="utf-8", **kwargs)


# ------------------------------------------
# Parquet（可选，基于 pandas/pyarrow）
# ------------------------------------------
def read_parquet(path: PathLike, **kwargs):
    if pd is None:
        raise ImportError("需要 pandas/pyarrow：pip install pandas pyarrow")
    return pd.read_parquet(path, **kwargs)


def write_parquet(df, path: PathLike, **kwargs) -> None:
    if pd is None:
        raise ImportError("需要 pandas/pyarrow：pip install pandas pyarrow")
    p = ensure_dir(path)
    df.to_parquet(p, **kwargs)


# ------------------------------------------
# Pickle
# ------------------------------------------
def write_pickle_safe(obj: Any, path: PathLike) -> None:
    """写二进制对象（自动建目录）"""
    p = ensure_dir(path)
    with p.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle_safe(path: PathLike) -> Any:
    """读二进制对象"""
    with Path(path).open("rb") as f:
        return pickle.load(f)


# ------------------------------------------
# 简易日志器（可选）
# ------------------------------------------
def get_logger(name: str = "io"):
    import logging

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    h = logging.StreamHandler(stream=sys.stdout)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)
    return logger


# ------------------------------------------
# 自检
# ------------------------------------------
if __name__ == "__main__":
    log = get_logger("config_io")
    root = get_project_root()
    log.info(f"project_root = {root}")

    # JSON round-trip
    tmp = root / "_tmp_io_check" / "a.json"
    dump_json({"hello": "world", "num": 42}, tmp)
    data = load_json(tmp)
    log.info(f"json ok: {data}")

    # Pickle round-trip
    pkl = root / "_tmp_io_check" / "b.pkl"
    write_pickle_safe({"x": [1, 2, 3]}, pkl)
    obj = read_pickle_safe(pkl)
    log.info(f"pickle ok: {obj}")

    # CSV/YAML/Parquet 仅在依赖可用时自检
    if pd is not None:
        import pandas as _pd
        df = _pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        csvp = root / "_tmp_io_check" / "c.csv"
        write_csv_smart(df, csvp, index=False)
        _ = read_csv_smart(csvp)
        log.info("csv ok")

    if yaml is not None:
        yp = root / "_tmp_io_check" / "d.yaml"
        dump_yaml({"k": ["v1", "v2"]}, yp)
        _ = load_yaml(yp)
        log.info("yaml ok")

    log.info("self-check done ✔")
