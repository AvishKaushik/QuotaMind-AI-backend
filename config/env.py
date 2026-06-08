"""Small .env reader used by runtime modules.

The project keeps credentials in `.env`; this helper loads that file into
process env and provides typed accessors without routing through settings.py.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"


def _load_env_file() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(ENV_PATH, override=False)
        return
    except ImportError:
        pass

    if not ENV_PATH.exists():
        return

    for raw in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env_file()


def get_env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def get_int_env(key: str, default: int) -> int:
    raw = get_env(key)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def get_float_env(key: str, default: float) -> float:
    raw = get_env(key)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def get_list_env(key: str, default: str = "") -> list[str]:
    raw = get_env(key, default)
    return [item.strip() for item in raw.split(",") if item.strip()]
