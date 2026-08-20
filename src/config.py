from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

# The whole reason the config files exist in duplicate was the disk: the same
# run needed /data on one machine, /data1 on another, /data3 on a third. Nine
# files then had to be edited in step on every move, and were not -- the
# per-GPU memory budget added after a placement failure went into four of them
# and not the other five. One variable per machine replaces that.
_DEFAULTS = {
    "MEDICAL_NLA_DATA_ROOT": "/data1/heejae",
    "MEDICAL_NLA_CODE_ROOT": str(REPO_ROOT),
}

_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _substitute(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        value = os.environ.get(name, _DEFAULTS.get(name))
        if value is None:
            # Left as a literal, this becomes a directory named "${VAR}" that a
            # run happily writes into and nobody finds again.
            raise KeyError(
                f"Config refers to ${{{name}}}, which is neither set in the "
                f"environment nor given a default. Known defaults: "
                f"{sorted(_DEFAULTS)}."
            )
        return value

    return os.path.expanduser(_PLACEHOLDER.sub(replace, text))


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return _substitute(value)
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a config, resolving ${VAR} against the environment.

    Substitution is applied to every string in the file, not only to paths, so
    a machine-specific value can be lifted out wherever it appears. An
    unresolved placeholder raises rather than surviving into a filename.
    """
    with Path(path).open("r", encoding="utf-8") as f:
        return _expand(yaml.safe_load(f))


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def torch_dtype(name: str):
    import torch

    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return mapping[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {name}") from exc
