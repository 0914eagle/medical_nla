from __future__ import annotations

import importlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
from huggingface_hub import snapshot_download

from .nla import cjk_fraction


CONFAB_TERMS = [
    "fever",
    "chills",
    "elevated wbc",
    "white blood cell",
    "creatinine",
    "urinalysis",
    "sepsis",
    "pneumonia",
    "ace inhibitor",
]

HIGH_NORM_THRESHOLD = 12000.0


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def infer_domain(path: str | Path) -> str:
    name = Path(path).name.lower()
    if "medical" in name or "medqa" in name:
        return "medical"
    if "general" in name or "control" in name:
        return "general"
    raise ValueError(f"Could not infer domain from input filename: {path}")


def confab_regex(prompt: str, explanation: str, terms: list[str] | None = None) -> bool:
    terms = terms or CONFAB_TERMS
    prompt_l = prompt.lower()
    expl_l = explanation.lower()
    for term in terms:
        pattern = r"(?<!\w)" + re.escape(term.lower()) + r"(?!\w)"
        if re.search(pattern, expl_l) and not re.search(pattern, prompt_l):
            return True
    return False


def high_norm_flag(row: dict[str, Any], threshold: float = HIGH_NORM_THRESHOLD) -> bool:
    try:
        return float(row.get("activation_norm", 0.0)) > threshold
    except (TypeError, ValueError):
        return False


def resolve_ar_checkpoint(ar: str, cache_dir: str | None) -> str:
    p = Path(ar)
    if p.exists():
        return str(p)
    return snapshot_download(repo_id=ar, cache_dir=cache_dir)


def load_nla_critic_class(extra_path: str | None = None):
    candidate_paths = [extra_path, "/data1/heejae/nla-inference", "nla-inference"]
    for candidate in candidate_paths:
        if candidate and (Path(candidate) / "nla_inference.py").exists():
            sys.path.insert(0, str(candidate))
    try:
        module = importlib.import_module("nla_inference")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not import nla_inference.NLACritic. Install it on the server with:\n"
            "  cd /data1/heejae\n"
            "  git clone https://github.com/kitft/nla-inference\n"
            "  cd nla-inference\n"
            "  pip install -e .\n"
            "or pass --nla-inference-path /data1/heejae/nla-inference."
        ) from exc
    try:
        return module.NLACritic
    except AttributeError as exc:
        raise AttributeError("nla_inference module does not expose NLACritic.") from exc


def torch_dtype(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in ("bf16", "bfloat16"):
        return torch.bfloat16
    if normalized in ("fp16", "float16"):
        return torch.float16
    if normalized in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def load_activation(path: str | Path) -> torch.Tensor:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"activation_path does not exist: {p}")
    activation = torch.load(p, map_location="cpu")
    if activation.ndim != 1:
        raise ValueError(f"Expected 1-D activation at {p}, got shape {tuple(activation.shape)}")
    return activation.float()


def cjk_frac_for_row(row: dict[str, Any]) -> float:
    if "cjk_fraction" in row:
        return float(row["cjk_fraction"])
    if "cjk_frac" in row:
        return float(row["cjk_frac"])
    return cjk_fraction(str(row.get("raw_nla_output") or row.get("nla_output") or ""))


def validate_mse(mse: float, row_id: str) -> None:
    if not math.isfinite(mse):
        raise ValueError(f"Non-finite recon_mse for {row_id}: {mse}")
    if mse < -1e-4 or mse > 4.0001:
        raise ValueError(
            f"recon_mse out of [0,4] for {row_id}: {mse}. "
            "Check AR normalization and do not use AV injection_scale for scoring."
        )


def summarize_scored_rows(rows: list[dict[str, Any]], margin: float = 0.1) -> str:
    cells: dict[tuple[str, bool], list[float]] = defaultdict(list)
    excluded = 0
    for row in rows:
        if row.get("high_norm_flag"):
            excluded += 1
            continue
        cells[(row["domain"], bool(row["confab_regex"]))].append(float(row["recon_mse"]))

    def cell_line(domain: str, confab: bool) -> str:
        values = cells.get((domain, confab), [])
        mean = sum(values) / len(values) if values else float("nan")
        mean_text = f"{mean:.4f}" if values else "NA"
        return f"| {domain} | {confab} | {len(values)} | {mean_text} |"

    med_confab = cells.get(("medical", True), [])
    med_non = cells.get(("medical", False), [])
    if med_confab and med_non:
        confab_mean = sum(med_confab) / len(med_confab)
        non_mean = sum(med_non) / len(med_non)
        if confab_mean <= non_mean + margin:
            verdict = (
                "전제 실증: 복원 통과 + 내용 틀림. "
                f"medical confab mean MSE={confab_mean:.4f}, "
                f"medical non-confab mean MSE={non_mean:.4f}."
            )
        else:
            verdict = (
                "가설 재검토: reconstruction이 confab을 일부 포착. "
                f"medical confab mean MSE={confab_mean:.4f}, "
                f"medical non-confab mean MSE={non_mean:.4f}."
            )
    else:
        verdict = "자동 판정 불가: medical confab 또는 non-confab 셀이 비어 있음."

    lines = [
        "# AR Reconstruction MSE Summary",
        "",
        f"High-norm rows excluded from summary statistics: {excluded}",
        "",
        "| domain | confab_regex | n | mean_recon_mse |",
        "| --- | --- | ---: | ---: |",
        cell_line("general", False),
        cell_line("general", True),
        cell_line("medical", False),
        cell_line("medical", True),
        "",
        "## Automatic Verdict",
        "",
        verdict,
        "",
        "Confabulation here is a regex proxy only; claim-level groundedness is out of scope.",
    ]
    return "\n".join(lines) + "\n"
