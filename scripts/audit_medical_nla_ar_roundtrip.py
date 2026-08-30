"""Prepare and score the D22 public-AR matched-versus-shuffled diagnostic.

The diagnostic is validation-only. DiReCT text remains in private storage;
only aggregate arm metrics are suitable for version control.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.jsonl import read_jsonl, write_jsonl
from src.reconstruction_scoring import (
    load_activation,
    load_nla_critic_class,
    resolve_ar_checkpoint,
    torch_dtype,
)


def clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_path_map(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected OLD=NEW")
    old, new = value.split("=", 1)
    if not old or not new:
        raise argparse.ArgumentTypeError("Expected non-empty OLD=NEW")
    return old, new


def mapped_path(value: Any, mappings: list[tuple[str, str]]) -> Path:
    text = str(value or "")
    for old, new in mappings:
        if text.startswith(old):
            text = new + text[len(old) :]
            break
    return Path(text)


def base_id(row: dict[str, Any]) -> str:
    return str(row.get("base_id") or row.get("id") or "")


def index_rows(path: Path, *, original_only: bool = False) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if original_only and str(row.get("variant") or "original") != "original":
            continue
        identifier = str(row.get("id") or "")
        if not identifier or identifier in output:
            raise ValueError(f"Missing or duplicate id in {path}: {identifier!r}")
        output[identifier] = row
    if not output:
        raise ValueError(f"No rows in {path}")
    return output


def deterministic_donors(
    rows: list[dict[str, Any]], *, namespace: str
) -> dict[str, str]:
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_stratum[str(row["stratum"])].append(str(row["base_id"]))
    donors: dict[str, str] = {}
    for stratum, identifiers in sorted(by_stratum.items()):
        ordered = sorted(
            set(identifiers),
            key=lambda value: hashlib.sha256(
                f"{namespace}\0{stratum}\0{value}".encode()
            ).hexdigest(),
        )
        if len(ordered) < 2:
            continue
        for index, identifier in enumerate(ordered):
            donors[identifier] = ordered[(index + 1) % len(ordered)]
    return donors


def prepare(args: argparse.Namespace) -> None:
    direct_manifest = {
        base_id(row): row for row in read_jsonl(args.direct_manifest) if base_id(row)
    }
    if not direct_manifest:
        raise ValueError("No DiReCT manifest rows")
    direct_texts: dict[tuple[str, str], str] = {}
    for request in read_jsonl(args.direct_private_bundle):
        identifier = str(request["base_id"])
        for method in request["methods"]:
            key = (identifier, str(method["method"]))
            text = str(method.get("method_output") or "").strip()
            if key in direct_texts and clean(direct_texts[key]) != clean(text):
                raise ValueError(f"Conflicting DiReCT text for {key}")
            direct_texts[key] = text

    prepared: list[dict[str, Any]] = []
    direct_cases: dict[str, dict[str, Any]] = {}
    for identifier, manifest in direct_manifest.items():
        activation = mapped_path(manifest.get("activation_path"), args.path_map)
        if not activation.is_file():
            raise FileNotFoundError(activation)
        stratum = str(
            manifest.get("disease_category")
            or manifest.get("diagnosis_id")
            or manifest.get("canonical_pdd")
            or ""
        )
        if not stratum:
            raise ValueError(f"No DiReCT diagnosis stratum for {identifier}")
        direct_cases[identifier] = {
            "activation_path": str(activation),
            "stratum": stratum,
        }
    direct_donors = deterministic_donors(
        [
            {"base_id": identifier, "stratum": values["stratum"]}
            for identifier, values in direct_cases.items()
        ],
        namespace="direct-d22",
    )
    for (identifier, method), text in sorted(direct_texts.items()):
        if identifier not in direct_cases or identifier not in direct_donors:
            continue
        donor = direct_donors[identifier]
        prepared.append(
            {
                "id": f"direct::{identifier}::{method}",
                "dataset": "direct",
                "arm": method,
                "base_id": identifier,
                "stratum": direct_cases[identifier]["stratum"],
                "text": text,
                "activation_path": direct_cases[identifier]["activation_path"],
                "donor_base_id": donor,
                "donor_activation_path": direct_cases[donor]["activation_path"],
            }
        )

    ddx_manifest = index_rows(args.ddx_manifest, original_only=True)
    ddx_readouts = index_rows(args.structured_reader, original_only=True)
    common_ids = sorted(set(ddx_manifest) & set(ddx_readouts))
    ddx_cases = []
    for identifier in common_ids:
        manifest = ddx_manifest[identifier]
        activation = mapped_path(manifest.get("activation_path"), args.path_map)
        if not activation.is_file():
            raise FileNotFoundError(activation)
        stratum = str(manifest.get("diagnosis_id") or "")
        if not stratum:
            raise ValueError(f"No DDXPlus diagnosis for {identifier}")
        ddx_cases.append(
            {
                "base_id": identifier,
                "stratum": stratum,
                "activation_path": str(activation),
                "text": str(ddx_readouts[identifier].get("observed") or "").strip(),
            }
        )
    ddx_donors = deterministic_donors(ddx_cases, namespace="ddxplus-d22")
    ddx_by_id = {row["base_id"]: row for row in ddx_cases}
    for row in ddx_cases:
        identifier = row["base_id"]
        if identifier not in ddx_donors:
            continue
        donor = ddx_donors[identifier]
        prepared.append(
            {
                "id": f"ddxplus::{identifier}::structured_reader",
                "dataset": "ddxplus",
                "arm": "structured_reader",
                "base_id": identifier,
                "stratum": row["stratum"],
                "text": row["text"],
                "activation_path": row["activation_path"],
                "donor_base_id": donor,
                "donor_activation_path": ddx_by_id[donor]["activation_path"],
            }
        )

    if args.limit_per_arm is not None:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in prepared:
            grouped[f"{row['dataset']}::{row['arm']}"].append(row)
        prepared = []
        for label, rows in sorted(grouped.items()):
            rows.sort(
                key=lambda row: hashlib.sha256(
                    f"17\0{label}\0{row['base_id']}".encode()
                ).hexdigest()
            )
            prepared.extend(rows[: args.limit_per_arm])

    if not prepared:
        raise ValueError("No D22 diagnostic rows")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "private_manifest.jsonl", prepared)
    counts: dict[str, int] = defaultdict(int)
    for row in prepared:
        counts[f"{row['dataset']}::{row['arm']}"] += 1
    protocol = {
        "schema_version": 1,
        "validation_only": True,
        "locked_test_read": False,
        "control": "deterministic different-case activation within diagnosis stratum",
        "limit_per_arm": args.limit_per_arm,
        "counts": dict(sorted(counts.items())),
        "sources": {
            str(path): sha256_file(path)
            for path in (
                args.direct_manifest,
                args.direct_private_bundle,
                args.ddx_manifest,
                args.structured_reader,
            )
        },
    }
    (args.out_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[prepare] rows={len(prepared)} arms={len(counts)}")
    for label, count in sorted(counts.items()):
        print(f"[arm] {label} n={count}")


def reconstruct(critic: Any, text: str) -> torch.Tensor:
    with torch.inference_mode():
        vector = critic.reconstruct(text)
    if not isinstance(vector, torch.Tensor):
        vector = torch.as_tensor(vector)
    vector = vector.detach().float().cpu().flatten()
    if vector.ndim != 1 or not torch.isfinite(vector).all():
        raise ValueError("AR returned an invalid reconstruction")
    return vector


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() != right.numel():
        raise ValueError(f"Vector width mismatch: {left.numel()} != {right.numel()}")
    value = float(F.cosine_similarity(left, right.float().flatten(), dim=0))
    if not math.isfinite(value):
        raise ValueError("Non-finite cosine")
    return value


def score(args: argparse.Namespace) -> None:
    rows = list(read_jsonl(args.manifest))
    if not rows:
        raise ValueError("Empty D22 manifest")
    ar_dir = resolve_ar_checkpoint(args.ar, args.cache_dir)
    critic_cls = load_nla_critic_class(args.nla_inference_path)
    critic = critic_cls(ar_dir, device=args.device, dtype=torch_dtype(args.dtype))
    scored = []
    for index, row in enumerate(rows, start=1):
        vector = reconstruct(critic, str(row["text"]))
        own = load_activation(row["activation_path"])
        donor = load_activation(row["donor_activation_path"])
        own_cos = cosine(vector, own)
        donor_cos = cosine(vector, donor)
        scored.append(
            {
                **row,
                "reconstruction_cosine_own": own_cos,
                "reconstruction_cosine_shuffled": donor_cos,
                "matched_over_shuffled_gap": own_cos - donor_cos,
                "word_count": len(clean(row["text"]).split()),
            }
        )
        if index % 25 == 0 or index == len(rows):
            print(f"[score] {index}/{len(rows)}", flush=True)
    write_jsonl(args.output, scored)
    print(f"[scored] {len(scored)} -> {args.output}")


def bootstrap_ci(values: list[float], *, seed: int = 17) -> tuple[float, float]:
    rng = random.Random(seed)
    draws = []
    for _ in range(5000):
        sample = [values[rng.randrange(len(values))] for _ in values]
        draws.append(statistics.fmean(sample))
    draws.sort()
    return draws[124], draws[4874]


def summarize(args: argparse.Namespace) -> None:
    rows = list(read_jsonl(args.scored))
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), str(row["arm"]))].append(row)
    metrics = {}
    for (dataset, arm), arm_rows in sorted(grouped.items()):
        own = [float(row["reconstruction_cosine_own"]) for row in arm_rows]
        shuffled = [float(row["reconstruction_cosine_shuffled"]) for row in arm_rows]
        gaps = [float(row["matched_over_shuffled_gap"]) for row in arm_rows]
        lower, upper = bootstrap_ci(gaps)
        metrics[f"{dataset}::{arm}"] = {
            "n": len(arm_rows),
            "mean_own_cosine": statistics.fmean(own),
            "mean_shuffled_cosine": statistics.fmean(shuffled),
            "mean_gap": statistics.fmean(gaps),
            "bootstrap_95_ci": [lower, upper],
            "positive_gap_rate": sum(value > 0 for value in gaps) / len(gaps),
            "mean_word_count": statistics.fmean(
                float(row["word_count"]) for row in arm_rows
            ),
        }
    reader = metrics.get("ddxplus::structured_reader")
    source = metrics.get("direct::source_cot")
    positive_controls_passed = bool(
        reader
        and source
        and reader["bootstrap_95_ci"][0] > 0
        and source["bootstrap_95_ci"][0] > 0
    )
    result = {
        "schema_version": 1,
        "validation_only": True,
        "locked_test_read": False,
        "positive_control_rule": (
            "structured_reader and source_cot matched-over-shuffled bootstrap lower bounds > 0"
        ),
        "positive_controls_passed": positive_controls_passed,
        "arms": metrics,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# D22 Public-AR Matched-vs-Shuffled Diagnostic",
        "",
        "Validation-only. The same reconstructed vector is compared with its own activation",
        "and a deterministic different-case activation from the same diagnosis stratum.",
        "",
        f"- positive controls passed: **{positive_controls_passed}**",
        "- locked test read: **no**",
        "",
        "| dataset | arm | n | own cosine | shuffled cosine | gap | bootstrap 95% CI | own win | words |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, values in metrics.items():
        dataset, arm = key.split("::", 1)
        lower, upper = values["bootstrap_95_ci"]
        lines.append(
            f"| {dataset} | {arm} | {values['n']} | "
            f"{values['mean_own_cosine']:.4f} | "
            f"{values['mean_shuffled_cosine']:.4f} | "
            f"{values['mean_gap']:+.4f} | [{lower:+.4f}, {upper:+.4f}] | "
            f"{values['positive_gap_rate']:.4f} | {values['mean_word_count']:.1f} |"
        )
    lines.extend(
        [
            "",
            "A positive-control failure means the released AR is not a valid measurement",
            "instrument for this medical distribution. It does not establish that the text",
            "or activation lacks clinical information.",
            "",
        ]
    )
    args.summary_md.write_text("\n".join(lines), encoding="utf-8")
    print(args.summary_md.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prep = subparsers.add_parser("prepare")
    prep.add_argument("--direct-manifest", required=True, type=Path)
    prep.add_argument("--direct-private-bundle", required=True, type=Path)
    prep.add_argument("--ddx-manifest", required=True, type=Path)
    prep.add_argument("--structured-reader", required=True, type=Path)
    prep.add_argument("--path-map", action="append", default=[], type=parse_path_map)
    prep.add_argument("--limit-per-arm", type=int)
    prep.add_argument("--out-dir", required=True, type=Path)

    run = subparsers.add_parser("score")
    run.add_argument("--manifest", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--ar", default="kitft/nla-gemma3-12b-L32-ar")
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--dtype", default="bfloat16")
    run.add_argument("--cache-dir", default="/data1/heejae/hf_cache")
    run.add_argument("--nla-inference-path")

    report = subparsers.add_parser("summarize")
    report.add_argument("--scored", required=True, type=Path)
    report.add_argument("--output-json", required=True, type=Path)
    report.add_argument("--summary-md", required=True, type=Path)

    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "score":
        score(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
