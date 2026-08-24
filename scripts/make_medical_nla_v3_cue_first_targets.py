"""Rewrite an existing SFT split with cue-first (v3) targets.

v1/v2 targets put the diagnosis label in a closed `<answer>` slot, which the
diagnosis-heldout experiment showed trains a seen-class classifier (heldout
answer_hit 0/800). The v3 target makes the case-specific cue list the only
supervised content. Cue combinations are far higher-entropy than diagnosis
labels, so class memorization alone scores poorly — though not zero: a model
can still emit the typical cues of the nearest seen class, which is why the
acceptance gate also includes cue precision, mismatched-activation ranking,
and cue-removal counterfactuals rather than recall alone.

By default no diagnosis text appears in the target at all: an assessment
sentence naming the diagnosis would reopen the label-shortcut that v3 exists
to close. `--include-assessment` restores it for later variants.

This script deliberately reuses an existing split directory (e.g. the
diagnosis-heldout split) unchanged: same rows, same activations, same
train/val/test_seen/test_heldout assignment. Only `target_text` and
`target_style` change, so v3 results are directly comparable to the v1 run on
the same split.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.jsonl import read_jsonl, write_jsonl

# The conclusion schema's two content fields. Non-greedy and DOTALL so a
# multi-line <supporting_cues> is one span rather than none.
_CONCLUSION_FIELD = re.compile(r"<(answer|supporting_cues)>(.*?)</\1>", re.DOTALL)

SPLITS = ("train", "val", "test_seen", "test_heldout")
TARGET_STYLE = "structured_readout_v3_cue_first"


def xml_text(value: Any) -> str:
    text = " ".join(str(value or "").split())
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def cue_list(row: dict[str, Any], *, max_cues: int, seed: int) -> list[str]:
    """Deduped cues in a per-row deterministic shuffled order, capped at max_cues.

    Shuffling breaks any fixed cue-order pattern the model could latch onto;
    seeding by row id keeps the dataset reproducible.
    """
    cues = row.get("cue_targets") or []
    if isinstance(cues, str):
        cues = [cues]
    deduped: list[str] = []
    for cue in cues:
        text = " ".join(str(cue or "").split())
        if text and text.lower() not in {item.lower() for item in deduped}:
            deduped.append(text)
    rng = random.Random(f"{seed}:{row.get('id')}")
    rng.shuffle(deduped)
    return deduped[:max_cues]


def cue_first_target_text(
    row: dict[str, Any],
    *,
    max_cues: int,
    seed: int,
    include_assessment: bool = False,
) -> str:
    cues = cue_list(row, max_cues=max_cues, seed=seed)
    if not cues:
        raise ValueError(f"Row {row.get('id')} has no cue_targets; cannot build a v3 target.")
    observed = "\n".join(f"- {xml_text(cue)}" for cue in cues)
    lines = [
        "<explanation>",
        "<readout>",
        "<observed>",
        observed,
        "</observed>",
    ]
    if include_assessment:
        dx = xml_text(row.get("diagnosis_name"))
        assessment = (
            f"Findings most consistent with {dx}." if dx else "Findings are non-specific."
        )
        lines.append(f"<assessment>{assessment}</assessment>")
    lines.extend(["</readout>", "</explanation>"])
    return "\n".join(lines)


def content_char_spans(target_text: str) -> list[tuple[int, int]]:
    """Character ranges of the cue text inside a target this module wrote.

    The inverse of `cue_first_target_text`, and here beside it so the two
    cannot drift: everything except the "- <cue>" lines is the same XML in
    every row of the corpus, so a loss averaged over the whole target is mostly
    a measure of how well the adapter has learned six constant lines. The
    training loop uses these spans to report the finding's tokens separately.

    Characters rather than re-tokenized pieces, because the boundaries have to
    be mapped onto the training tokenization rather than change it: the target
    is still encoded in one call, and offsets locate the content within it.

    Two target shapes reach this function. The cue-first targets this module
    writes put each finding on its own "- <cue>" line. The conclusion readouts
    -- the v2 answer-position schema, and the MCR conclusion split built on it
    -- have no bullets at all: their content is what sits inside <answer> and
    <supporting_cues>, and every other character is the same XML in every row.
    Handling only the first shape returned an empty span list for the second,
    which cost a full MCR training run: no content tokens means a content loss
    of NaN, `NaN < best` is False at every epoch, and the loop trained for
    hours without ever saving an adapter. Bullets still take precedence, so
    nothing about the cue-position corpora changes.
    """
    spans = []
    offset = 0
    for line in target_text.split("\n"):
        if line.startswith("- ") and len(line) > 2:
            spans.append((offset + 2, offset + len(line)))
        offset += len(line) + 1
    if spans:
        return spans
    # <task_type> is deliberately absent: it is a constant word in every row,
    # which is the definition of scaffold here.
    for match in _CONCLUSION_FIELD.finditer(target_text):
        start, end = match.span(2)
        if end > start:
            spans.append((start, end))
    return spans


def write_summary(path: Path, *, counts: Counter, cue_counts: list[int], source_dir: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Medical-NLA v3 Cue-First Target Summary\n\n")
        f.write(
            "Same rows/activations/split as the source directory; only the SFT "
            "target changed to cue-first. The OOD gate scores <observed> cue "
            "content, not the <assessment> diagnosis sentence.\n\n"
        )
        f.write(f"- source_split_dir: {source_dir}\n")
        f.write(f"- target_style: {TARGET_STYLE}\n")
        for split in SPLITS:
            f.write(f"- {split}_rows: {counts[split]}\n")
        if cue_counts:
            f.write(f"- mean_cues_per_target: {sum(cue_counts) / len(cue_counts):.2f}\n")
            f.write(f"- min_cues_per_target: {min(cue_counts)}\n")
            f.write(f"- max_cues_per_target: {max(cue_counts)}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-dir",
        required=True,
        help="Existing split directory with sft_*.jsonl, manifest_*.jsonl, metadata.json.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-cues", type=int, default=12)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--include-assessment",
        action="store_true",
        help=(
            "Append a diagnosis-naming <assessment> sentence to each target. "
            "Off by default: naming the diagnosis reopens the label shortcut."
        ),
    )
    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    out_dir = Path(args.out_dir)
    if out_dir.resolve() == split_dir.resolve():
        raise ValueError("--out-dir must differ from --split-dir; the source split stays intact.")
    out_dir.mkdir(parents=True, exist_ok=True)

    counts: Counter = Counter()
    cue_counts: list[int] = []
    for split in SPLITS:
        sft_path = split_dir / f"sft_{split}.jsonl"
        rows_out = []
        for row in read_jsonl(sft_path):
            out = dict(row)
            out["target_text"] = cue_first_target_text(
                row,
                max_cues=args.max_cues,
                seed=args.seed,
                include_assessment=args.include_assessment,
            )
            out["target_style"] = TARGET_STYLE
            rows_out.append(out)
            cue_counts.append(len(cue_list(row, max_cues=args.max_cues, seed=args.seed)))
        counts[split] = len(rows_out)
        write_jsonl(out_dir / f"sft_{split}.jsonl", rows_out)
        manifest_path = split_dir / f"manifest_{split}.jsonl"
        if manifest_path.exists():
            shutil.copy2(manifest_path, out_dir / manifest_path.name)

    source_metadata_path = split_dir / "metadata.json"
    source_metadata = (
        json.loads(source_metadata_path.read_text(encoding="utf-8"))
        if source_metadata_path.exists()
        else {}
    )
    metadata = {
        **source_metadata,
        "source_split_dir": str(split_dir),
        "target_style": TARGET_STYLE,
        "max_cues": args.max_cues,
        "v3_seed": args.seed,
        "include_assessment": args.include_assessment,
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_summary(
        out_dir / "summary.md",
        counts=counts,
        cue_counts=cue_counts,
        source_dir=str(split_dir),
    )
    total = sum(counts.values())
    print(f"[done] wrote v3 cue-first SFT files to {out_dir} ({total} rows)")


if __name__ == "__main__":
    main()
