"""Publish the processed case corpora to a Hugging Face dataset repo.

Regenerating DDXPlus means rescanning a million-row CSV and replaying every
cue-rendering decision; MedCaseReasoning means re-segmenting 12k case reports.
Neither is expensive in compute, but both are expensive in decisions, and a
server move should not be an opportunity to silently pick different ones.
Pushing the artifacts makes the next machine's corpus identical by construction
rather than by re-running the same commands correctly.

The dataset card is generated from the artifacts themselves: the generator flags
that shaped the corpus are recorded on every case row, so the card reports what
the data actually is rather than what a command line was supposed to do.

Activations are deliberately not covered here. They are a function of the
prompt, the backbone and the layer, so they are only worth publishing once the
prompt is frozen -- and being derived from Gemma, they carry that model's terms
in a way these text artifacts do not.

Private by default. Making it public redistributes derivatives of the source
corpora, so check the attribution in the card first.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.jsonl import read_jsonl

# Flags recorded on case rows that change what reaches the prompt. Reporting
# them in the card is what makes a downloaded corpus interpretable.
PROVENANCE_FIELDS = ("clean_cues", "negative_cues", "prefer_symptoms", "target_style")

SOURCES = {
    "ddxplus": (
        "[DDXPlus](https://huggingface.co/datasets/aai530-group6/ddxplus) "
        "(Tchango et al., NeurIPS 2022 Datasets & Benchmarks), CC BY 4.0"
    ),
    "mcr": (
        "[MedCaseReasoning](https://huggingface.co/datasets/zou-lab/MedCaseReasoning) "
        "(Stanford), CC BY 4.0"
    ),
}


def describe_artifact(path: Path, sample: int = 2000) -> dict[str, Any]:
    """Summarize one JSONL artifact, including the flags that shaped it."""
    rows = []
    for index, row in enumerate(read_jsonl(str(path))):
        if index >= sample:
            break
        rows.append(row)
    n_lines = sum(1 for _ in path.open(encoding="utf-8"))

    provenance = {
        field: rows[0][field] for field in PROVENANCE_FIELDS if rows and field in rows[0]
    }
    # Case files carry a cue list; cue-position rows carry one cue each.
    cue_counts = [
        len(row.get("cue_targets") or []) if row.get("cue_targets") else (1 if row.get("cue_text") else 0)
        for row in rows
    ]
    polarity: Counter = Counter()
    for row in rows:
        for value in row.get("cue_polarities") or []:
            polarity[str(value)] += 1

    return {
        "name": path.name,
        "rows": n_lines,
        "sampled": len(rows),
        "provenance": provenance,
        "mean_cues": round(sum(cue_counts) / len(cue_counts), 2) if cue_counts else None,
        "max_cues": max(cue_counts) if cue_counts else None,
        "cue_polarity": dict(polarity),
        "fields": sorted(rows[0]) if rows else [],
        "example_prompt": (rows[0].get("prompt") if rows else None),
    }


def write_parquet(path: Path, out_dir: Path) -> Path | None:
    """Also publish Parquet, so the dataset is browsable without our loader.

    Not CSV: these rows carry list-valued fields (`cue_targets`,
    `cue_polarities`) and prompts containing newlines, both of which CSV
    flattens into strings that the reader has to parse back -- the same class of
    bug that stringified evidence lists caused upstream. Parquet keeps the types.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        print("[parquet] pyarrow not installed; skipping (JSONL is the source of truth)")
        return None

    rows = list(read_jsonl(str(path)))
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{path.stem}.parquet"
    pq.write_table(pa.Table.from_pylist(rows), target, compression="zstd")
    return target


def build_card(repo_id: str, artifacts: list[dict[str, Any]], private: bool) -> str:
    lines = [
        "---",
        "license: cc-by-4.0",
        "language:",
        "  - en",
        "tags:",
        "  - medical",
        "  - interpretability",
        "---",
        "",
        f"# {repo_id}",
        "",
        "Processed case corpora for the Medical-NLA experiments: prompts whose",
        "clinical evidence spans are known exactly, so an activation taken at a",
        "span can be scored against the text that produced it.",
        "",
        "These are derivatives of the datasets listed under Sources; they contain",
        "no model outputs. Activations are not included, since they depend on the",
        "backbone and layer and are only meaningful with a frozen prompt.",
        "",
        "`data/` holds JSONL, which is what the pipeline reads and the source of",
        "truth. `parquet/` holds the same rows for the Hub viewer. There is no CSV:",
        "these rows carry list-valued fields and prompts containing newlines, and",
        "CSV would flatten both into strings the reader has to parse back.",
        "",
        "## Artifacts",
        "",
    ]
    for artifact in artifacts:
        lines.append(f"### `{artifact['name']}`")
        lines.append("")
        lines.append(f"- rows: {artifact['rows']:,}")
        sampled = artifact["sampled"]
        exact = sampled >= artifact["rows"]
        note = "" if exact else f" (first {sampled:,} rows)"
        if artifact["mean_cues"] is not None:
            lines.append(
                f"- cues per case: mean {artifact['mean_cues']}, "
                f"max {artifact['max_cues']}{note}"
            )
        if artifact["cue_polarity"]:
            lines.append(f"- cue polarity: {artifact['cue_polarity']}{note}")
        if artifact["provenance"]:
            lines.append("- generator settings recorded on every row:")
            for key, value in artifact["provenance"].items():
                lines.append(f"  - `{key}`: `{value}`")
        if artifact["fields"]:
            lines.append(f"- fields: {', '.join(f'`{f}`' for f in artifact['fields'])}")
        if artifact["example_prompt"]:
            lines.append("")
            lines.append("Example prompt:")
            lines.append("")
            lines.append("```text")
            lines.append(str(artifact["example_prompt"]))
            lines.append("```")
        lines.append("")

    lines += [
        "## How the cues are defined",
        "",
        "**DDXPlus** stores `(question id, answer value)` rather than sentences, so",
        "each cue phrase is built here: the subject-auxiliary inversion is undone",
        "(`Is the rash swollen?` becomes `the rash is swollen`), answers are placed",
        "inside the statement rather than appended, negatively-answered items are",
        "rendered by negating the auxiliary (`has not traveled out of the country`),",
        "and one item answered with several values becomes one cue naming them all.",
        "Anything still shaped like a question is dropped rather than emitted.",
        "",
        "**MedCaseReasoning** cues are clause spans cut from the patient",
        "presentation itself, so every cue is verbatim in its prompt by",
        "construction. The quoted spans inside `diagnostic_reasoning` are *not*",
        "used: only 1.7% of them appear in the case prompt, 42.7% appear only in",
        "the full article, and the rest are the reasoning's own paraphrase, so that",
        "field does not annotate which part of the presentation is evidence.",
        "",
        "Every cue is an exact substring of its own `prompt`, which is what lets",
        "extraction resolve it to a token span.",
        "",
        "## Sources",
        "",
    ]
    seen: set[str] = set()
    for artifact in artifacts:
        for key, citation in SOURCES.items():
            if key in artifact["name"].lower() and key not in seen:
                seen.add(key)
                lines.append(f"- {citation}")
    if not seen:
        lines.extend(f"- {citation}" for citation in SOURCES.values())
    lines += [
        "",
        "Redistributed under CC BY 4.0 with attribution to the sources above.",
        "",
    ]
    if private:
        lines.append("_This repository is private. Review the attribution above before making it public._")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="e.g. kitft/medical-nla-cases")
    parser.add_argument("--files", nargs="+", required=True, help="JSONL artifacts to upload.")
    parser.add_argument(
        "--private",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Private by default; publishing redistributes derivatives of the sources.",
    )
    parser.add_argument("--path-in-repo", default="data")
    parser.add_argument(
        "--parquet",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also upload a Parquet copy, so the Hub viewer can render the rows.",
    )
    parser.add_argument("--parquet-dir", default=None, help="Where to write Parquet copies.")
    parser.add_argument("--card-only", action="store_true", help="Write the card locally and stop.")
    parser.add_argument("--card-out", default=None)
    args = parser.parse_args()

    paths = [Path(path) for path in args.files]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise SystemExit(f"missing artifacts: {missing}")

    artifacts = [describe_artifact(path) for path in paths]
    card = build_card(args.repo_id, artifacts, private=args.private)

    if args.card_out:
        Path(args.card_out).write_text(card, encoding="utf-8")
        print(f"[card] wrote {args.card_out}")
    if args.card_only:
        print(card)
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    parquet_dir = Path(args.parquet_dir or (paths[0].parent / "parquet"))
    for path, artifact in zip(paths, artifacts):
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=f"{args.path_in_repo}/{path.name}",
            repo_id=args.repo_id,
            repo_type="dataset",
        )
        print(f"[push] {path.name} ({artifact['rows']:,} rows)")
        if args.parquet:
            table = write_parquet(path, parquet_dir)
            if table is not None:
                api.upload_file(
                    path_or_fileobj=str(table),
                    path_in_repo=f"parquet/{table.name}",
                    repo_id=args.repo_id,
                    repo_type="dataset",
                )
                print(f"[push] {table.name}")
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
    )
    visibility = "private" if args.private else "PUBLIC"
    print(f"[done] {visibility} -> https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
