"""Summarize seen vs heldout Medical-NLA readouts for the OOD experiment.

Inputs are scored rows from `scripts/score_medical_nla_v2_readouts.py` for the
`test_seen` and `test_heldout` manifests of a diagnosis-heldout split, plus the
split directory (for the train diagnosis vocabulary).

Beyond answer_hit / cue_recall per pool, this reports the classifier-collapse
check: on heldout rows, how often does the readout `<answer>` fall inside the
train-class name/alias vocabulary? A model that only ever emits train-class
names on unseen diagnoses is behaving like a seen-class classifier, whatever
its seen-class accuracy is.

Interpretation guide (handoff doc):
- heldout answer_hit low + cue_recall high: reads cue semantics, does not
  generalize diagnosis names.
- heldout answer_hit low + cue_recall low: likely a seen-class classifier.
- heldout answer_hit high + cue_recall high: strong OOD readout.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.score_specificity_outputs import contains_term
from src.jsonl import read_jsonl, write_jsonl


def train_vocabulary(sft_train_path: Path) -> dict[str, list[str]]:
    """Map train diagnosis_id -> deduped name/alias terms."""
    vocab: dict[str, list[str]] = {}
    for row in read_jsonl(sft_train_path):
        diagnosis_id = str(row["diagnosis_id"])
        if diagnosis_id in vocab:
            continue
        terms: list[str] = []
        for value in [
            row.get("diagnosis_name"),
            *(row.get("diagnosis_aliases") or []),
            diagnosis_id.replace("_", " "),
        ]:
            text = " ".join(str(value or "").split())
            if text and text.lower() not in {term.lower() for term in terms}:
                terms.append(text)
        vocab[diagnosis_id] = terms
    if not vocab:
        raise ValueError(f"No train vocabulary rows in {sft_train_path}")
    return vocab


def train_vocab_matches(answer: str, vocab: dict[str, list[str]]) -> list[str]:
    if not answer:
        return []
    return [
        diagnosis_id
        for diagnosis_id, terms in sorted(vocab.items())
        if any(contains_term(answer, term) for term in terms)
    ]


def pool_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    cue_recalls = [float(row["cue_recall"]) for row in rows if row.get("cue_recall") is not None]
    return {
        "n": n,
        "parsed_readout": sum(bool(row.get("parsed_readout")) for row in rows),
        "parsed_answer": sum(bool(row.get("parsed_answer")) for row in rows),
        "answer_hit": sum(bool(row.get("answer_hit")) for row in rows),
        "answer_hit_rate": sum(bool(row.get("answer_hit")) for row in rows) / n if n else 0.0,
        "output_answer_hit_rate": sum(bool(row.get("output_answer_hit")) for row in rows) / n
        if n
        else 0.0,
        "mean_cue_recall": mean(cue_recalls) if cue_recalls else 0.0,
        "cue_recall_n": len(cue_recalls),
    }


def interpretation(
    heldout: dict[str, Any],
    *,
    answer_hit_high: float,
    cue_recall_high: float,
) -> str:
    hit_high = heldout["answer_hit_rate"] >= answer_hit_high
    recall_high = heldout["mean_cue_recall"] >= cue_recall_high
    if hit_high and recall_high:
        return "strong OOD readout: heldout answer_hit and cue_recall are both high"
    if not hit_high and recall_high:
        return (
            "semantic-but-not-generalizing: cue semantics are read on unseen diagnoses, "
            "but diagnosis names do not generalize"
        )
    if not hit_high and not recall_high:
        return "likely seen-class classifier: neither answers nor cues transfer to unseen diagnoses"
    return (
        "answer_hit high but cue_recall low on heldout: check for alias leakage or "
        "cue-target coverage issues before interpreting"
    )


def write_summary(
    path: Path,
    *,
    seen: dict[str, Any] | None,
    heldout: dict[str, Any],
    heldout_rows: list[dict[str, Any]],
    heldout_diagnoses: list[str],
    answer_hit_high: float,
    cue_recall_high: float,
) -> None:
    in_vocab = [row for row in heldout_rows if row["answer_in_train_vocab"]]
    by_dx: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in heldout_rows:
        by_dx[str(row.get("diagnosis_id") or row.get("gold_diagnosis_id"))].append(row)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Diagnosis-Heldout Readout Summary\n\n")
        f.write("Lexical alias/cue screening only; same scorer as v2 readouts.\n\n")
        f.write("## Seen vs Heldout\n\n")
        f.write(
            "| pool | n | parsed_readout | answer_hit_rate "
            "| output_answer_hit_rate | mean_cue_recall |\n"
        )
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for label, metrics in (("test_seen", seen), ("test_heldout", heldout)):
            if metrics is None:
                continue
            f.write(
                f"| {label} | {metrics['n']} | {metrics['parsed_readout']} | "
                f"{metrics['answer_hit_rate']:.4f} | {metrics['output_answer_hit_rate']:.4f} | "
                f"{metrics['mean_cue_recall']:.4f} |\n"
            )
        if seen is not None:
            f.write(
                f"\n- seen_minus_heldout_answer_hit: "
                f"{seen['answer_hit_rate'] - heldout['answer_hit_rate']:+.4f}\n"
            )
            f.write(
                f"- seen_minus_heldout_cue_recall: "
                f"{seen['mean_cue_recall'] - heldout['mean_cue_recall']:+.4f}\n"
            )

        f.write("\n## Classifier-Collapse Check on Heldout\n\n")
        f.write(
            "Fraction of heldout readout answers that match any train-class "
            "name/alias. High values mean the model keeps emitting seen-class "
            "names on unseen diagnoses.\n\n"
        )
        n = len(heldout_rows)
        f.write(f"- heldout_rows: {n}\n")
        f.write(f"- answer_in_train_vocab: {len(in_vocab)}/{n}\n")
        f.write(f"- answer_in_train_vocab_rate: {len(in_vocab) / n if n else 0:.4f}\n\n")
        cross = Counter(
            (bool(row.get("answer_hit")), bool(row["answer_in_train_vocab"]))
            for row in heldout_rows
        )
        f.write("| answer_hit | answer_in_train_vocab | n |\n")
        f.write("|---:|---:|---:|\n")
        for hit, vocab_hit in ((False, False), (False, True), (True, False), (True, True)):
            f.write(f"| {hit} | {vocab_hit} | {cross[(hit, vocab_hit)]} |\n")
        f.write(
            "\nRows with answer_hit and answer_in_train_vocab both true usually "
            "indicate name/alias overlap between a heldout and a train diagnosis; "
            "inspect them before counting either way.\n"
        )

        f.write("\n## Interpretation\n\n")
        f.write(f"- answer_hit_high_threshold: {answer_hit_high}\n")
        f.write(f"- cue_recall_high_threshold: {cue_recall_high}\n")
        f.write(
            f"- verdict (heuristic): "
            + interpretation(
                heldout, answer_hit_high=answer_hit_high, cue_recall_high=cue_recall_high
            )
            + "\n"
        )

        f.write("\n## By Heldout Diagnosis\n\n")
        f.write(
            "| diagnosis_id | n | answer_hit_rate | mean_cue_recall "
            "| in_train_vocab_rate | top readout answers |\n"
        )
        f.write("|---|---:|---:|---:|---:|---|\n")
        for diagnosis_id in sorted(set(heldout_diagnoses) | set(by_dx)):
            items = by_dx.get(diagnosis_id, [])
            if not items:
                f.write(f"| {diagnosis_id} | 0 | - | - | - | - |\n")
                continue
            hits = sum(bool(row.get("answer_hit")) for row in items)
            recalls = [
                float(row["cue_recall"]) for row in items if row.get("cue_recall") is not None
            ]
            vocab_rate = sum(bool(row["answer_in_train_vocab"]) for row in items) / len(items)
            top_answers = Counter(
                " ".join(str(row.get("answer_readout") or "-").split()) for row in items
            ).most_common(3)
            answers_text = "; ".join(f"{answer} ({count})" for answer, count in top_answers)
            if len(answers_text) > 160:
                answers_text = answers_text[:157] + "..."
            f.write(
                f"| {diagnosis_id} | {len(items)} | {hits / len(items):.4f} | "
                f"{mean(recalls) if recalls else 0.0:.4f} | {vocab_rate:.4f} | {answers_text} |\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heldout-scored", required=True)
    parser.add_argument("--seen-scored", default=None)
    parser.add_argument(
        "--split-dir",
        required=True,
        help="Diagnosis-heldout split directory containing metadata.json and sft_train.jsonl.",
    )
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-md", required=True)
    parser.add_argument("--answer-hit-high", type=float, default=0.5)
    parser.add_argument("--cue-recall-high", type=float, default=0.5)
    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    metadata = json.loads((split_dir / "metadata.json").read_text(encoding="utf-8"))
    heldout_diagnoses = [str(dx) for dx in metadata["heldout_diagnoses"]]
    vocab = train_vocabulary(split_dir / "sft_train.jsonl")

    heldout_rows = []
    for row in read_jsonl(args.heldout_scored):
        answer = str(row.get("answer_readout") or "")
        matches = train_vocab_matches(answer, vocab)
        heldout_rows.append(
            {
                **row,
                "train_vocab_matches": matches,
                "answer_in_train_vocab": bool(matches),
            }
        )
    if not heldout_rows:
        raise ValueError("No heldout scored rows.")
    seen_rows = list(read_jsonl(args.seen_scored)) if args.seen_scored else []

    in_vocab_rate = sum(bool(r["answer_in_train_vocab"]) for r in heldout_rows) / len(heldout_rows)
    heldout_metrics = pool_metrics(heldout_rows)
    seen_metrics = pool_metrics(seen_rows) if seen_rows else None
    write_jsonl(Path(args.output_jsonl), heldout_rows)
    write_summary(
        Path(args.summary_md),
        seen=seen_metrics,
        heldout=heldout_metrics,
        heldout_rows=heldout_rows,
        heldout_diagnoses=heldout_diagnoses,
        answer_hit_high=args.answer_hit_high,
        cue_recall_high=args.cue_recall_high,
    )
    print(f"[done] wrote {len(heldout_rows)} heldout rows to {args.output_jsonl}")
    print(f"[done] wrote summary to {args.summary_md}")
    print(
        f"[done] heldout answer_hit_rate={heldout_metrics['answer_hit_rate']:.4f} "
        f"mean_cue_recall={heldout_metrics['mean_cue_recall']:.4f} "
        f"in_train_vocab_rate={in_vocab_rate:.4f}"
    )


if __name__ == "__main__":
    main()
