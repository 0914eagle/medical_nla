"""SFT splits for an MCR conclusion readout, in the shape the trainer expects.

The v2 adapter reads the answer position of a DDXPlus prompt and writes the
diagnosis it finds there plus the findings it rests on. MCR needs its own,
because DDXPlus taught the adapter a 49-name vocabulary and bullet-list prose,
and a case report is neither: 6,934 distinct diagnoses, most appearing once,
written as published clinical narrative. Reading MCR internals in words is the
one thing the existing adapter cannot do, and it is the column the open-
vocabulary argument needs filled.

Nothing here requires a judge. The target is assembled from fields the corpus
already carries -- `diagnosis_name` for the conclusion, `cue_targets` for the
grounds -- exactly as the DDXPlus targets were, so training is rule-built end
to end. Judging enters only later, and optionally, when scoring the readouts.

**The split respects the corpus's own train/test division.** MCR's split
exists to fine-tune a reasoner and nothing in the behavioural experiments
trains on it, so those were free to pool both halves. An adapter is different:
it trains, so it trains on MCR's train split and is read out on test. Cases
whose presentation names its own diagnosis are dropped, since a readout that
recites a name printed in the prompt has demonstrated nothing.

**Training rows are restricted to cases the source model answers correctly,
and this build omitted that.** It is the rule the DDXPlus pipeline is built on
-- `make_medical_nla_v2_source_aligned_splits.py` exists for exactly this --
and the reason is that nothing here knows what an activation contains. The
only handle is the model's own output, so `gold` may be used as the target
only where the model reached it. Everywhere else, training a readout to emit
the gold teaches it to guess the right answer from context, which is the
failure the readout is supposed to detect.

DDXPlus survived the omission because it is a 49-class task the model mostly
gets right, so gold and conclusion coincide in most rows. MCR is 6,934
diagnoses over published case reports, and the same missing filter is a
different experiment: the first conclusion adapter was trained on states that
had concluded something else, and read out as a diagnosis in the right
specialty on grounds belonging to no patient. Pass `--answers` and the split
keeps only the rows where the gold is what the state actually reached.

The row schema is copied from an existing split rather than declared, because
the trainer reads keys this script does not own. Pass `--template` at any
`sft_train.jsonl` the trainer already accepts; every key present there is
emitted, filled where this builder knows the value and carried from the
manifest otherwise.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.make_hint_injection_cases import gold_is_written_in, presentation_of
from src.jsonl import read_jsonl, write_jsonl

TARGET_STYLE = "structured_readout_v2_conclusion_mcr"


def target_text(diagnosis: str, cues: list[str]) -> str:
    """The v2 schema: what this state concluded, and what it rests on."""
    grounds = "; ".join(c for c in (str(c).strip() for c in cues) if c)
    return (
        "<readout>\n"
        "  <task_type>diagnosis</task_type>\n"
        f"  <answer>{diagnosis}</answer>\n"
        f"  <supporting_cues>{grounds}</supporting_cues>\n"
        "</readout>"
    )


def split_of(base_id: str, corpus_split: str, val_share: float, seed: int) -> str:
    """train / val / test, with val carved out of the corpus's own train half.

    Keyed on base_id rather than sampled positionally, so re-running with the
    same seed reproduces the assignment and a case never drifts across the
    boundary between one build and the next.
    """
    if corpus_split != "train":
        return "test"
    return "val" if random.Random(f"{seed}:{base_id}").random() < val_share else "train"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", required=True,
                        help="MCR case files; each row needs a corpus_split field "
                        "or --split-name must say which half the file is.")
    parser.add_argument("--split-name", nargs="+", default=None,
                        help="One of train/test per --cases file, when the rows "
                        "do not carry the split themselves.")
    parser.add_argument("--manifest", nargs="+", required=True,
                        help="Answer-position extraction manifest(s).")
    parser.add_argument("--template", required=True,
                        help="An sft_train.jsonl the trainer already accepts; "
                        "its key set defines the output schema.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--val-share", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--answers", nargs="+", default=None,
        help="MCR direct source answers. Rows the model got wrong are dropped "
        "from train and val: the target is the gold diagnosis, and a state "
        "that concluded something else does not contain it.",
    )
    parser.add_argument(
        "--keep-source-wrong-in-test",
        action=argparse.BooleanOptionalAction, default=True,
        help="Keep source-wrong cases in the test split. They are the rows "
        "where readout and output disagree, which is the measurement the "
        "paper rests on -- they must not be trained on and must not be lost.",
    )
    args = parser.parse_args()

    correct: set[str] | None = None
    if args.answers:
        correct = {
            str(row.get("base_id") or row.get("id") or "")
            for path in args.answers
            for row in read_jsonl(path)
            if row.get("source_correct")
        } - {""}
        print(f"source-correct cases: {len(correct):,}")
    else:
        print("⚠ no --answers: train rows will include states that concluded "
              "something other than the gold, which teaches the readout to "
              "guess rather than read. This is how the first MCR conclusion "
              "adapter was built.")

    template_keys = list(next(iter(read_jsonl(args.template))).keys())
    print(f"schema from template: {template_keys}")

    activation: dict[str, dict[str, Any]] = {}
    for path in args.manifest:
        for row in read_jsonl(path):
            base_id = str(row.get("base_id") or "")
            if base_id and row.get("activation_path"):
                activation[base_id] = row
    print(f"activations: {len(activation):,}")

    if args.split_name and len(args.split_name) != len(args.cases):
        raise SystemExit("--split-name needs one entry per --cases file")

    rows: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    dropped = {"no activation": 0, "no diagnosis or cues": 0, "gold printed in prompt": 0,
               "source-wrong (train/val only)": 0}
    for index, path in enumerate(args.cases):
        declared = args.split_name[index] if args.split_name else None
        for case in read_jsonl(path):
            base_id = str(case.get("base_id") or case.get("id") or "")
            act = activation.get(base_id)
            if act is None:
                dropped["no activation"] += 1
                continue
            diagnosis = str(case.get("diagnosis_name") or "").strip()
            cues = [str(c) for c in (case.get("cue_targets") or []) if str(c).strip()]
            if not diagnosis or not cues:
                dropped["no diagnosis or cues"] += 1
                continue
            prompt = str(case.get("prompt") or "")
            presentation = presentation_of(prompt) or prompt
            if gold_is_written_in(presentation, case):
                dropped["gold printed in prompt"] += 1
                continue

            corpus_split = declared or str(case.get("corpus_split") or "train")
            split = split_of(base_id, corpus_split, args.val_share, args.seed)
            # Source-wrong rows never train. They stay in test, where the
            # disagreement between what the state holds and what the model
            # said is the measurement rather than a defect.
            if correct is not None and base_id not in correct:
                if split != "test" or not args.keep_source_wrong_in_test:
                    dropped["source-wrong (train/val only)"] += 1
                    continue
            built = {key: act.get(key, case.get(key)) for key in template_keys}
            built.update(
                {
                    "base_id": base_id,
                    "activation_path": act.get("activation_path"),
                    "target_text": target_text(diagnosis, cues),
                    "target_style": TARGET_STYLE,
                }
            )
            built["source_correct"] = correct is None or base_id in correct
            rows[split].append(built)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, split_rows in rows.items():
        if not split_rows:
            print(f"⚠ {name}: empty")
            continue
        write_jsonl(out / f"sft_{name}.jsonl", split_rows)
        n_ok = sum(1 for r in split_rows if r.get("source_correct"))
        print(f"  sft_{name}.jsonl  {len(split_rows):,} rows"
              f"   source-correct {n_ok:,} ({n_ok / len(split_rows):.3f})")
    for reason, count in dropped.items():
        if count:
            print(f"dropped {count:,}: {reason}")

    missing = [k for k in template_keys if any(r.get(k) is None for r in rows["train"][:50])]
    if missing:
        print(f"\n⚠ keys the template has but this builder could not fill: {missing}")
        print("  Check them before training -- a null the trainer reads is a "
              "silent difference from the DDXPlus run, not an error it raises.")


if __name__ == "__main__":
    main()
