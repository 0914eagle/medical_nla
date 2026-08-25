"""The probe rebuttal, run rather than argued: same activations, cheaper reader.

Table 4's winning signal reads the final-token state with the NLA verbalizer
and flags cases whose spoken answer fails to contain the internal conclusion.
The obvious objection is that a 49-way linear probe on the *same* vectors
could produce the same flag without any verbalizer. This script builds that
probe and runs it through the identical harness, so the objection gets a
number instead of a hand-wave.

Pre-registered readings (written before the run):
- If probe disagreement lands near the readout's 0.84, the *flag* is not
  unique to verbalization and the paper's uniqueness claim rests on content:
  open-vocabulary description, encoded findings, and the r5 ladder margin.
- If it falls short, the readout wins on numbers too.

Leakage discipline: the probe is trained on the no-note arm of *other*
cases only (two folds, split within diagnosis), so it never sees the
patient it is asked to judge, in either arm. Labels are the gold diagnosis,
which on these direct-correct cases equals the model's own no-note answer --
training the probe to decode "what the model concluded", the same quantity
the readout reports. The wrong-note arm activation it scores is the very
tensor the readout read.

Signals, all single-run (activation + spoken answer, nothing counterfactual):
- containment flag: probe's argmax diagnosis not contained in the answer
  (aliases honored) -- the exact analogue of the readout flag;
- probability form: 1 - p(class matching the answer), the graded version;
- confidence forms: 1 - max prob, and entropy -- "the state looks unsure"
  without naming any class.
"""

from __future__ import annotations

import argparse
import sys
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_hint_effect import (
    Case,
    group_by_case,
    require_canonical_no_note_correct,
)
from scripts.compare_channels_on_attribution import moved, report
from src.answer_matching import is_correct
from src.ddxplus_aliases import aliases_for
from src.jsonl import read_jsonl, write_jsonl


def fold_of(base_id: str) -> int:
    """Deterministic 2-fold split; crc32 so a rerun assigns identically."""
    return zlib.crc32(str(base_id).encode("utf-8")) % 2


def final_activation_paths(manifests: list[str]) -> dict[tuple[str, str], str]:
    """(base_id, hint_variant) -> activation tensor path, final position only."""
    out: dict[tuple[str, str], str] = {}
    for manifest in manifests:
        for row in read_jsonl(manifest):
            if str(row.get("target_role") or "") != "final":
                continue
            base_id = str(row.get("base_id") or "")
            variant = str(row.get("hint_variant") or "")
            path = str(row.get("activation_path") or "")
            if base_id and variant and path:
                out[(base_id, variant)] = path
    return out


def class_of_answer(answer: str, class_names: list[str]) -> int | None:
    """The probe class the spoken answer names, via the scoring matcher.

    None when the answer matches no class -- a free-text answer outside the
    49 labels. For the probability signal that reads as p=0: the probe puts
    no mass on what the model said, maximal disagreement.
    """
    for index, name in enumerate(class_names):
        if is_correct(answer, name, aliases_for(name)):
            return index
    return None


def train_probe(train_x: "Any", train_y: "Any", n_classes: int, seed: int) -> "Any":
    import torch

    torch.manual_seed(seed)
    probe = torch.nn.Linear(train_x.shape[1], n_classes)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(300):
        optimizer.zero_grad()
        loss = loss_fn(probe(train_x), train_y)
        loss.backward()
        optimizer.step()
    return probe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--answers", nargs="+", required=True, help="Direct answers on the hint cases."
    )
    parser.add_argument("--cases", help="Hint case file, for runs predating carried arms.")
    parser.add_argument(
        "--manifests",
        nargs="+",
        required=True,
        help="Extraction manifest.jsonl files carrying final-position rows "
        "with activation_path (layer32/…/manifest.jsonl).",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--require-canonical-no-note-correct",
        action="store_true",
        help="Restrict the population before fitting the cross-fitted probe to "
        "cases whose no-note answer is canonically correct.",
    )
    parser.add_argument(
        "--dump",
        help="Write per-case probe verdicts (base_id, flag, argmax) to this "
        "jsonl, for joining with the correction-ladder results.",
    )
    args = parser.parse_args()

    import torch

    cases = group_by_case(args.answers, args.cases)
    if args.require_canonical_no_note_correct:
        before = len(cases)
        cases = require_canonical_no_note_correct(cases)
        print(f"[cohort] canonical no-note eligible: {len(cases):,}/{before:,}")
        if not cases:
            raise SystemExit("no canonically correct no-note cases remain")
    paths = final_activation_paths(args.manifests)

    class_names = sorted(
        {str(c["wrong"].get("diagnosis_name") or "") for c in cases.values()} - {""}
    )
    class_index = {name: i for i, name in enumerate(class_names)}
    print(f"cases {len(cases):,}   classes {len(class_names)}")

    usable: list[tuple[str, Case]] = []
    missing = 0
    for base_id, case in cases.items():
        if ("wrong" not in case) or (base_id, "none") not in paths or (
            base_id,
            "wrong",
        ) not in paths:
            missing += 1
            continue
        usable.append((base_id, case))
    if missing:
        print(f"[!] {missing:,} cases dropped: no wrong arm or no final activation")
    if not usable:
        raise SystemExit("no usable cases; check --manifests point at final rows")

    def load(base_id: str, variant: str) -> torch.Tensor:
        return torch.load(
            paths[(base_id, variant)], map_location="cpu", weights_only=True
        ).float().flatten()

    # Cross-fit: each fold is judged by a probe trained on the other fold's
    # no-note activations, so no probe ever saw its test patient.
    probs = torch.zeros(len(usable), len(class_names))
    for fold in (0, 1):
        train = [
            (bid, case) for bid, case in usable if fold_of(bid) != fold
        ]
        train_x = torch.stack([load(bid, "none") for bid, _ in train])
        mean, std = train_x.mean(0), train_x.std(0).clamp_min(1e-6)
        train_x = (train_x - mean) / std
        train_y = torch.tensor(
            [class_index[str(c["wrong"]["diagnosis_name"])] for _, c in train]
        )
        probe = train_probe(train_x, train_y, len(class_names), args.seed + fold)
        with torch.no_grad():
            heldout_none = torch.stack(
                [load(bid, "none") for bid, _ in usable if fold_of(bid) == fold]
            )
            heldout_pred = probe(((heldout_none - mean) / std)).argmax(1)
            heldout_y = torch.tensor(
                [
                    class_index[str(c["wrong"]["diagnosis_name"])]
                    for bid, c in usable
                    if fold_of(bid) == fold
                ]
            )
            print(
                f"fold {fold}: train {len(train):,}   heldout no-note decode "
                f"accuracy {(heldout_pred == heldout_y).float().mean():.4f}"
            )
            for i, (bid, _) in enumerate(usable):
                if fold_of(bid) == fold:
                    x = ((load(bid, "wrong") - mean) / std).unsqueeze(0)
                    probs[i] = torch.softmax(probe(x), dim=1)[0]

    features: dict[str, list[float]] = defaultdict(list)
    labels: list[bool] = []
    diagnosis: list[str] = []
    silent: list[bool] = []
    argmax_correct: list[bool] = []
    flags: list[bool] = []
    dump_rows: list[dict[str, Any]] = []
    for i, (base_id, case) in enumerate(usable):
        wrong = case["wrong"]
        answer = str(wrong.get("answer") or "")
        hint = str(wrong.get("hint_diagnosis_name") or "")
        gold = str(wrong.get("diagnosis_name") or "")
        argmax = int(probs[i].argmax())
        argmax_name = class_names[argmax]
        answer_class = class_of_answer(answer, class_names)

        flag = not is_correct(answer, argmax_name, aliases_for(argmax_name))
        entropy = float(-(probs[i] * probs[i].clamp_min(1e-12).log()).sum())
        features["probe disagreement (containment flag)"].append(float(flag))
        features["probe disagreement (1 - p of answer's class)"].append(
            1.0 - (float(probs[i][answer_class]) if answer_class is not None else 0.0)
        )
        features["probe confidence (1 - max prob)"].append(1.0 - float(probs[i].max()))
        features["probe entropy"].append(entropy)
        labels.append(moved(case))
        diagnosis.append(gold)
        silent.append(not is_correct(answer, hint, aliases_for(hint)))
        argmax_correct.append(is_correct(argmax_name, gold, aliases_for(gold)))
        flags.append(flag)
        dump_rows.append(
            {
                "base_id": base_id,
                "diagnosis_name": gold,
                "moved": bool(labels[-1]),
                "answer_is_suggestion": not silent[-1],
                "probe_flag": flag,
                "probe_argmax": argmax_name,
                "probe_p_answer": float(probs[i][answer_class])
                if answer_class is not None
                else 0.0,
                "probe_disagreement_probability": features[
                    "probe disagreement (1 - p of answer's class)"
                ][-1],
                "probe_low_confidence": features[
                    "probe confidence (1 - max prob)"
                ][-1],
                "probe_entropy": entropy,
            }
        )

    report("PROBE ON THE WRONG-NOTE ARM (all cases)", dict(features), labels, diagnosis)
    keep = [i for i, s in enumerate(silent) if s]
    if keep and any(labels[i] for i in keep) and not all(labels[i] for i in keep):
        report(
            "SILENT SUBSET (answer does not name the suspicion)",
            {k: [v[i] for i in keep] for k, v in features.items()},
            [labels[i] for i in keep],
            [diagnosis[i] for i in keep],
        )

    n = len(usable)
    n_moved = sum(labels)
    fired = sum(flags)
    hit = sum(1 for f, m in zip(flags, labels, strict=True) if f and m)
    print("\nCONTENT AND FLAG, side by side with the readout:")
    print(f"  probe argmax accuracy, all wrong-arm    {sum(argmax_correct) / n:.4f}")
    moved_idx = [i for i, m in enumerate(labels) if m]
    if moved_idx:
        acc = sum(argmax_correct[i] for i in moved_idx) / len(moved_idx)
        print(f"  probe argmax accuracy, moved            {acc:.4f}   (readout conclusion: 0.5185)")
    print(
        f"  flag fired {fired:,}/{n:,}   recall {hit}/{n_moved} = {hit / n_moved:.4f}"
        f"   precision {hit}/{fired} = {hit / fired:.4f}"
        f"   (readout flag: 758, recall 0.8457, precision 0.3615)"
        if fired and n_moved
        else "  flag never fired or no moved cases"
    )
    if args.dump:
        write_jsonl(Path(args.dump), dump_rows)
        print(f"  wrote {len(dump_rows):,} probe verdicts to {args.dump}")


if __name__ == "__main__":
    main()
