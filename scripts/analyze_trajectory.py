"""The anchoring trajectory, drawn: where the original answer dies.

Input is the trajectory extraction (make_trajectory_rows -> extract at L32).
At every landmark a linear probe is trained on the *other* fold's cases
(gold labels, split within diagnosis) and applied to this fold, so no probe
ever saw its test patient. Each probe is then read twice on its held-out
cases -- once on the wrong arm, once on the same cases' none arm. That second
read is the control the headline needs: the findings are bit-identical up to
the note, so "the moved cases still carry the gold" is only a claim about
anchoring if the note-free vectors of those same cases carry *more*. The gap
between the two curves is the note's entire effect on the internals, and the
last_cue landmark is its own check -- the arms are the same tensor there, so
the two curves must coincide exactly or the extraction is wrong.

Read out per landmark, moved cases against not-moved:
- p(gold) and p(suggestion): the two masses whose crossing is the flip;
- argmax == gold: how much of the population still holds the original
  answer at this depth;
- and per moved case, the first landmark where argmax lands on the
  suggestion -- the flip-point distribution, "where it universally breaks".

The probes locate the flip; the verbalizer narrates it on sampled cases.
This is the division of labor the probe result forces: numbers from the
classifier, words from the readout.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_hint_effect import group_by_case, took_the_hint
from scripts.compare_channels_on_attribution import moved
from scripts.evaluate_probe_disagreement import fold_of, train_probe
from src.answer_matching import is_correct
from src.ddxplus_aliases import aliases_for
from src.jsonl import read_jsonl

LANDMARK_ORDER = ["last_cue", "note", "question", "constraint", "format", "final"]


def load_paths(manifests: list[str]) -> dict[tuple[str, str, str], str]:
    out: dict[tuple[str, str, str], str] = {}
    for manifest in manifests:
        for row in read_jsonl(manifest):
            key = (
                str(row.get("base_id") or ""),
                str(row.get("hint_variant") or ""),
                str(row.get("target_role") or ""),
            )
            path = str(row.get("activation_path") or "")
            if all(key) and path:
                out[key] = path
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", nargs="+", required=True)
    parser.add_argument("--cases", help="Hint case file, for runs predating carried arms.")
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    import torch

    cases = group_by_case(args.answers, args.cases)
    paths = load_paths(args.manifests)
    class_names = sorted(
        {str(c["wrong"].get("diagnosis_name") or "") for c in cases.values()} - {""}
    )
    class_index = {name: i for i, name in enumerate(class_names)}

    def load(key: tuple[str, str, str]) -> "torch.Tensor":
        return torch.load(paths[key], map_location="cpu", weights_only=True).float().flatten()

    roles = [r for r in LANDMARK_ORDER if any(k[2] == r for k in paths)]
    print(f"cases {len(cases):,}   classes {len(class_names)}   landmarks {roles}")

    # Per moved case, the argmax verdict at each landmark, to find flip points.
    verdicts: dict[str, dict[str, str]] = defaultdict(dict)

    for role in roles:
        # The note exists only in the wrong arm; every shared landmark trains
        # on the none arm so the probe never conditions on a note at all.
        train_variant = "wrong" if role == "note" else "none"
        usable = [
            (bid, case)
            for bid, case in cases.items()
            if "wrong" in case
            and (bid, train_variant, role) in paths
            and (bid, "wrong", role) in paths
        ]
        if len(usable) < 100:
            print(f"\n{role}: only {len(usable)} usable cases, skipped")
            continue

        # moved splits in two: cases pulled ONTO the suggestion should show the
        # suggestion's mass overtaking, while cases that merely lost the gold
        # may show only the gold's collapse -- averaging them smears both.
        stats: dict[str, list[tuple[float, float, bool]]] = {
            "kept": [],
            "moved-onto-hint": [],
            "moved-lost-gold": [],
        }
        # The same probe read on the SAME cases' none-arm vectors. Without it
        # "the moved cases still carry the gold" has a deflationary reading --
        # the findings are bit-identical up to the note, so a probe that can
        # read gold anywhere would read it here too, note or no note. The
        # counterfactual curve is what turns "gold is present" into "the note
        # barely moved the state": the wrong-arm drop from this baseline is
        # the note's whole effect on the internals.
        counterfactual: dict[str, list[tuple[float, float, bool]]] = {
            "kept": [],
            "moved-onto-hint": [],
            "moved-lost-gold": [],
        }
        heldout_hits = heldout_n = 0
        for fold in (0, 1):
            train = [(b, c) for b, c in usable if fold_of(b) != fold]
            test = [(b, c) for b, c in usable if fold_of(b) == fold]
            train_x = torch.stack([load((b, train_variant, role)) for b, _ in train])
            mean, std = train_x.mean(0), train_x.std(0).clamp_min(1e-6)
            train_y = torch.tensor(
                [class_index[str(c["wrong"]["diagnosis_name"])] for _, c in train]
            )
            probe = train_probe(
                (train_x - mean) / std, train_y, len(class_names), args.seed + fold
            )
            with torch.no_grad():
                for bid, case in test:
                    x = ((load((bid, "wrong", role)) - mean) / std).unsqueeze(0)
                    p = torch.softmax(probe(x), dim=1)[0]
                    gold = str(case["wrong"]["diagnosis_name"])
                    hint = str(case["wrong"].get("hint_diagnosis_name") or "")
                    argmax_name = class_names[int(p.argmax())]
                    heldout_n += 1
                    heldout_hits += argmax_name == gold if train_variant == "none" else 0
                    hint_mass = max(
                        (
                            float(p[i])
                            for i, n in enumerate(class_names)
                            if hint and is_correct(n, hint, aliases_for(hint))
                        ),
                        default=0.0,
                    )
                    if not moved(case):
                        group = "kept"
                    elif took_the_hint(case, "wrong"):
                        group = "moved-onto-hint"
                    else:
                        group = "moved-lost-gold"
                    stats[group].append(
                        (float(p[class_index[gold]]), hint_mass, argmax_name == gold)
                    )
                    none_key = (bid, "none", role)
                    if none_key in paths:
                        xn = ((load(none_key) - mean) / std).unsqueeze(0)
                        pn = torch.softmax(probe(xn), dim=1)[0]
                        n_argmax = class_names[int(pn.argmax())]
                        n_hint = max(
                            (
                                float(pn[i])
                                for i, n in enumerate(class_names)
                                if hint and is_correct(n, hint, aliases_for(hint))
                            ),
                            default=0.0,
                        )
                        counterfactual[group].append(
                            (float(pn[class_index[gold]]), n_hint, n_argmax == gold)
                        )
                    if moved(case):
                        verdicts[bid][role] = (
                            "gold"
                            if argmax_name == gold
                            else "hint"
                            if hint and is_correct(argmax_name, hint, aliases_for(hint))
                            else "other"
                        )

        print(f"\n{role.upper()}  (n={len(usable):,}"
              + (f", wrong-arm argmax==gold {heldout_hits / heldout_n:.3f})"
                 if train_variant == "none" else ", trained on wrong arm)"))
        for group in ("kept", "moved-onto-hint", "moved-lost-gold"):
            rows = stats[group]
            if not rows:
                continue
            n = len(rows)
            print(
                f"  {group:<16} p(gold) {sum(r[0] for r in rows) / n:.3f}"
                f"   p(suggestion) {sum(r[1] for r in rows) / n:.3f}"
                f"   still holds gold {sum(r[2] for r in rows) / n:.3f}"
            )
            base = counterfactual[group]
            if base:
                m = len(base)
                b_gold = sum(r[0] for r in base) / m
                print(
                    f"  {'  same cases, no note':<16} p(gold) {b_gold:.3f}"
                    f"   p(suggestion) {sum(r[1] for r in base) / m:.3f}"
                    f"   still holds gold {sum(r[2] for r in base) / m:.3f}"
                    f"   [note costs {sum(r[0] for r in rows) / n - b_gold:+.3f}]"
                )

    flips = Counter()
    for bid, by_role in verdicts.items():
        flip = next((r for r in LANDMARK_ORDER if by_role.get(r) == "hint"), None)
        flips[flip or "never"] += 1
    if flips:
        print("\nFLIP POINT (moved cases: first landmark where the probe reads the suggestion):")
        for role in [*LANDMARK_ORDER, "never"]:
            if flips.get(role):
                print(f"  {role:<12} {flips[role]:>5}")


if __name__ == "__main__":
    main()
