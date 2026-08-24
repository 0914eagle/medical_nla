"""Score the held-out cue readouts by an external judge, beside the hand pass.

Table 1's semantic row carries three numbers -- .340 / .731 / .557 for L16, L24
and v4 -- and a blank where the scorer should be named. The blank is not an
oversight: those numbers came from a hand pass over pair-level A/B/C/D labels,
row-weighted, counting A+B. A hand pass by the same people who want the number
to be high is exactly what a reviewer will ask about, so the cell stayed empty
until an external judge could stand in it.

This reads the external judge's verdicts and recomputes the identical
statistic, then reports how far the two scorers are apart. Both are printed.
Replacing one number with another silently would throw away the only thing that
makes either credible -- that two independent scorers looked at the same 238
pairs and can be compared.

Pairs, not rows. DDXPlus renders cues from a fixed questionnaire, so 438 rows
collapse to 92 / 72 / 74 distinct (gold, readout) pairs. The judge sees each
pair once; the rate is then row-weighted back up, which is what the hand pass
did and what the .340 / .731 / .557 reproduce exactly.

    python scripts/analyze_readout_semantic_judgements.py \
        --index   $DATA/judge_readout_semantic/L24_v5/judge_index.jsonl \
        --judged  $ART/results/judge_readout_semantic_L24_v5.jsonl \
        --hand    results_snapshot/L24_v5_heldout_pairs_hand_labeled.jsonl \
        --label   L24_v5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.jsonl import read_jsonl

GRADES = ("A", "B", "C", "D")


def norm(text: Any) -> str:
    """Collapse whitespace and drop a leading bullet.

    The hand file kept the '- ' the model emitted; the request builder strips
    it. Joining on raw text silently matches nothing, and a zero-row join looks
    like a judge that answered nothing rather than a punctuation mismatch.
    """
    return re.sub(r"^[-*•]\s*", "", " ".join(str(text).split()))


def grade_of(response: Any) -> str:
    """First A/B/C/D character in the reply, or '' when the judge gave none."""
    return next((c for c in str(response or "").strip().upper() if c in GRADES), "")


# Words that name WHERE a finding is, rather than what it is. DDXPlus renders
# these from the questionnaire, so a pair differing only in one of them is the
# same question asked about a different part of the body.
SITE_WORDS = {
    # which side
    "l", "r", "left", "right", "bilateral", "unilateral",
    # which end
    "upper", "lower", "top", "bottom", "middle",
    "dorsal", "ventral", "lateral", "medial", "anterior", "posterior",
    "proximal", "distal", "superior", "inferior", "front", "back",
    # the locator nouns these attach to. 'dorsal aspect of the foot' against
    # 'lateral side of the foot' is the same foot: only the face of it moved,
    # and without these the pair reads as two different findings.
    "aspect", "side", "region", "area", "part", "portion", "surface",
}
_WORD = re.compile(r"[a-z]+")


def differs_only_by_site(gold: str, read: str) -> bool:
    """True when the two texts agree except on words naming a location.

    The rubric puts a wrong attribute at B and a different finding at C, and
    tells the scorer to answer C when both are arguable. Laterality sits
    exactly on that line, and it is the worst place to be generous in a
    medical readout: 'swelling located thigh(L)' against 'thigh(R)' is one
    token and the wrong leg. Counting these separately keeps a scorer's margin
    from being read as agreement about content when it is agreement about
    everything except which side.
    """
    g = _WORD.findall(gold.lower())
    r = _WORD.findall(read.lower())
    gc, rc = Counter(g), Counter(r)
    only_g = gc - rc
    only_r = rc - gc
    if not only_g and not only_r:
        return False  # identical wording; nothing to attribute
    leftover = set(only_g) | set(only_r)
    return bool(leftover) and leftover <= SITE_WORDS


def weighted(counts: Counter[str]) -> tuple[float, int]:
    total = sum(counts.values())
    if not total:
        return float("nan"), 0
    return (counts["A"] + counts["B"]) / total, total


def kappa(pairs: list[tuple[str, str]]) -> float:
    """Cohen's kappa over the four grades."""
    if not pairs:
        return float("nan")
    n = len(pairs)
    observed = sum(1 for a, b in pairs if a == b) / n
    left = Counter(a for a, _ in pairs)
    right = Counter(b for _, b in pairs)
    expected = sum((left[g] / n) * (right[g] / n) for g in GRADES)
    if expected >= 1.0:
        return float("nan")
    return (observed - expected) / (1 - expected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True,
                        help="judge_index.jsonl from make_readout_judge_requests.")
    parser.add_argument("--judged", help="run_judge.py output for those requests.")
    parser.add_argument("--hand", help="The hand-labeled pair file, for comparison.")
    parser.add_argument("--label", default="", help="Layer name, for the header.")
    parser.add_argument("--show", type=int, default=8,
                        help="Disagreeing pairs to print.")
    args = parser.parse_args()

    index = list(read_jsonl(args.index))
    by_key = {(norm(e["gold"]), norm(e["read"])): e for e in index}
    rows_of = {str(e["n"]): len(e.get("ids") or []) for e in index}
    key_of = {str(e["n"]): (norm(e["gold"]), norm(e["read"])) for e in index}

    hand: dict[tuple[str, str], str] = {}
    if args.hand:
        for row in read_jsonl(args.hand):
            hand[(norm(row.get("gold_cue")), norm(row.get("emitted")))] = str(
                row.get("label") or ""
            ).strip().upper()

    judge: dict[tuple[str, str], str] = {}
    unparsed = 0
    if args.judged:
        for row in read_jsonl(args.judged):
            key = key_of.get(str(row.get("id")))
            if key is None:
                continue
            grade = grade_of(row.get("response"))
            if not grade:
                unparsed += 1
                continue
            judge[key] = grade

    head = f"held-out cue readouts, semantic grading"
    if args.label:
        head += f" -- {args.label}"
    print(head)
    print(f"  pairs in index      {len(index):,}")
    print(f"  rows covered        {sum(rows_of.values()):,}")

    scorers: list[tuple[str, dict[tuple[str, str], str]]] = []
    if hand:
        scorers.append(("hand", hand))
    if judge:
        scorers.append(("judge", judge))
    if not scorers:
        print("\n  nothing to score -- pass --hand and/or --judged.")
        return

    print("\n  scorer   pairs   A     B     C     D     A+B (row-weighted)")
    for name, table in scorers:
        pair_counts: Counter[str] = Counter()
        row_counts: Counter[str] = Counter()
        for key, entry in by_key.items():
            grade = table.get(key)
            if not grade:
                continue
            pair_counts[grade] += 1
            row_counts[grade] += len(entry.get("ids") or [])
        rate, n_rows = weighted(row_counts)
        cells = "  ".join(f"{pair_counts[g]:>3}" for g in GRADES)
        print(f"  {name:<7} {sum(pair_counts.values()):>5}   {cells}     "
              f"{rate:.4f}  (n={n_rows:,})")

    # D is "empty, refused, or no clinical content". A scorer that never uses
    # it is not finding every readout substantive; it is declining to use the
    # floor of the scale, which shifts everything above it.
    for name, table in scorers:
        used = {table[k] for k in by_key if k in table}
        if used and "D" not in used:
            print(f"\n  ⚠ '{name}' never assigned D on any pair. The grade "
                  f"exists for empty or\n    contentless readouts; a scorer "
                  f"that never reaches for it is rating the\n    scale, not "
                  f"only the readouts.")

    if unparsed:
        print(f"\n  unparseable judge replies {unparsed:,} -- dropped, not "
              f"defaulted. A judge that did not answer is missing data, and "
              f"any default moves the rate being measured.")

    missing = [k for k in by_key if k not in judge] if judge else []
    if missing:
        print(f"  pairs the judge has not answered {len(missing):,} -- the "
              f"judge column above is over the answered subset only.")

    if hand and judge:
        both = [(hand[k], judge[k]) for k in by_key if k in hand and k in judge]
        if both:
            exact = sum(1 for a, b in both if a == b) / len(both)
            collapsed = sum(
                1 for a, b in both if (a in "AB") == (b in "AB")
            ) / len(both)
            print(f"\n  agreement on {len(both):,} pairs judged by both")
            print(f"    exact 4-way        {exact:.4f}")
            print(f"    collapsed A+B/C+D  {collapsed:.4f}")
            print(f"    Cohen's kappa      {kappa(both):.4f}")
            print("\n  This comparison is the point. Either scorer alone is a "
                  "claim;\n  the two together bound how much the number depends "
                  "on who scored it.")

        disagreeing = [
            (k, hand[k], judge[k])
            for k in by_key
            if k in hand and k in judge and (hand[k] in "AB") != (judge[k] in "AB")
        ]

        # Where the judge's extra credit lands, in rows rather than pairs.
        # A margin made of laterality is a different claim from a margin made
        # of paraphrase, and the two should not be summarised by one delta.
        promoted = [
            (k, h, j) for k, h, j in disagreeing if j in "AB" and h not in "AB"
        ]
        if promoted:
            site = [k for k, _, _ in promoted if differs_only_by_site(*k)]
            rows_promoted = sum(
                len(by_key[k].get("ids") or []) for k, _, _ in promoted
            )
            rows_site = sum(len(by_key[k].get("ids") or []) for k in site)
            total_rows = sum(len(e.get("ids") or []) for e in index)
            print(f"\n  where the judge is more generous than the hand pass")
            print(f"    pairs promoted into A+B        {len(promoted):,}"
                  f"  ({rows_promoted:,} rows, {rows_promoted / total_rows:.4f})")
            print(f"    of those, differing ONLY by a site/laterality word"
                  f"   {len(site):,}"
                  f"  ({rows_site:,} rows, {rows_site / total_rows:.4f})")
            if rows_promoted and rows_site / rows_promoted >= 0.3:
                print("    ⚠ a large share of the judge's margin is laterality "
                      "and site.\n      The rubric puts a wrong attribute at B "
                      "and says to answer C when B\n      and C are both "
                      "arguable; 'thigh(L)' against 'thigh(R)' is one token "
                      "and\n      the wrong leg. Report the margin with this "
                      "split, not as one number.")

        if disagreeing and args.show:
            print(f"\n  pairs that cross the A+B boundary ({len(disagreeing):,}):")
            for (gold, read), h, j in disagreeing[: args.show]:
                tag = "  [site/laterality only]" if differs_only_by_site(gold, read) else ""
                print(f"    hand {h} / judge {j}{tag}")
                print(f"      GOLD: {gold[:96]}")
                print(f"      READ: {read[:96]}")

    print("\n  Table 1's scorer cell takes the judge row plus the judge's model "
          "id\n  and date; the hand row stays in the audit record, not in the "
          "table.")


if __name__ == "__main__":
    main()
