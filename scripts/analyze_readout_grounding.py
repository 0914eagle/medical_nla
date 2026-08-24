"""Are a readout's supporting cues taken from this case, or written from scratch?

Reading six MCR readouts by eye showed the same shape every time: the <answer>
lands in the right clinical neighbourhood (a knee case reads as the same joint,
an oral nodule as an oral nodule), while <supporting_cues> reads like a generic
case-report workup for that specialty -- right register, wrong patient. Ages
disagree with the prompt, laterality flips, lab values appear that the prompt
never gave, and in one Guillain-Barre case the cue asserts *normal* CSF where
the prompt reports protein 76 mg/dL, which is the textbook finding inverted.

An impression is not a result, and a description rate cannot separate the two
failures: a readout that names the diagnosis on fabricated grounds scores the
same as one that reads the state. So this measures grounding directly.

**The control is the whole point.** Cue-to-own-prompt overlap alone is not
interpretable, because clinical prose shares boilerplate -- "laboratory studies
were within normal limits" overlaps every chart ever written. Each cue is
therefore scored twice: against its own prompt, and against another row's
prompt drawn by a fixed permutation. The gap between the two is the only part
that is case-specific information. If the gap is ~0, the cues carry nothing
about this activation no matter how medical they sound.

Boilerplate is counted separately: a cue sentence emitted verbatim across many
rows is prompt-independent by construction, and its recurrence is a rate we can
report rather than an anecdote about seeing the same sentence twice.

Overlap is measured by word trigram containment rather than string matching,
because a faithful cue is usually a paraphrase -- "bilateral alveolar
infiltrates" written back as "bilateral pulmonary edema" is a real read, and an
exact-match scorer would call it a fabrication.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.jsonl import read_jsonl

CUES = re.compile(r"<supporting_cues>(.*?)</supporting_cues>", re.DOTALL)
ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
WORD = re.compile(r"[a-z0-9]+")

# Grounded at half its trigrams: a cue that shares most of its phrasing with
# the chart is a read, one that shares a fragment is a coincidence of register.
GROUNDED_AT = 0.5


def words(text: str) -> list[str]:
    return WORD.findall((text or "").lower())


def trigrams(tokens: list[str]) -> set[tuple[str, ...]]:
    return {tuple(tokens[i : i + 3]) for i in range(len(tokens) - 2)}


def containment(cue: str, haystack_trigrams: set[tuple[str, ...]]) -> float | None:
    """Share of the cue's trigrams that occur in the haystack.

    None for a cue too short to have trigrams -- scoring those as 0 would
    punish terse cues and as 1 would reward them, and neither is measured.
    """
    grams = trigrams(words(cue))
    if not grams:
        return None
    return len(grams & haystack_trigrams) / len(grams)


def split_cues(text: str) -> list[str]:
    match = CUES.search(text or "")
    if not match:
        return []
    return [part.strip() for part in match.group(1).split(";") if part.strip()]


def answer_of(text: str) -> str:
    match = ANSWER.search(text or "")
    return match.group(1).strip() if match else ""


def report(name: str, rows: list[dict]) -> None:
    prompts = [str(r.get("prompt") or "") for r in rows]
    grams = [trigrams(words(p)) for p in prompts]
    n = len(rows)
    # A fixed derangement, so "another case" is deterministic and no row is
    # ever compared against itself.
    other = [(i + 1) % n for i in range(n)]

    own_scores: list[float] = []
    other_scores: list[float] = []
    grounded = other_grounded = 0
    seen: Counter[str] = Counter()
    empty = 0

    for i, row in enumerate(rows):
        cues = split_cues(str(row.get("nla_output") or ""))
        if not cues:
            empty += 1
            continue
        for cue in cues:
            seen[" ".join(words(cue))] += 1
            own = containment(cue, grams[i])
            alt = containment(cue, grams[other[i]])
            if own is None or alt is None:
                continue
            own_scores.append(own)
            other_scores.append(alt)
            grounded += own >= GROUNDED_AT
            other_grounded += alt >= GROUNDED_AT

    if not own_scores:
        print(f"{name}: no scorable cues in {n:,} rows")
        return

    m_own = sum(own_scores) / len(own_scores)
    m_alt = sum(other_scores) / len(other_scores)
    total = len(own_scores)
    repeated = sum(c for cue, c in seen.items() if c > 1)
    print(f"\n=== {name} ===")
    print(f"rows {n:,}   rows with no cue block {empty:,}   cues scored {total:,}")
    print(f"  trigram containment, own prompt        {m_own:.3f}")
    print(f"  trigram containment, another prompt    {m_alt:.3f}   (control)")
    print(f"  case-specific gap                      {m_own - m_alt:+.3f}")
    print(f"  cues grounded (>= {GROUNDED_AT:.0%} trigrams)      "
          f"{grounded:,}/{total:,} = {grounded / total:.3f}")
    print(f"  same, against another prompt           "
          f"{other_grounded:,}/{total:,} = {other_grounded / total:.3f}   (control)")
    print(f"  cue sentences emitted more than once   "
          f"{repeated:,}/{total:,} = {repeated / max(total, 1):.3f}")
    for cue, count in seen.most_common(5):
        if count > 1:
            print(f"      x{count:<4} {cue[:88]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--readouts", nargs="+", required=True,
                        help="run_nla output(s); each file reported separately.")
    args = parser.parse_args()

    for path in args.readouts:
        rows = [r for r in read_jsonl(path) if r.get("prompt")]
        if len(rows) < 2:
            print(f"{path}: needs at least two rows to build the control")
            continue
        report(Path(path).name, rows)

    print("\nThe control column is the reading. A gap near zero means the cues "
          "are medical prose in the right register carrying no information "
          "about this case's activation -- which a description rate would "
          "score as a success.")


if __name__ == "__main__":
    main()
