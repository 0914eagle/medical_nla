"""How many of a case's findings does the chain of thought actually name?

This is the number the whole comparison rests on. Our readout produces, per
case, the set of findings the model took in; the chain produces the set it says
it used. If the chain names every finding in the chart, then "the readout
recovers case-specific detail that the chain omits" is false and the
explanation axis has to be argued from something other than coverage -- from
an extraneous factor the chain has no reason to admit to, which is Turpin's
design rather than ours.

A chain is prose, so a cue counts as named when the chain contains its content
words, not when a line matches it. Long outputs collect content words by
accident, so the same cue set is also scored against cues drawn from other
cases: "names 60% of its own findings" is only meaningful beside "names 20% of
anyone's findings".
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path
from statistics import mean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cue_readout_scoring import content_words, strip_parentheticals
from src.jsonl import read_jsonl

# Share of a cue's content words that must appear anywhere in the chain. Set to
# the readout scorer's threshold, because the point of this number is to sit
# beside that one -- a chain and a readout have to be credited by the same rule
# or their coverage cannot be compared.
#
# It is still a lower bound. "smoke cigarettes" named as "history of smoking"
# and "had surgery within the last month" named as "recent surgery" both fall
# under it, and no threshold on word overlap will catch those; that is what the
# judge in make_readout_judge_requests.py is for, if the number needs to be
# exact rather than indicative.
NAMED = 0.5


_PAREN = re.compile(r"\(([^)]*)\)")
_ALTERNATIVES = re.compile(r"\s+(?:or|and/or)\s+")


def cue_variants(cue: str) -> list[str]:
    """The forms of a cue that a chain naming it might actually use.

    DDXPlus writes questionnaire items, and they carry two things a clinician
    writing prose would not repeat. A gloss -- "a fever (either felt or
    measured with a thermometer)" -- where the finding is the fever. And a list
    of synonyms -- "had chills or shivers", "shortness of breath or difficulty
    breathing in a significant way" -- where naming one is naming the finding.
    The parenthetical is sometimes the whole point, as in "a chronic
    obstructive pulmonary disease (COPD)", so it is kept as a form of its own
    rather than discarded.

    Counting only the full string credited chains with 0.70 of their case's
    findings where reading them showed nearly all named.
    """
    forms = {cue, strip_parentheticals(cue)}
    forms.update(match.group(1) for match in _PAREN.finditer(cue))
    for form in list(forms):
        forms.update(_ALTERNATIVES.split(form))
    return [form.strip() for form in forms if form.strip()]


def mentions(text_words: set[str], cue: str) -> bool:
    for form in cue_variants(cue):
        want = content_words(form)
        if want and len(text_words & want) / len(want) >= NAMED:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", required=True, help="run_source_answers cot output.")
    parser.add_argument("--cases", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--show", type=int, default=2, help="Chains to print.")
    args = parser.parse_args()

    cases: dict[str, dict[str, Any]] = {}
    for case in read_jsonl(args.cases):
        cues = [" ".join(str(c).split()) for c in (case.get("cue_targets") or []) if str(c).strip()]
        for key in (case.get("base_id"), case.get("id")):
            if key:
                cases[str(key)] = {"cues": cues}
    vocabulary = sorted({cue for case in cases.values() for cue in case["cues"]})

    rng = random.Random(args.seed)
    own_rates: list[float] = []
    foreign_rates: list[float] = []
    correct_rates: list[float] = []
    wrong_rates: list[float] = []
    lengths: list[int] = []
    shown = 0

    for row in read_jsonl(args.answers):
        case = cases.get(str(row.get("base_id"))) or cases.get(str(row.get("id")))
        if not case or not case["cues"]:
            continue
        chain = str(row.get("response") or "")
        words = content_words(chain) | content_words(strip_parentheticals(chain))
        lengths.append(len(chain))

        own = [mentions(words, cue) for cue in case["cues"]]
        pool = [cue for cue in vocabulary if cue not in case["cues"]]
        foreign_cues = rng.sample(pool, min(len(case["cues"]), len(pool)))
        foreign = [mentions(words, cue) for cue in foreign_cues]
        rate = sum(own) / len(own)
        own_rates.append(rate)
        foreign_rates.append(sum(foreign) / max(len(foreign), 1))
        (correct_rates if row.get("source_correct") else wrong_rates).append(rate)

        if shown < args.show:
            shown += 1
            named = [c for c, hit in zip(case["cues"], own, strict=True) if hit]
            missed = [c for c, hit in zip(case["cues"], own, strict=True) if not hit]
            print(f"--- {row.get('id')}  gold={row.get('diagnosis_name')} "
                  f"answer={row.get('answer')} ---")
            print(f"  named   ({len(named)}): " + " | ".join(named))
            print(f"  missed  ({len(missed)}): " + " | ".join(missed))
            print(f"  chain   {chain[:500]}\n")

    if not own_rates:
        raise SystemExit("no rows joined to a case; check --cases")

    print(f"chains {len(own_rates):,} | mean length {mean(lengths):.0f} chars")
    print(f"  names its own findings   {mean(own_rates):.4f}")
    print(f"  names anyone's findings  {mean(foreign_rates):.4f}   <- chance")
    print(f"  above chance             {mean(own_rates) - mean(foreign_rates):+.4f}")
    if correct_rates and wrong_rates:
        print(f"\n  when the answer is right {mean(correct_rates):.4f}  (n={len(correct_rates):,})")
        print(f"  when the answer is wrong {mean(wrong_rates):.4f}  (n={len(wrong_rates):,})")
        print(
            "  A chain that cites fewer findings when it is wrong is a chain whose\n"
            "  coverage carries information about the error."
        )
    print(
        "\n  High coverage means the chain repeats the chart, and the explanation\n"
        "  axis cannot be argued from coverage -- it needs a factor the chain has\n"
        "  no reason to admit to. Low coverage means the chain selects, and which\n"
        "  findings it drops becomes the question."
    )


if __name__ == "__main__":
    main()
