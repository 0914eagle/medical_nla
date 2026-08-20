"""Decide whether a free-text diagnosis names the gold condition.

Two corpora, two problems. DDXPlus has 49 fixed pathology names, so a model's
answer either is one of them or is not, and containment settles it. MCR is
free text -- 6,934 distinct labels over 12,766 cases -- where the same
condition is written many ways, and containment is brittle in exactly the
direction that costs accuracy: "phototoxic drug reaction" does not contain
"phototoxic reaction", because one word intervenes.

Both rules are therefore kept and reported side by side. `is_correct` stays
the strict, quotable metric; `token_f1` measures how much it is undercounting,
so a low accuracy can be attributed to the model rather than to the scorer.
Neither is adopted silently: the gap between them is a number the audit prints.
"""

from __future__ import annotations

# Words that carry no diagnostic content on their own. Deliberately short:
# "acute", "chronic", "primary" and "left" all change what is being named, so
# dropping them to raise a match rate would be scoring a different question.
FILLER_WORDS = frozenset({"a", "an", "the", "of", "with", "and", "to", "due", "in", "on"})


def normalize(text: str) -> str:
    return " ".join(str(text or "").split()).lower().strip(" .")


def content_tokens(text: str) -> set[str]:
    return {word for word in normalize(text).replace("-", " ").split() if word not in FILLER_WORDS}


def is_correct(answer: str | None, gold: str, aliases: list[str]) -> bool:
    """Strict rule: the answer contains the gold name, or the gold contains it.

    Containment in either direction, since a model may answer "acute otitis
    media" for "Otitis media" or the reverse; both name the same condition.
    """
    if not answer:
        return False
    got = normalize(answer)
    for candidate in [gold, *aliases]:
        want = normalize(candidate)
        if want and (want in got or got in want):
            return True
    return False


def token_f1(answer: str | None, gold: str, aliases: list[str]) -> float:
    """Best F1 over content words between the answer and any accepted name.

    F1 rather than containment so that neither padding the answer nor
    truncating it is rewarded: "reaction" scores 0.67 against "phototoxic
    reaction", and so does "phototoxic drug allergic reaction".
    """
    if not answer:
        return 0.0
    got = content_tokens(answer)
    if not got:
        return 0.0
    best = 0.0
    for candidate in [gold, *aliases]:
        want = content_tokens(candidate)
        if not want:
            continue
        shared = len(got & want)
        if not shared:
            continue
        precision = shared / len(got)
        recall = shared / len(want)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best
