"""Scoring a cue-position readout against the one cue it was asked to read.

The existing v2 scorer asks whether the gold cue appears in the readout as a
literal substring. On diagnosis names that is nearly right. On these cues it is
not: they are rendered questionnaire answers up to twenty-one words long, and
the adapter paraphrases. The first two heldout rows generated were

    gold  they feel that their eyes produce excessive tears
    read  their eyes produce too many tears

    gold  slightly dizzy or lightheaded
    read  they do feel like they are about to faint or are lightheaded

both of which the substring rule scores zero. Reporting a heldout read rate
from that rule would have said the readout carries nothing when the first row
is a correct reading of a cue string that was never supervised.

So overlap is measured over content words, discarding a fixed list of English
function words. Fixed, and not a frequency cutoff over the corpus: the first
attempt here dropped any token appearing in more than a fifth of gold cues,
which is exactly the kind of rule that deletes "pain" from a chest-pain corpus
and reports the readout as content-free. What is safe to write down in advance
is the closed grammatical classes -- pronouns, determiners, prepositions,
conjunctions, and the auxiliaries these questionnaire renderings are framed
with ("they do have ..."). Which *clinical* words matter is never decided here.

None of this is the metric. Overlap is triage: it orders rows so the labelling
that decides the number is spent where the rules disagree, exactly as the
answer scoring was settled.
"""

from __future__ import annotations

import re

from .answer_matching import normalize

# Closed-class English, plus the periphrastic do/have these cues are rendered
# with. No clinical term appears here, and none should: a word is filler
# because of its grammatical class, not because it was common in a sample.
FUNCTION_WORDS = frozenset(
    """
    a an the this that these those
    i me my we us our you your he him his she her it its they them their
    is are was were be been being am
    do does did done have has had having
    of in on at to for with without from by as into over under about
    and or but if then than so because while when
    not no any some all both each
    too very quite slightly more most much many few
    there here what which who whom whose
    like feel feels felt feeling
    """.split()
)


def observed_items(readout_text: str) -> list[str]:
    """The cue lines a readout emitted, one string each.

    Tag extraction collapses newlines, so a bullet surrounded by whitespace
    separates items as well as one at the start of a line. A hyphen inside a
    word is left alone.
    """
    if not readout_text:
        return []
    parts = re.split(r"(?:^|\s)[-•*](?:\s+|$)|\n", readout_text)
    return [item.strip() for item in parts if item and item.strip()]


# Suffixes stripped before comparing, longest first. Without this, "coughing up
# blood" read as "a cough that produced blood" scored 0.29 and "undergo
# dialysis" read as "have to dialyze" scored 0.00 -- both correct readings of a
# cue that was never supervised. Mechanical, and applied to gold and readout
# alike, so it cannot favour either.
_SUFFIXES = ("ations", "ation", "ings", "ing", "ies", "ied", "es", "ed", "s")


def stem(word: str) -> str:
    for suffix in _SUFFIXES:
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            base = word[: -len(suffix)]
            # "ies" -> "y" keeps "arteries"/"artery" together.
            return base + "y" if suffix in ("ies", "ied") else base
    return word


_PARENTHETICAL = re.compile(r"\([^)]*\)")


def strip_parentheticals(text: str) -> str:
    """Drop bracketed text, which on these cues is the questionnaire's own gloss.

    DDXPlus renders a cue with the clarification the survey put in the
    question: "a fever (either felt or measured with a thermometer)". The
    finding is the fever. A readout that says "had a fever (defined as 100F or
    higher)" is exactly right and scores 0.22 against the full string -- eleven
    of the twenty-five lowest-scoring rows in the first heldout sample were
    that one cue.
    """
    return _PARENTHETICAL.sub(" ", text)


def content_words(text: str) -> set[str]:
    return {
        stem(word)
        for word in normalize(text).split()
        if word and word not in FUNCTION_WORDS
    }


def overlap(emitted: str, gold: str) -> tuple[float, float, float]:
    """Precision, recall and F1 over content words.

    All three are reported because the vanilla AV baseline separates them.
    Given the same vector it names the finding correctly and then writes 1,600
    characters about what token might come next -- so its recall is high and
    its precision is near zero, and a single F1 would report it as unable to
    read the cue when what it cannot do is stop. Which of those is true decides
    what the adapter is claimed to add.
    """
    best = (0.0, 0.0, 0.0)
    # Scored twice: as written, and with both sides' bracketed glosses removed.
    # The better of the two is kept, because a readout should not be penalised
    # for declining to repeat the questionnaire's parenthetical, nor for
    # supplying its own -- but a cue whose content sits inside the brackets
    # ("the human immunodeficiency virus (HIV)") must not be gutted either.
    for transform in (lambda text: text, strip_parentheticals):
        got = content_words(transform(emitted))
        want = content_words(transform(gold))
        if not got or not want:
            continue
        shared = len(got & want)
        if not shared:
            continue
        precision = shared / len(got)
        recall = shared / len(want)
        f1 = 2 * precision * recall / (precision + recall)
        if f1 > best[2]:
            best = (precision, recall, f1)
    return best


def overlap_f1(emitted: str, gold: str) -> float:
    """F1 alone, so neither padding nor truncating is rewarded."""
    return overlap(emitted, gold)[2]


def exact_containment(emitted: str, gold: str) -> bool:
    """The rule the v2 scorer used, kept so the two can be compared."""
    got, want = normalize(emitted), normalize(gold)
    if not got or not want:
        return False
    return want in got or got in want


def score_readout(emitted_text: str, gold_cue: str) -> dict[str, object]:
    """Best-matching emitted item against the row's single gold cue.

    Best rather than first: a readout that emits two lines is right if either
    reads the cue, and penalising the second line belongs to precision, which
    is reported separately rather than folded into the read rate.
    """
    items = observed_items(emitted_text) or ([emitted_text] if emitted_text else [])
    if not items:
        return {
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "output_precision": 0.0,
            "output_recall": 0.0,
            "exact": False,
            "best_item": "",
            "n_items": 0,
            "n_chars": 0,
        }
    whole = overlap(emitted_text, gold_cue)
    scored = [(overlap(item, gold_cue), item) for item in items]
    (best_p, best_r, best_f1), best_item = max(scored, key=lambda pair: pair[0][2])
    return {
        "f1": best_f1,
        "precision": best_p,
        "recall": best_r,
        # Over the whole output rather than its best line. The best line asks
        # "is there a usable sentence in here", which a rambling baseline
        # passes; these ask "did it read the finding" and "is the output about
        # the finding", and it passes the first and fails the second.
        "output_precision": whole[0],
        "output_recall": whole[1],
        "exact": any(exact_containment(item, gold_cue) for item in items),
        "best_item": best_item,
        "n_items": len(items),
        "n_chars": len(emitted_text),
    }
