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

import re
import unicodedata

# MedCaseReasoning's `final_diagnosis` is not consistently a diagnosis *name*:
# alongside "Posterior ischemic optic neuropathy" it carries identifiers --
# "Scleroderma_renal_crisis", "AmeloblasticFibroma", "PFAPAsyndrome". Left as
# they are, a model answering "scleroderma renal crisis" scores zero, and, worse,
# the filter that drops presentations naming their own diagnosis looks for the
# identifier in the prose and never finds it.
_UNDERSCORE = re.compile(r"[_]+")
# "AmeloblasticFibroma" -> "Ameloblastic Fibroma".
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
# "PFAPAsyndrome" -> "PFAPA syndrome": an acronym run running into a word. Two
# preceding capitals are required so ordinary "Fibroma" is left alone, and four
# following lowercase letters so "PEComa" is not cut into "PEC oma".
_ACRONYM_BOUNDARY = re.compile(r"(?<=[A-Z]{2})(?=[a-z]{4,})")

# "Eagle's syndrome" / "Eagle syndrome". Both glyphs, because NFKD leaves the
# typographic apostrophe alone -- it is a distinct character, not an accent.
_POSSESSIVE = re.compile("[\u2019']s\\b")

# British against American spelling is orthography, not disagreement, and it
# separates a published case report ("pulmonary oedema", "anaemia") from what a
# US-trained model writes. Spelled out rather than derived: folding "ae" and
# "oe" wholesale would also rewrite "aerosol" and "coeliac trunk".
_SPELLING_VARIANTS = (
    ("oedema", "edema"),
    ("anaemia", "anemia"),
    ("ischaemi", "ischemi"),
    ("haemo", "hemo"),
    ("haemat", "hemat"),
    ("haemorrhag", "hemorrhag"),
    ("oesophag", "esophag"),
    ("diarrhoea", "diarrhea"),
    ("paediatric", "pediatric"),
    ("orthopaedic", "orthopedic"),
    ("gynaecolog", "gynecolog"),
    ("leukaemia", "leukemia"),
    ("oestrogen", "estrogen"),
    ("caecum", "cecum"),
    ("coeliac", "celiac"),
    ("tumour", "tumor"),
    ("aetiolog", "etiolog"),
)

# Words that carry no diagnostic content on their own. Deliberately short:
# "acute", "chronic", "primary" and "left" all change what is being named, so
# dropping them to raise a match rate would be scoring a different question.
FILLER_WORDS = frozenset({"a", "an", "the", "of", "with", "and", "to", "due", "in", "on"})


def readable_diagnosis_label(label: str) -> str:
    """Recover the diagnosis name from a label that may be an identifier.

    Splitting is per whitespace token, so a label that is already prose --
    "SBP-101 induced retinal toxicity", "Guillain-Barre syndrome" -- passes
    through untouched. The transformation is heuristic and reversible only by
    inspection, so callers keep the raw label as an accepted alias and the
    generator dumps every label it changed.
    """
    words: list[str] = []
    for token in _UNDERSCORE.sub(" ", str(label or "")).split():
        token = _ACRONYM_BOUNDARY.sub(" ", _CAMEL_BOUNDARY.sub(" ", token))
        words.extend(token.split())
    return " ".join(words)


def normalize(text: str) -> str:
    """Fold everything that is presentation rather than content.

    Four kinds of difference all cost matches and none of them is a
    disagreement about the diagnosis: the markup a chat model writes
    ("**Erythema Multiforme**"), the typography of a published label
    ("Whipple's disease", "vitamin D-dependent" with an en dash), and accents
    ("Guillain-Barre" against "Guillain-Barré"). Accents are folded by
    decomposing and dropping the combining marks; everything that is not a
    letter or a digit becomes a space. The fourth is British against American
    spelling, which separates a published case report from a US-trained model.
    """
    decomposed = unicodedata.normalize("NFKD", str(text or ""))
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    # The possessive on an eponym is a house style, not a claim: "Eagle's
    # syndrome" and "Eagle syndrome" are the same name and both are in use.
    # Dropped before punctuation becomes whitespace, or it would survive as a
    # stray "s" token and still block the match.
    depossessed = _POSSESSIVE.sub("", stripped.lower())
    for british, american in _SPELLING_VARIANTS:
        depossessed = depossessed.replace(british, american)
    return " ".join(re.sub(r"[^a-z0-9]+", " ", depossessed).split())


def content_tokens(text: str) -> set[str]:
    return {word for word in normalize(text).split() if word not in FILLER_WORDS}


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
