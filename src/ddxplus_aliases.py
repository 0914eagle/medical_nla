"""Accepted names for DDXPlus's 49 pathologies.

Open-ended answers are scored by containment against the gold label. That works
because DDXPlus has a closed label set, but the labels are written for a
classifier, not spoken by a clinician: `URTI` and `PSVT` are abbreviations that
share no words with what a model writes out in full, `Larygospasm` is DDXPlus's
own typo, and `Pancreatic neoplasm` is not the phrase anyone reaches for.
Measured on 4,900 answers, `Pancreatic neoplasm` alone lost 30 correct answers
to `Pancreatic Cancer` and `Pancreatic adenocarcinoma`.

The set is closed and small, so this is a finite piece of writing rather than a
judgement rule -- which is the whole reason it is acceptable here and would not
be on MedCaseReasoning's 6,934 free-text labels.

**What belongs here**: an expansion of an abbreviation, a correction of a typo
in the label, and a name for the *same* condition. **What does not**: an
adjacent or parent condition. `Heart failure` is not accepted for `Acute
pulmonary edema` even though 63 answers said it -- and especially not, because
"heart failure" is itself a cue string in these prompts (evidence E_106), so
accepting it would score the model for reading a finding back out. `Allergic
rhinitis` is not accepted for `Allergic sinusitis`. Two entries are judgement
and are marked: the neoplasm labels, where "cancer" and "adenocarcinoma" are
taken to name the same DDXPlus class.

Nothing is inferred: the audit checks every key against the labels the corpus
actually contains, so a key written from memory that does not exist shows up as
a failure rather than as a silently unused line.
"""

from __future__ import annotations

from typing import Iterable

from .answer_matching import normalize

# Keyed by the normalized gold label. Containment already covers a name the
# gold contains or is contained by, so only what containment cannot reach is
# listed: "pneumothorax" for "Spontaneous pneumothorax" is already a match and
# is not repeated here.
_ALIASES: dict[str, tuple[str, ...]] = {
    "urti": (
        "upper respiratory tract infection",
        "upper respiratory infection",
        "common cold",
    ),
    "psvt": (
        "paroxysmal supraventricular tachycardia",
        "supraventricular tachycardia",
        "svt",
    ),
    "gerd": (
        "gastroesophageal reflux disease",
        "gastro esophageal reflux",
        "acid reflux",
    ),
    "sle": ("systemic lupus erythematosus", "lupus"),
    "hiv initial infection": (
        "acute hiv infection",
        "primary hiv infection",
        "acute retroviral syndrome",
        "hiv seroconversion illness",
    ),
    # DDXPlus's own spelling; the 'n' is missing.
    "larygospasm": ("laryngospasm", "laryngeal spasm"),
    "boerhaave": (
        "boerhaave syndrome",
        "esophageal rupture",
        "spontaneous esophageal perforation",
    ),
    "whooping cough": ("pertussis", "bordetella pertussis infection"),
    "chagas": ("chagas disease", "american trypanosomiasis"),
    "scombroid food poisoning": ("scombroid poisoning", "histamine fish poisoning"),
    "guillain barre syndrome": (
        "gbs",
        "acute inflammatory demyelinating polyneuropathy",
        "aidp",
    ),
    "myasthenia gravis": ("mg",),
    "atrial fibrillation": ("afib", "af", "a fib"),
    "pulmonary embolism": ("pe", "pulmonary thromboembolism"),
    "possible nstemi stemi": (
        "myocardial infarction",
        "acute myocardial infarction",
        "heart attack",
    ),
    "acute pulmonary edema": (
        "flash pulmonary edema",
        "acute cardiogenic pulmonary edema",
    ),
    "acute copd exacerbation infection": (
        "acute exacerbation of copd",
        "infective exacerbation of copd",
    ),
    "acute dystonic reactions": ("acute dystonia", "drug induced dystonia"),
    "bronchospasm acute asthma exacerbation": (
        "acute asthma",
        "asthma attack",
        "status asthmaticus",
    ),
    # Judgement, and the only two of that kind here: DDXPlus's neoplasm labels
    # cover the malignancy, which is what a model names.
    "pancreatic neoplasm": (
        "pancreatic cancer",
        "pancreatic adenocarcinoma",
        "pancreatic tumor",
        "pancreatic carcinoma",
    ),
    "pulmonary neoplasm": (
        "lung cancer",
        "lung carcinoma",
        "lung neoplasm",
        "pulmonary tumor",
        "bronchogenic carcinoma",
    ),
}


def aliases_for(diagnosis_name: str) -> list[str]:
    """Accepted alternative names, or an empty list where none were needed."""
    return list(_ALIASES.get(normalize(diagnosis_name), ()))


def unknown_alias_keys(diagnosis_names: Iterable[str]) -> list[str]:
    """Keys that match no label in the corpus, i.e. entries written wrongly.

    A key with a typo would otherwise sit here forever, accepted by the reader
    as covering a case it never reaches.
    """
    present = {normalize(name) for name in diagnosis_names}
    return sorted(key for key in _ALIASES if key not in present)


def diagnoses_without_aliases(diagnosis_names: Iterable[str]) -> list[str]:
    """Labels this table says nothing about, for review rather than alarm."""
    return sorted(
        {str(name) for name in diagnosis_names if normalize(name) not in _ALIASES}
    )
