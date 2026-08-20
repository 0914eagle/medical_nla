"""Create probe-ready prompt/activation rows from DDXPlus.

The output has two JSONL files:
- cases: one row per selected DDXPlus patient.
- variants: extraction rows compatible with `python -m src.extract_activations`.

The generated prompts intentionally include cue phrases verbatim so target_text
matching is stable.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import random
import re
from collections import Counter, defaultdict
from itertools import islice
from pathlib import Path
from typing import Any, Iterable

from src.case_prompts import build_prompt, findings_prefix


NEGATIVE_VALUE_LABELS = {
    "n",
    "no",
    "false",
    "absent",
    "none",
    "never",
    "not",
    "negative",
    "0",
}

LOW_INFORMATION_VALUE_LABELS = {
    "nowhere",
    "no where",
    "none",
    "unknown",
    "unspecified",
}

# "unknown" says the question was not answered; "no" says it was answered in the
# negative. Only the second is a clinical finding, so only the second can be
# rendered as a cue.
UNINFORMATIVE_VALUE_LABELS = {"unknown", "unspecified", "na", "n/a", "not applicable", "-"}

# An affirmative answer adds nothing to the phrase, so it must not be appended.
# "Y" was missing from this set, which is why cues read "peel off Y".
AFFIRMATIVE_VALUE_LABELS = {"y", "yes", "true", "present", "positive", "1"}

# DDXPlus stores (question id, answer value), not sentences, so a cue phrase has
# to be built. For an affirmative answer the auxiliary is dropped and the
# remainder reads as a finding ("Do you have a cough?" -> "a cough"). A negative
# answer has nowhere to attach once the auxiliary is gone, which is why the
# original code appended the value and produced "a cough no". Keeping the
# auxiliary and negating it is what makes a negative finding readable.
NEGATION_AUXILIARIES = (
    ("has the patient", "has not"),
    ("does the patient", "does not"),
    ("do you", "does not"),
    ("did you", "did not"),
    ("are you", "is not"),
    ("were you", "was not"),
    ("have you", "has not"),
    ("has your", "has not"),
    ("is your", "is not"),
)

# Yes/no questions front an auxiliary. Undoing that inversion is what turns the
# questionnaire item into a finding: "Is the rash swollen?" -> "the rash is
# swollen". Only `you`-subject questions can drop the auxiliary outright.
AUXILIARIES = (
    "do",
    "does",
    "did",
    "is",
    "are",
    "was",
    "were",
    "have",
    "has",
    "can",
    "could",
    "will",
    "would",
)

DETERMINERS = {
    "the", "a", "an", "your", "his", "her", "their", "its", "my", "our",
    "this", "that", "these", "those", "any", "some", "both", "either",
}

WH_WORDS = ("where", "what", "which", "when", "how", "why", "who")

# "How severe is X?" asks about severity, not about "severe". Rating an answer
# needs the noun; without one the dimension is left out rather than guessed.
DIMENSION_NOUNS = {
    "severe": "severity",
    "intense": "intensity",
    "painful": "pain",
    "precise": "precision",
    "fast": "onset speed",
    "long": "duration",
    "often": "frequency",
    "bad": "severity",
    "high": "level",
    "much": "amount",
    "many": "number",
}

# A rendering that still opens like a question, or that puts the auxiliary in
# front of a preposition/conjunction, did not survive the rules. Such cues are
# reported and dropped rather than silently written into a prompt.
MALFORMED_CUE_PATTERNS = [
    r"^(?:" + "|".join(WH_WORDS) + r")\b",
    # A leading auxiliary means the question was never un-inverted -- unless it
    # is a negated finding, where "has not traveled..." is exactly right.
    r"^(?:" + "|".join(AUXILIARIES) + r")\b(?!\s+not\b)",
    r"\b(?:is|are|was|were|do|does|did|have|has)\s+"
    r"(?:or|and|of|to|in|on|at|for|with|from|than)\b",
    # A second clause left in question form: "..., or are the patient underweight".
    r"\bor\s+(?:is|are|was|were|do|does|did|have|has)\b",
    # "the patient" is third-person singular, so a plural verb means the rewrite
    # produced an agreement error: "the patient do feel like the patient are ...".
    r"\bthe patient\s+(?:do|are|were|have)\b",
    # Stripping a clause-heading verb leaves a dangling subordinator, which
    # cannot stand as a finding: "like they are choking".
    r"^(?:like|that|as if|as though)\b",
    # Inversion surviving mid-sentence, where a leading-auxiliary check cannot
    # see it: "in the last month, have they been in contact with ...".
    r"\b(?:have|has|is|are|was|were|do|does|did)\s+(?:they|the patient)\b",
]

# The questionnaire is a fixed vocabulary, so the handful of questions the rules
# cannot reach (compound subjects, two questions in one) are written out. Keyed
# by the normalized question text, with both polarities given explicitly since
# auxiliary negation cannot be derived for these shapes either.
CUE_PHRASE_OVERRIDES: dict[str, dict[str, str]] = {
    "is your nose or the back of your throat itchy": {
        "positive": "an itchy nose or an itchy back of the throat",
        "negative": "no itching of the nose or the back of the throat",
    },
    "is your bmi less than 18.5, or are you underweight": {
        "positive": "a BMI under 18.5 or being underweight",
        "negative": "a BMI of 18.5 or above and not underweight",
    },
}


def cue_phrase_override(question: str, polarity: str) -> str | None:
    entry = CUE_PHRASE_OVERRIDES.get(normalize_label(question).strip(" ?."))
    return entry.get(polarity) if entry else None

GENERIC_CUE_PATTERNS = [
    r"\bpain somewhere\b",
    r"\bhow fast did the pain appear\b",
    r"\bhow intense is the pain\b",
    r"\bhow precisely is the pain located\b",
    r"\bcharacterize their pain\b",
    # Compound screening questions: not one finding, and they restate the
    # specific cues recorded alongside them.
    r"\bany lesions, redness or problems\b",
    r"\bany new fatigue, generalized and vague discomfort\b",
]

# Questions whose answer belongs somewhere other than the end of the phrase.
# Written as (pattern, template) so the value is placed, not appended: the
# radiation item names its sites in place of "another location".
VALUE_PHRASE_TEMPLATES = [
    (
        re.compile(
            r"^(?:does|do)\s+(?P<subject>.+?)\s+radiate to another location$", re.I
        ),
        "{subject} radiates to {value}",
    ),
]


def drop_nested_cues(cues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove cues contained in another cue of the same case.

    "a cough" alongside "a cough that produces colored sputum" gives the readout
    credit for both when it emits one, and leaves two cues resolving to
    overlapping token spans. The MCR segmenter already drops nested spans; this
    is the same rule for assembled prompts. The longer cue is kept, since it
    carries the more specific finding.
    """
    ordered = sorted(cues, key=lambda cue: -len(str(cue.get("cue_text") or "")))
    kept: list[dict[str, Any]] = []
    for cue in ordered:
        text = str(cue.get("cue_text") or "").lower()
        if any(text in str(other.get("cue_text") or "").lower() for other in kept):
            continue
        kept.append(cue)
    keep_ids = {id(cue) for cue in kept}
    return [cue for cue in cues if id(cue) in keep_ids]


def join_values(values: list[str]) -> str:
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return ", ".join(values[:-1]) + f", and {values[-1]}"


def merge_multivalue_cues(cues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse one questionnaire item answered with several values into one cue.

    A patient whose pain radiates to five sites produces five entries of the same
    item, which otherwise become five near-identical cues that crowd out the rest
    of the presentation. One cue naming all the sites carries the same content.
    """
    order: list[str] = []
    groups: dict[str, list[dict[str, Any]]] = {}
    for cue in cues:
        key = str(cue.get("evidence_id"))
        if key not in groups:
            order.append(key)
            groups[key] = []
        groups[key].append(cue)

    merged: list[dict[str, Any]] = []
    for key in order:
        group = groups[key]
        labels = [str(cue.get("value_label") or "") for cue in group]
        if len(group) == 1 or not all(labels):
            merged.extend(group)
            continue
        combined = fold_value_into_question(str(group[0].get("question") or ""), join_values(labels))
        if not combined or is_malformed_cue(combined):
            merged.extend(group)
            continue
        out = dict(group[0])
        out["cue_text"] = combined
        # Joined rather than listed: a merged cue's value fields must stay the
        # same type as an unmerged cue's, or the column holds both strings and
        # lists and no typed reader (Arrow, datasets) can load it.
        out["value_id"] = ",".join(str(cue.get("value_id") or "") for cue in group)
        out["value_label"] = join_values(labels)
        out["merged_value_count"] = len(group)
        merged.append(out)
    return merged


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def read_patient_rows(path: Path) -> Iterable[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
        return
    if suffix == ".json":
        data = read_json(path)
        if isinstance(data, list):
            yield from data
            return
        if isinstance(data, dict):
            for key in ("data", "rows", "patients"):
                if isinstance(data.get(key), list):
                    yield from data[key]
                    return
        raise ValueError(f"Could not find patient rows in JSON file {path}")

    with path.open(encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def get_field(row: dict[str, Any], names: list[str], *, required: bool = True) -> Any:
    lowered = {key.lower(): key for key in row}
    for name in names:
        key = lowered.get(name.lower())
        if key is not None:
            return row[key]
    if required:
        raise KeyError(f"None of fields {names} found in row keys {list(row.keys())}")
    return None


def parse_maybe_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    if text[0] in "[{":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(text)
            except (SyntaxError, ValueError):
                return value
    return value


def parse_evidence_entries(value: Any) -> list[str]:
    parsed = parse_maybe_json(value)
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    if isinstance(parsed, dict):
        return [str(key) for key, present in parsed.items() if present]
    if isinstance(parsed, str):
        for sep in (";", "|"):
            if sep in parsed:
                return [part.strip() for part in parsed.split(sep) if part.strip()]
        if "," in parsed:
            return [part.strip() for part in parsed.split(",") if part.strip()]
        if parsed.strip():
            return [parsed.strip()]
    return []


def evidence_base_and_value(entry: str) -> tuple[str, str | None]:
    for sep in ("_@_", "@", ":"):
        if sep in entry:
            base, value = entry.split(sep, 1)
            return base, value
    return entry, None


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_label(text: str) -> str:
    return normalize_space(str(text).strip().strip("\"'")).lower()


def strip_question_to_phrase(text: str) -> str:
    phrase = normalize_space(text)
    phrase = phrase.strip(" ?.")
    phrase = re.sub(
        r"^(do you|did you|are you|were you|have you|has the patient|does the patient)\s+",
        "",
        phrase,
        flags=re.I,
    )
    phrase = re.sub(
        r"^(been|have|has|feel|experience|experiencing|suffer from|present with|"
        r"noticed|notice|observed|observe|objectified|felt|measured)\s+"
        r"(?!(?:like|that|as if)\b)",
        "",
        phrase,
        flags=re.I,
    )
    phrase = re.sub(
        r"^(feel|felt|experience|experiencing|notice|noticed)\s+(?=(?:like|that|as if)\b)",
        r"they \1 ",
        phrase,
        flags=re.I,
    )
    phrase = rewrite_second_person(phrase)
    if not phrase:
        phrase = normalize_space(text).strip(" ?.")
    return lower_first(phrase)


# Words after which "you" is still the subject. Anywhere else it is an object,
# where "they" is wrong: "keeping you from turning your head" must not become
# "keeping they from turning their head".
SUBJECT_LICENSORS = {
    "that", "when", "if", "and", "or", "but", "because", "while", "whether",
    "as", "since", "unless", "until", "before", "after", "though", "although",
    "do", "does", "did", "are", "were", "have", "has", "is", "was", "can", "will",
}


def second_person_pronoun(preceding: str, following: str) -> str:
    """Subject or object case for a rewritten `you`.

    Two signals, either sufficient. A following auxiliary marks a subject even
    deep inside a clause ("the condition you are consulting for"); a preceding
    conjunction or clause opener marks one at the start of a clause. Anything
    else is an object, where "they" would be wrong.
    """
    words = preceding.split()
    last = words[-1].lower().strip(",;:") if words else ""
    nxt = following.split()[0].lower().strip(",;:.") if following.split() else ""
    if nxt in AUXILIARIES or nxt == "not":
        return "they"
    return "they" if (not last or last in SUBJECT_LICENSORS) else "them"


def rewrite_second_person(text: str) -> str:
    text = re.sub(r"\bwhen you exhale\b", "when exhaling", text, flags=re.I)
    text = re.sub(r"\byour\b", "their", text, flags=re.I)
    text = re.sub(
        r"\byou\b",
        lambda match: second_person_pronoun(text[: match.start()], text[match.end() :]),
        text,
        flags=re.I,
    )
    text = re.sub(r"\b(yes or no|right now|currently)\b", "", text, flags=re.I)
    return normalize_space(text).strip(" ?.:;,-")


def lower_first(text: str) -> str:
    return text[:1].lower() + text[1:] if text else text


def split_subject(words: list[str]) -> tuple[list[str], list[str]]:
    """Guess where the subject ends, so the auxiliary can move behind it.

    A determiner or possessive takes the following noun with it ("your
    symptoms"); anything else is treated as a one-word subject ("there").
    Compound subjects defeat this and are caught by the malformed-cue check.
    """
    if not words:
        return [], []
    take = 2 if words[0].lower() in DETERMINERS and len(words) > 2 else 1
    return words[:take], words[take:]


def uninvert_question(question: str) -> str | None:
    """Undo subject-auxiliary inversion: 'Is the rash swollen' -> 'the rash is swollen'."""
    text = normalize_space(question).strip(" ?.")
    # "Is the lesion (or are the lesions) larger than 1cm?" -- the parenthetical
    # restates the question for a plural subject and only blocks the rewrite.
    text = normalize_space(re.sub(r"\((?:or|and)\s+[^)]*\)", "", text, flags=re.I))
    match = re.match(rf"^({'|'.join(AUXILIARIES)})\s+(.+)$", text, flags=re.I)
    if not match:
        return None
    auxiliary = match.group(1).lower()
    subject, tail = split_subject(match.group(2).split())
    if not subject or not tail:
        return None
    phrase = rewrite_second_person(" ".join([*subject, auxiliary, *tail]))
    phrase = re.sub(r"^there\s+(is|are)\s+any\b", r"there \1", phrase, flags=re.I)
    return lower_first(phrase)


def drop_do_support(question: str) -> str | None:
    """Delete a plural do/did, which carries no meaning once the question is a statement.

    Needed when the subject runs long ("any members of your immediate family"),
    where guessing the subject boundary to move the auxiliary behind it fails.
    Restricted to do/did because dropping `does` leaves a person-agreement error.
    """
    text = normalize_space(question).strip(" ?.")
    match = re.match(r"^(do|did)\s+(.+)$", text, flags=re.I)
    if not match:
        return None
    return lower_first(rewrite_second_person(match.group(2)))


def fold_value_into_question(question: str, value_label: str) -> str | None:
    """Place an answer inside the statement instead of after it.

    'Where is the swelling located?' + 'iliac wing(R)' becomes 'the swelling is
    located in the iliac wing(R)' rather than the two pasted together. Returning
    a single phrase for a whole answer set is also what lets one item answered
    with several values collapse into one cue.
    """
    text = normalize_space(question).strip(" ?.")
    for pattern, template in VALUE_PHRASE_TEMPLATES:
        match = pattern.match(text)
        if match:
            filled = template.format(value=with_article(value_label), **match.groupdict())
            return lower_first(rewrite_second_person(filled))
    value = normalize_space(value_label)
    # A bare site name reads as a location only with an article: "in the chest".
    site = value if value.split()[:1] and value.split()[0].lower() in DETERMINERS else f"the {value}"

    match = re.match(r"^where\s+(is|are)\s+(.+?)\s+located$", text, flags=re.I)
    if match:
        return lower_first(
            rewrite_second_person(f"{match.group(2)} {match.group(1)} located in {site}")
        )
    match = re.match(r"^where\s+(is|are)\s+(.+)$", text, flags=re.I)
    if match:
        return lower_first(rewrite_second_person(f"{match.group(2)} {match.group(1)} in {site}"))
    # "What color is the rash?" -> "the rash color is red"
    match = re.match(r"^what\s+(\w+)\s+(is|are)\s+(.+)$", text, flags=re.I)
    if match:
        return lower_first(
            rewrite_second_person(f"{match.group(3)} {match.group(1)} {match.group(2)} {value}")
        )
    # "How severe is the itching?" answered 2 must not come out as "the itching
    # is severe: 2" -- that asserts severity the rating contradicts.
    match = re.match(r"^how\s+(\w+)\s+(is|are|does|did)\s+(.+)$", text, flags=re.I)
    if match:
        subject, dimension = match.group(3), DIMENSION_NOUNS.get(match.group(1).lower())
        if is_numeric_value(value):
            rated = f"{subject} rated {value} for {dimension}" if dimension else f"{subject} rated {value}"
            return lower_first(rewrite_second_person(rated))
        return lower_first(rewrite_second_person(f"{subject} {match.group(2)} {value}"))
    return None


def with_article(value_label: str) -> str:
    """A bare site name reads as a location only with an article: 'in the chest'."""
    value = normalize_space(value_label)
    first = value.split()[:1]
    if first and first[0].lower() in DETERMINERS:
        return value
    return f"the {value}"


def is_malformed_cue(text: str) -> bool:
    lowered = normalize_label(text)
    return any(re.search(pattern, lowered) for pattern in MALFORMED_CUE_PATTERNS)


def lookup_meta(evidence_meta: dict[str, Any], evidence_id: str) -> dict[str, Any]:
    meta = evidence_meta.get(evidence_id)
    if isinstance(meta, dict):
        return meta
    return {}


def meta_text(meta: dict[str, Any], fallback: str) -> str:
    for key in (
        "question_en",
        "question",
        "name",
        "label",
        "display_name",
        "description",
        "text",
    ):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return fallback


def possible_value_label(meta: dict[str, Any], value_id: str | None) -> str | None:
    if value_id is None:
        return None
    value_meaning = meta.get("value_meaning")
    if isinstance(value_meaning, dict):
        value_meta = value_meaning.get(value_id)
        if isinstance(value_meta, str):
            return value_meta
        if isinstance(value_meta, dict):
            for label_key in (
                "en",
                "label",
                "name",
                "value",
                "text",
                "meaning",
                "value_meaning",
                "value_label",
            ):
                label = value_meta.get(label_key)
                if isinstance(label, str) and label.strip():
                    return label
    for label_key in ("value_label", "meaning"):
        label = meta.get(label_key)
        if isinstance(label, str) and label.strip():
            return label
    for key in ("possible-values", "possible_values", "values"):
        values = meta.get(key)
        if not isinstance(values, dict):
            continue
        value_meta = values.get(value_id)
        if isinstance(value_meta, str):
            return value_meta
        if isinstance(value_meta, dict):
            for label_key in (
                "en",
                "label",
                "name",
                "value",
                "text",
                "meaning",
                "value_meaning",
                "value_label",
            ):
                label = value_meta.get(label_key)
                if isinstance(label, str) and label.strip():
                    return label
    # No label for this value id. Returning None would erase the answer, and a
    # negative answer erased is worse than dropped: the phrase then asserts the
    # affirmative ("traveled out of the country" for a patient who did not).
    # The raw id still carries the polarity for Y/N-style values; opaque codes
    # are caught by is_opaque_value_id.
    return value_id


def is_negative_or_low_information_value(value_label: str | None) -> bool:
    if value_label is None:
        return False
    label = normalize_label(value_label)
    return label in NEGATIVE_VALUE_LABELS or label in LOW_INFORMATION_VALUE_LABELS


OPAQUE_VALUE_ID = re.compile(r"^[a-z]{1,3}[_-]?\d+$")
NUMERIC_VALUE = re.compile(r"^\d+(?:\.\d+)?$")


def is_numeric_value(value_label: str | None) -> bool:
    """True for a rating-scale answer, which must not be pasted on as a bare number."""
    if value_label is None:
        return False
    return bool(NUMERIC_VALUE.match(normalize_label(value_label)))


def attach_value(phrase: str, value_label: str) -> str:
    """Append an answer value in a form that reads as an answer.

    A bare number produced "the rash is swollen 3"; parenthesising it says the
    same thing without pretending to be part of the finding.
    """
    if is_numeric_value(value_label):
        return normalize_space(f"{phrase} (rated {normalize_space(value_label)})")
    return normalize_space(f"{phrase} {value_label}")


def is_opaque_value_id(value_label: str | None) -> bool:
    """True for an unlabelled code like 'V_29', which must not reach the prompt."""
    if value_label is None:
        return False
    return bool(OPAQUE_VALUE_ID.match(normalize_label(value_label)))


def is_uninformative_value(value_label: str | None) -> bool:
    """True when the value records a missing answer rather than a negative one."""
    if value_label is None:
        return False
    return normalize_label(value_label) in UNINFORMATIVE_VALUE_LABELS


def render_negative_phrase(question: str) -> str | None:
    """Turn a questionnaire item answered in the negative into a finding phrase.

    Returns None when the question does not open with a recognized auxiliary;
    the caller then excludes the cue rather than emit something ungrammatical.
    Prefixing "no" instead would break on verb phrases ("no traveled out of the
    country"), so the auxiliary is negated in place.
    """
    text = normalize_space(question).strip(" ?.")
    for auxiliary, negated in NEGATION_AUXILIARIES:
        match = re.match(rf"^{auxiliary}\s+(.*)$", text, flags=re.I)
        if not match:
            continue
        rest = match.group(1)
        rest = re.sub(r"\bwhen you exhale\b", "when exhaling", rest, flags=re.I)
        rest = re.sub(r"\byour\b", "their", rest, flags=re.I)
        rest = re.sub(r"\byou\b", "the patient", rest, flags=re.I)
        rest = re.sub(r"\b(yes or no|right now|currently)\b", "", rest, flags=re.I)
        phrase = normalize_space(f"{negated} {rest}").strip(" ?.:;,-")
        return phrase or None
    return None


def is_generic_cue_text(text: str) -> bool:
    normalized = normalize_label(text)
    return any(re.search(pattern, normalized) for pattern in GENERIC_CUE_PATTERNS)


def is_antecedent(meta: dict[str, Any]) -> bool:
    value = meta.get("is_antecedent")
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return False


def cue_from_entry(
    entry: str,
    evidence_meta: dict[str, Any],
    *,
    clean_cues: bool,
    negative_cues: bool = False,
) -> dict[str, Any]:
    base_id, value_id = evidence_base_and_value(entry)
    meta = lookup_meta(evidence_meta, base_id)
    question = meta_text(meta, base_id)
    phrase = cue_phrase_override(question, "positive") or strip_question_to_phrase(question)
    value_label = possible_value_label(meta, value_id)
    polarity = "positive"
    excluded = False
    exclusion_reason = None

    if clean_cues and is_uninformative_value(value_label):
        # A missing answer is not a finding in either direction.
        excluded = True
        exclusion_reason = "uninformative_value"
    elif clean_cues and is_negative_or_low_information_value(value_label):
        if negative_cues:
            negated = cue_phrase_override(question, "negative") or render_negative_phrase(question)
            if negated:
                phrase = negated
                polarity = "negative"
            else:
                excluded = True
                exclusion_reason = "negative_value_unrenderable"
        else:
            excluded = True
            exclusion_reason = "negative_or_low_information_value"
    elif clean_cues and is_opaque_value_id(value_label):
        excluded = True
        exclusion_reason = "opaque_value_id"
    elif clean_cues and value_label and normalize_label(value_label) not in AFFIRMATIVE_VALUE_LABELS:
        # A real answer value ("chest", "red") belongs inside the statement, not
        # pasted onto the end of the question.
        folded = fold_value_into_question(question, value_label)
        phrase = folded if folded else attach_value(phrase, value_label)
    elif not clean_cues and value_label and value_label.lower() not in {"yes", "true", "present"}:
        phrase = normalize_space(f"{phrase} {value_label}")

    if clean_cues and not excluded and is_malformed_cue(phrase):
        # The question kept its interrogative shape; try undoing the inversion,
        # then simply deleting do-support, before giving up on it.
        repaired = next(
            (
                candidate
                for candidate in (uninvert_question(question), drop_do_support(question))
                if candidate and not is_malformed_cue(candidate)
            ),
            None,
        )
        if repaired:
            if value_label and normalize_label(value_label) not in AFFIRMATIVE_VALUE_LABELS:
                repaired = attach_value(repaired, value_label)
            phrase = repaired
        else:
            excluded = True
            exclusion_reason = "unrenderable_question"

    # Match the question as well as the rendering: un-inverting moves the
    # auxiliary into the middle ("the pain does radiate"), which would otherwise
    # walk a screening question straight past the generic filter.
    if clean_cues and (is_generic_cue_text(phrase) or is_generic_cue_text(question)):
        excluded = True
        exclusion_reason = exclusion_reason or "generic_cue"
    return {
        "evidence_id": base_id,
        "evidence_entry": entry,
        "question": question,
        "value_id": value_id,
        "value_label": value_label,
        "cue_text": phrase,
        "cue_type": "antecedent" if is_antecedent(meta) else "symptom",
        "cue_polarity": polarity,
        "is_antecedent": is_antecedent(meta),
        "excluded": excluded,
        "exclusion_reason": exclusion_reason,
    }


def slug(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return value or "unknown"


def join_cues(cues: list[str]) -> str:
    if len(cues) == 1:
        return cues[0]
    if len(cues) == 2:
        return f"{cues[0]} and {cues[1]}"
    return ", ".join(cues[:-1]) + f", and {cues[-1]}"


def make_prompt(
    cues: list[str], *, condition: str = "direct", age: Any = None, sex: Any = None
) -> str:
    """Presentation plus one condition's instruction; see src.case_prompts."""
    return build_prompt(findings_prefix(cues, age=age, sex=sex), condition)


def parse_differential(value: Any) -> list[dict[str, Any]]:
    """DDXPlus's probability-weighted differential, as [{diagnosis, probability}].

    Stored as a stringified list of [name, probability] pairs. Kept because it is
    a gold ranking rather than a single label: it lets a predicted diagnosis be
    scored by where it sits in the differential, so choosing the second-ranked
    condition can be told apart from choosing an unrelated one.
    """
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                out.append({"diagnosis": str(item[0]), "probability": float(item[1])})
            except (TypeError, ValueError):
                continue
    return out


def make_case(
    row: dict[str, Any],
    *,
    row_index: int,
    evidence_meta: dict[str, Any],
    rng: random.Random,
    prefer_symptoms: bool,
    max_cues: int,
    clean_cues: bool = True,
    negative_cues: bool = False,
) -> dict[str, Any] | None:
    pathology = str(get_field(row, ["PATHOLOGY", "pathology", "diagnosis", "label"]))
    patient_id = get_field(row, ["id", "patient_id", "PATIENT", "patient"], required=False)
    if patient_id is None:
        patient_id = f"row_{row_index:07d}"
    entries = parse_evidence_entries(get_field(row, ["EVIDENCES", "evidences", "evidence"]))
    all_cues = [
        cue_from_entry(
            entry, evidence_meta, clean_cues=clean_cues, negative_cues=negative_cues
        )
        for entry in entries
    ]
    cues = drop_nested_cues(
        merge_multivalue_cues(
            [
                cue
                for cue in all_cues
                if cue["cue_text"] and cue["cue_text"].lower() != "none" and not cue["excluded"]
            ]
        )
    )
    symptom_cues = [cue for cue in cues if not cue["is_antecedent"]]
    candidate_cues = symptom_cues if prefer_symptoms and len(symptom_cues) >= max_cues else cues
    if len(candidate_cues) < max_cues:
        return None
    selected = rng.sample(candidate_cues, max_cues)
    cue_targets = [cue["cue_text"] for cue in selected]
    diagnosis_id = slug(pathology)
    case_id = f"ddxplus_{diagnosis_id}_{row_index:07d}"
    return {
        "id": case_id,
        "source": "ddxplus",
        "patient_id": str(patient_id),
        "diagnosis_id": diagnosis_id,
        "diagnosis_name": pathology,
        "diagnosis_aliases": [pathology],
        "cue_targets": cue_targets,
        "cue_types": [cue["cue_type"] for cue in selected],
        "cue_polarities": [cue["cue_polarity"] for cue in selected],
        "cue_evidence_ids": [cue["evidence_id"] for cue in selected],
        "cue_evidence_entries": [cue["evidence_entry"] for cue in selected],
        "cue_value_ids": [cue["value_id"] for cue in selected],
        "cue_value_labels": [cue["value_label"] for cue in selected],
        "clean_cues": clean_cues,
        "negative_cues": negative_cues,
        "prefer_symptoms": prefer_symptoms,
        "excluded_cue_count": sum(1 for cue in all_cues if cue["excluded"]),
        "excluded_cue_entries": [
            {
                "evidence_entry": cue["evidence_entry"],
                "cue_text": cue["cue_text"],
                "exclusion_reason": cue["exclusion_reason"],
            }
            for cue in all_cues
            if cue["excluded"]
        ],
        "single_prompt": make_prompt([cue_targets[0]]),
        "multi_prompt": make_prompt(cue_targets),
    }


def variant_rows(case: dict[str, Any]) -> list[dict[str, Any]]:
    common = {
        "base_id": case["id"],
        "source": case["source"],
        "patient_id": case["patient_id"],
        "diagnosis_id": case["diagnosis_id"],
        "diagnosis_name": case["diagnosis_name"],
        "diagnosis_aliases": case["diagnosis_aliases"],
        "cue_targets": case["cue_targets"],
        "cue_types": case["cue_types"],
        "cue_evidence_ids": case["cue_evidence_ids"],
        "cue_evidence_entries": case["cue_evidence_entries"],
        "cue_value_ids": case["cue_value_ids"],
        "cue_value_labels": case["cue_value_labels"],
        "clean_cues": case["clean_cues"],
        "excluded_cue_count": case["excluded_cue_count"],
        "excluded_cue_entries": case["excluded_cue_entries"],
    }
    rows = [
        {
            **common,
            "id": f"{case['id']}__single_cue",
            "variant": "single_cue",
            "condition": "single",
            "target_role": "cue",
            "cue_index": 1,
            "prompt": case["single_prompt"],
            "position_mode": "target_text",
            "target_text": case["cue_targets"][0],
            "target_text_strategy": "span_mean",
        },
        {
            **common,
            "id": f"{case['id']}__single_format",
            "variant": "single_format",
            "condition": "single",
            "target_role": "format",
            "cue_index": None,
            "prompt": case["single_prompt"],
            "position_mode": "last_token",
            "target_text": None,
            "target_text_strategy": None,
        },
    ]
    for idx, cue in enumerate(case["cue_targets"], start=1):
        rows.append(
            {
                **common,
                "id": f"{case['id']}__multi_cue_{idx}",
                "variant": f"multi_cue_{idx}",
                "condition": "multi",
                "target_role": "cue",
                "cue_index": idx,
                "prompt": case["multi_prompt"],
                "position_mode": "target_text",
                "target_text": cue,
                "target_text_strategy": "span_mean",
            }
        )
    rows.append(
        {
            **common,
            "id": f"{case['id']}__multi_format",
            "variant": "multi_format",
            "condition": "multi",
            "target_role": "format",
            "cue_index": None,
            "prompt": case["multi_prompt"],
            "position_mode": "last_token",
            "target_text": None,
            "target_text_strategy": None,
        }
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patients", required=True)
    parser.add_argument("--evidences", required=True)
    parser.add_argument("--cases-output", required=True)
    parser.add_argument("--variants-output", required=True)
    parser.add_argument("--max-diagnoses", type=int, default=49)
    parser.add_argument("--examples-per-diagnosis", type=int, default=100)
    parser.add_argument("--max-cues", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--clean-cues",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Filter negative/low-information values and generic pain metadata "
            "before sampling cues. Use --no-clean-cues to reproduce v1 behavior."
        ),
    )
    parser.add_argument(
        "--negative-cues",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Keep negatively-answered evidences as cues, rendered by negating the "
            "question's auxiliary ('has not traveled out of the country'). Without "
            "this they are dropped, so prompts carry positive findings only. "
            "Requires --clean-cues. The choice is recorded per case as negative_cues."
        ),
    )
    parser.add_argument(
        "--prefer-symptoms",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer non-antecedent symptom cues when at least --max-cues are available.",
    )
    parser.add_argument("--max-patient-rows", type=int, default=None)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    evidence_meta = read_json(Path(args.evidences))
    if not isinstance(evidence_meta, dict):
        raise ValueError("Evidence metadata must be a JSON object keyed by evidence id.")

    by_diagnosis: dict[str, list[dict[str, Any]]] = defaultdict(list)
    patients = read_patient_rows(Path(args.patients))
    if args.max_patient_rows is not None:
        patients = islice(patients, args.max_patient_rows)
    # The full release has over a million patient rows and nothing is written
    # until the scan finishes, so report progress rather than looking hung.
    for row_index, row in enumerate(patients):
        if row_index and row_index % 100_000 == 0:
            kept = sum(len(bucket) for bucket in by_diagnosis.values())
            print(
                f"[scan] {row_index:,} rows read | {len(by_diagnosis)} diagnoses "
                f"| {kept:,} cases kept",
                flush=True,
            )
        case = make_case(
            row,
            row_index=row_index,
            evidence_meta=evidence_meta,
            rng=rng,
            prefer_symptoms=args.prefer_symptoms,
            max_cues=args.max_cues,
            clean_cues=args.clean_cues,
            negative_cues=args.negative_cues,
        )
        if case is None:
            continue
        bucket = by_diagnosis[case["diagnosis_id"]]
        if len(bucket) < args.examples_per_diagnosis:
            bucket.append(case)
    print(f"[scan] done: {row_index + 1:,} rows read", flush=True)

    selected_diagnoses = [
        diagnosis
        for diagnosis, cases in sorted(
            by_diagnosis.items(), key=lambda item: (-len(item[1]), item[0])
        )
        if len(cases) >= args.examples_per_diagnosis
    ][: args.max_diagnoses]
    cases = [case for diagnosis in selected_diagnoses for case in by_diagnosis[diagnosis]]
    variants = [row for case in cases for row in variant_rows(case)]

    write_jsonl(Path(args.cases_output), cases)
    write_jsonl(Path(args.variants_output), variants)

    print(f"diagnoses_selected: {len(selected_diagnoses)}")
    print(f"cases_written: {len(cases)}")
    print(f"variants_written: {len(variants)}")
    print(f"variants_per_case: {2 + args.max_cues + 1}")
    print(f"clean_cues: {args.clean_cues}")
    print(f"excluded_cues_total: {sum(case['excluded_cue_count'] for case in cases)}")
    print("top_diagnoses:")
    for diagnosis, count in Counter(case["diagnosis_id"] for case in cases).most_common(20):
        print(f"  {diagnosis}: {count}")


if __name__ == "__main__":
    main()
