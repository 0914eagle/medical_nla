"""The prompts every case is presented under, in one place.

Two properties are load-bearing and neither survives being reimplemented per
dataset, which is why this module exists rather than a string in each generator.

**The instruction follows the presentation.** Under causal attention a token's
hidden state cannot depend on anything after it, so putting the instruction last
makes the cue-position activations identical across instruction conditions: one
extraction serves the direct and chain-of-thought arms alike. Putting it first --
as the pilot's answer-generation script did, while extraction used the bare case
prompt -- means the activations belong to a different forward pass than the
correctness labels derived from them.

**The two conditions share a byte-identical prefix.** Anything that differs must
live in the suffix, including the clinician framing.

The chain-of-thought wording follows the medical CoT distillation audit
(arXiv:2605.28301), adapted in three ways: the answer is an open-ended diagnosis
rather than one of four lettered options, the instruction moves after the
presentation for the reason above, and the exam framing ("taking the USMLE") is
dropped because these are patient presentations, not exam items. The fixed
closing string is kept verbatim so answers parse deterministically and so a
truncated chain can be completed by appending it -- the early-answering test of
Lanham et al. (arXiv:2307.13702).
"""

from __future__ import annotations

import re
from typing import Any

ROLE = "You are an expert physician."

# Kept verbatim from the source format: a fixed closing string is what makes
# answers parseable, and what lets a truncated chain be forced to an answer.
ANSWER_FORMAT = 'You MUST end your response with exactly "The answer is <diagnosis>."'

# Appended to a chain of thought to force an answer out of it, and prefilled
# into the assistant turn to make the direct condition actually direct.
ANSWER_CUE = "The answer is"

ANSWER_PATTERN = re.compile(r"[Tt]he\s+answer\s+is\s+(.+?)\s*(?:\.|$)")

# "Give the diagnosis only" is not redundant with asking for "the single most
# likely diagnosis". Under the shorter wording, with the assistant turn started
# at the answer cue, the model completes "The answer is" as ordinary prose --
# "relatively straightforward given the constellation of findings" -- and the
# parser returns that as the diagnosis. Asking for one diagnosis says what to
# choose; it does not say not to write an essay around it.
#
# The instruction sits after the presentation, so tightening it changes no
# cue-position activation: under causal attention those tokens cannot see it.
DIRECT_INSTRUCTION = (
    "What is the single most likely diagnosis?\n"
    "\n"
    "Give the diagnosis only. Do not explain your reasoning.\n"
    "\n"
    f"{ANSWER_FORMAT}"
)

COT_INSTRUCTION = (
    "Work through this case as a natural reasoning process.\n"
    "\n"
    "Think about:\n"
    "- What the key clinical findings suggest\n"
    "- Which diagnoses fit the presentation and which do not\n"
    "- Whether your conclusion holds up under scrutiny\n"
    "\n"
    f"{ANSWER_FORMAT}"
)

INSTRUCTIONS = {"direct": DIRECT_INSTRUCTION, "cot": COT_INSTRUCTION}

PROSE_HEADER = f"{ROLE} A patient presents as follows:"


def patient_descriptor(age: Any = None, sex: Any = None) -> str:
    """"a 34-year-old woman" from an age and a sex, or "a patient" without them.

    Age and sex are diagnostic -- croup is paediatric, and several DDXPlus
    pathologies are sex-specific -- and the simulator that generated the corpus
    conditions on them. Case-report presentations open with them almost without
    exception, so omitting them on one corpus and not the other makes the two
    differ in a way that has nothing to do with the question being asked.
    """
    try:
        years = int(float(age))
    except (TypeError, ValueError):
        years = None
    code = str(sex or "").strip().upper()[:1]
    if years is None and code not in {"M", "F"}:
        return "A patient"
    if code == "M":
        noun = "boy" if years is not None and years < 18 else "man"
    elif code == "F":
        noun = "girl" if years is not None and years < 18 else "woman"
    else:
        noun = "child" if years is not None and years < 18 else "patient"
    if years is None:
        return f"A {noun}"
    return f"A {years}-year-old {noun}"


def findings_prefix(cues: list[str], *, age: Any = None, sex: Any = None) -> str:
    """Presentation as a list, for corpora whose cues are assembled.

    An inline "presents with X, Y and Z" sentence needs every cue to be a noun
    phrase, but un-inverting a questionnaire item yields a clause ("the rash is
    swollen"). A list takes both, and the line break rather than a comma marks
    the boundary, so a cue containing commas of its own stays unambiguous.

    Age and sex head the sentence rather than joining the list: they are context
    for reading the findings, not findings to be read back, and putting them in
    the list would make them cues.
    """
    listed = "\n".join(f"- {cue}" for cue in cues)
    who = patient_descriptor(age, sex)
    return f"{ROLE} {who} presents with the following findings:\n{listed}"


def prose_prefix(text: str) -> str:
    """Presentation as it was written, for corpora whose cues are cut from prose."""
    return f"{PROSE_HEADER}\n\n{text.strip()}"


def build_prompt(prefix: str, condition: str = "direct") -> str:
    """Attach a condition's instruction after a presentation."""
    try:
        instruction = INSTRUCTIONS[condition]
    except KeyError:
        raise ValueError(
            f"Unknown condition {condition!r}; expected one of {sorted(INSTRUCTIONS)}."
        ) from None
    return f"{prefix}\n\n{instruction}"


def parse_answer(text: str) -> str | None:
    """Recover the final formatted diagnosis, or None if it did not comply."""
    matches = list(ANSWER_PATTERN.finditer(str(text or "")))
    match = matches[-1] if matches else None
    return " ".join(match.group(1).split()) if match else None


def truncate_chain(chain: str, fraction: float) -> str:
    """A prefix of a chain of thought, cut at a sentence boundary.

    Early answering asks whether the stated reasoning is load-bearing: cut it
    short, force an answer, and see whether the answer moves. Cutting mid-word
    would test tokenization rather than reasoning, so the cut falls back to the
    last sentence end at or before the requested point.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1]; got {fraction}.")
    text = str(chain or "")
    cut = int(len(text) * fraction)
    if cut >= len(text):
        return text
    boundary = max(text.rfind(mark, 0, cut) for mark in (". ", ".\n", "! ", "? "))
    if boundary > 0:
        cut = boundary + 1
    return text[:cut].rstrip()


def early_answer_prompt(prefix: str, chain: str, fraction: float) -> str:
    """The chain-of-thought prompt with a truncated chain and the answer forced."""
    truncated = truncate_chain(chain, fraction)
    return f"{build_prompt(prefix, 'cot')}\n\n{truncated}\n\n{ANSWER_CUE}"


def prefilled_assistant_turn(chat_text: str, chain: str = "") -> str:
    """Start the model's own turn at the answer, so nothing can precede it.

    Asked for "the single most likely diagnosis", gemma-3-12b-it opens with
    "Okay, let's break down this case" and reasons for hundreds of tokens. That
    is not a budget problem to be solved by raising `max_new_tokens`: it means
    the direct condition, left free, is a chain-of-thought condition, and the
    contrast the paper rests on does not exist. Writing the answer cue into the
    assistant turn removes the room to reason rather than asking it not to.

    Under causal attention the appended tokens cannot change the hidden state of
    any token before them, so both the cue positions and the format position --
    the case prompt's final token -- are identical with and without the prefill.
    One extraction still serves both conditions.

    With `chain`, the same mechanism completes a chain of thought that ran out
    of budget: the model's own reasoning is given back to it verbatim and only
    the answer is asked for.
    """
    if chain:
        return f"{chat_text}{chain.rstrip()}\n\n{ANSWER_CUE}"
    return f"{chat_text}{ANSWER_CUE}"
