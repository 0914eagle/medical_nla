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

ROLE = "You are an expert physician."

# Kept verbatim from the source format: a fixed closing string is what makes
# answers parseable, and what lets a truncated chain be forced to an answer.
ANSWER_FORMAT = 'You MUST end your response with exactly "The answer is <diagnosis>."'

# The suffix appended to a truncated chain of thought to force an answer.
ANSWER_CUE = "The answer is"

ANSWER_PATTERN = re.compile(r"[Tt]he\s+answer\s+is\s+(.+?)\s*(?:\.|$)")

DIRECT_INSTRUCTION = f"What is the single most likely diagnosis?\n\n{ANSWER_FORMAT}"

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

FINDINGS_HEADER = f"{ROLE} A patient presents with the following findings:"
PROSE_HEADER = f"{ROLE} A patient presents as follows:"


def findings_prefix(cues: list[str]) -> str:
    """Presentation as a list, for corpora whose cues are assembled.

    An inline "presents with X, Y and Z" sentence needs every cue to be a noun
    phrase, but un-inverting a questionnaire item yields a clause ("the rash is
    swollen"). A list takes both, and the line break rather than a comma marks
    the boundary, so a cue containing commas of its own stays unambiguous.
    """
    listed = "\n".join(f"- {cue}" for cue in cues)
    return f"{FINDINGS_HEADER}\n{listed}"


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
    """Recover the diagnosis from a response, or None if it did not comply."""
    match = ANSWER_PATTERN.search(str(text or ""))
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
