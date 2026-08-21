"""Text helpers for NLA output, with no model dependency.

Separated from `src.nla` because that module imports torch at load, and these
three functions are string handling. Every scoring and analysis script needed
them and so needed a GPU stack to import -- which also meant those scripts
could not be unit-tested anywhere the stack is absent.

`src.nla` re-exports them, so existing imports keep working.
"""

from __future__ import annotations

import re
from pathlib import Path

EXPLANATION_RE = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.DOTALL)

AV_PROMPT_FILENAME = "av_prompt.txt"


def extract_explanation(text: str) -> tuple[str, bool]:
    match = EXPLANATION_RE.search(text)
    if match is None:
        return text.strip(), False
    return match.group(1).strip(), True


def cjk_fraction(text: str) -> float:
    """Share of CJK characters, which is how a collapsed AV announces itself."""
    if not text:
        return 0.0
    cjk = 0
    for ch in text:
        code = ord(ch)
        if (
            0x4E00 <= code <= 0x9FFF
            or 0x3400 <= code <= 0x4DBF
            or 0x3040 <= code <= 0x30FF
            or 0xAC00 <= code <= 0xD7AF
        ):
            cjk += 1
    return cjk / len(text)


def adapter_av_prompt(adapter_id: str | None) -> str | None:
    """The AV prompt an adapter was trained under, if it recorded one.

    Training and generation must use the same prompt, and a flag that has to be
    remembered at both ends is a flag that gets forgotten: the pilot trained
    against an <observed> target while generation fell back to the checkpoint's
    own diagnosis prompt, so the adapter had learned a format nothing asked it
    for. The trainer writes the template it used beside the adapter; reading it
    back here makes the pair structural rather than remembered.

    Returns None for a hub id or a directory without the file, leaving the
    caller's own resolution untouched.
    """
    if not adapter_id:
        return None
    path = Path(adapter_id) / AV_PROMPT_FILENAME
    try:
        if path.is_file():
            return path.read_text(encoding="utf-8")
    except OSError:
        return None
    return None
