"""The finding's own tokens, located inside a target that is mostly scaffold.

DDXPlus training loss reached 1e-4 a quarter of the way through its first
epoch. Nothing was leaking -- the AV prompt is the same template for every row
and carries no case text -- but six of a target's seven lines are the same XML
in every row, so a mean over the whole target says mostly how well six constant
lines have been memorized. Validation selected the best epoch on that number
and the layer trajectory would have been read off it.
"""

from scripts.make_medical_nla_v3_cue_first_targets import (
    content_char_spans,
    cue_first_target_text,
)


def target(cues, max_cues=4):
    return cue_first_target_text(
        {"id": "r1", "cue_targets": cues}, max_cues=max_cues, seed=17
    )


def spans_of(text):
    return [text[start:end] for start, end in content_char_spans(text)]


def test_the_spans_recover_exactly_the_cues_the_writer_put_in():
    text = target(["pain in the lower chest"])
    assert spans_of(text) == ["pain in the lower chest"]


def test_every_cue_line_is_found_when_a_row_carries_several():
    text = target(["fever", "a dry cough", "pain on inspiration"])
    assert sorted(spans_of(text)) == ["a dry cough", "fever", "pain on inspiration"]


def test_the_scaffold_is_most_of_the_target():
    """The reason the split exists, as a number rather than an assertion in
    prose: if this ever stops holding, the two losses converge and the
    selection question goes away."""
    text = target(["pain in the lower chest"])
    content = sum(end - start for start, end in content_char_spans(text))
    assert content / len(text) < 0.3


def test_a_target_with_no_cue_lines_has_no_content():
    assert content_char_spans("<explanation>\n<observed>\n</observed>") == []


def test_a_hyphen_inside_a_cue_does_not_start_a_new_span():
    """Only a line *beginning* "- " is a cue line; a dash mid-cue is content."""
    text = target(["Mallory-Weiss tear - suspected"])
    assert spans_of(text) == ["Mallory-Weiss tear - suspected"]
