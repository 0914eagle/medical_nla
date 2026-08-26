import pytest

from src.extract_activations import (
    group_by_prompt,
    resolve_positions,
    select_from_span,
    shard_dir_name,
    substring_char_span,
    token_span_for_char_span,
)


def test_substring_char_span_case_insensitive():
    assert substring_char_span("ACE inhibitors can cause cough", "ace inhibitors") == (0, 14)


def test_substring_char_span_occurrence():
    assert substring_char_span("pain, no pain, more pain", "pain", occurrence=1) == (9, 13)


def test_substring_char_span_last_occurrence():
    text = "The answer is <diagnosis>. The answer is Pneumonia."
    start = text.rfind("The answer is")
    assert substring_char_span(text, "The answer is", occurrence=-1) == (
        start,
        start + len("The answer is"),
    )


def test_substring_char_span_rejects_other_negative_occurrences():
    with pytest.raises(ValueError, match="must be -1"):
        substring_char_span("pain", "pain", occurrence=-2)


def test_token_span_for_char_span_overlap():
    offsets = [(0, 0), (0, 3), (3, 7), (8, 12), (12, 15)]
    assert token_span_for_char_span(offsets, 2, 10) == (1, 4)


def encoding(text: str, offsets: list[tuple[int, int]]):
    return {"text": text, "offset_mapping": offsets, "n_tokens": len(offsets)}


def test_last_token_uses_the_unpadded_length():
    """Right padding means the real final token keeps its unpadded index."""
    encoded = encoding("abc", [(0, 1), (1, 2), (2, 3)])
    span, selections = resolve_positions({"id": "r", "position_mode": "last_token"}, encoded, {})
    assert span == (2, 3)
    assert selections == ["last_token"]


def test_target_text_returns_no_selection_so_the_caller_chooses():
    """target_text rows can carry several reductions; the span is what is fixed."""
    encoded = encoding("no fever", [(0, 2), (3, 8)])
    row = {"id": "r", "position_mode": "target_text", "target_text": "fever"}
    span, selections = resolve_positions(row, encoded, {})
    assert span == (1, 2)
    assert selections == []
    assert row["target_char_span"] == [3, 8]


def test_token_index_outside_the_sequence_is_refused():
    encoded = encoding("abc", [(0, 1), (1, 2), (2, 3)])
    row = {"id": "r", "position_mode": "token_index", "target_token_position": 9}
    with pytest.raises(ValueError, match="outside"):
        resolve_positions(row, encoded, {})


def test_group_by_prompt_collapses_rows_sharing_a_forward_pass():
    rows = [
        {"id": "a", "prompt": "P1"},
        {"id": "b", "prompt": "P2"},
        {"id": "c", "prompt": "P1"},
    ]
    groups = group_by_prompt(rows)
    assert list(groups) == ["P1", "P2"]
    assert [r["id"] for r in groups["P1"]] == ["a", "c"]


def test_group_by_prompt_refuses_a_row_without_a_prompt():
    with pytest.raises(ValueError, match="no prompt"):
        group_by_prompt([{"id": "a"}])


def test_group_by_prompt_collapses_rows_sharing_teacher_forced_messages():
    messages = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "The answer is Pneumonia."},
    ]
    rows = [
        {"id": "p1", "prompt": "question", "chat_messages": messages},
        {"id": "p2", "prompt": "question", "chat_messages": messages},
    ]
    groups = group_by_prompt(rows)
    assert len(groups) == 1
    assert [row["id"] for row in next(iter(groups.values()))] == ["p1", "p2"]


def test_shard_dir_name_is_stable_and_bounded():
    """Resuming must rewrite the same paths, so the shard cannot depend on order."""
    assert shard_dir_name("ddxplus_x__cuepos00", 256) == shard_dir_name(
        "ddxplus_x__cuepos00", 256
    )
    names = {shard_dir_name(f"row_{i}", 8) for i in range(200)}
    assert names <= {f"shard_{i:03d}" for i in range(8)}
    assert len(names) == 8


def test_select_from_span_reductions():
    torch = pytest.importorskip("torch")
    seq = torch.arange(24, dtype=torch.float32).reshape(6, 4)

    first, pos = select_from_span(seq, (2, 5), "first_subtoken")
    assert pos == "2" and torch.equal(first, seq[2])

    last, pos = select_from_span(seq, (2, 5), "last_subtoken")
    assert pos == "4" and torch.equal(last, seq[4])

    mean, pos = select_from_span(seq, (2, 5), "span_mean")
    assert pos == "2:5" and torch.equal(mean, seq[2:5].mean(dim=0))

    whole, pos = select_from_span(seq, (2, 5), "span")
    assert pos == "2:5" and whole.shape == (3, 4)


def test_select_from_span_rejects_an_unknown_selection():
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="Unsupported selection"):
        select_from_span(torch.zeros(3, 2), (0, 2), "median")
