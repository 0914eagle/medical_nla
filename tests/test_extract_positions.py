import torch

from src.extract_activations import substring_char_span, token_span_for_char_span


def test_substring_char_span_case_insensitive():
    assert substring_char_span("ACE inhibitors can cause cough", "ace inhibitors") == (0, 14)


def test_substring_char_span_occurrence():
    assert substring_char_span("pain, no pain, more pain", "pain", occurrence=1) == (9, 13)


def test_token_span_for_char_span_overlap():
    offsets = torch.tensor([[[0, 0], [0, 3], [3, 7], [8, 12], [12, 15]]])
    assert token_span_for_char_span(offsets, 2, 10) == (1, 4)
