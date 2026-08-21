"""Taking part of a diagnosis-grouped corpus.

The bug this closes was found by its own impossible result: --limit 200 on a
file holding a hundred cases per diagnosis left two labels, the candidate set
was collected from those rows, and the resulting two-way choice scored top1
200/200 against a source model that answers these cases at 0.3724.
"""

from src.sampling import distinct_values, sample_rows

# A hundred cases per label, in blocks, which is how every case file in this
# project is written.
GROUPED = [
    {"id": f"c{i}", "diagnosis_id": f"dx_{i // 100}"} for i in range(490)
]


def test_taking_the_front_of_a_grouped_file_covers_two_labels():
    """Not a claim about the helper -- a claim about the data, kept here so the
    reason the helper exists cannot quietly stop being true."""
    assert distinct_values(GROUPED[:200], "diagnosis_id") == 2
    assert distinct_values(GROUPED, "diagnosis_id") == 5


def test_sampling_covers_the_label_space():
    sampled = sample_rows(GROUPED, 200, seed=17, announce=False)
    assert len(sampled) == 200
    assert distinct_values(sampled, "diagnosis_id") == 5


def test_the_same_seed_selects_the_same_rows():
    first = sample_rows(GROUPED, 50, seed=17, announce=False)
    second = sample_rows(GROUPED, 50, seed=17, announce=False)
    assert [row["id"] for row in first] == [row["id"] for row in second]
    other = sample_rows(GROUPED, 50, seed=18, announce=False)
    assert [row["id"] for row in first] != [row["id"] for row in other]


def test_no_limit_and_an_oversized_limit_both_return_everything():
    assert len(sample_rows(GROUPED, None, announce=False)) == len(GROUPED)
    assert len(sample_rows(GROUPED, 10_000, announce=False)) == len(GROUPED)


def test_the_input_list_is_not_reordered_in_place():
    """Callers keep using the original list -- for the candidate set, among
    other things -- so sampling must not shuffle it underneath them."""
    rows = list(GROUPED)
    sample_rows(rows, 50, seed=17, announce=False)
    assert [row["id"] for row in rows] == [row["id"] for row in GROUPED]


def test_it_announces_the_size_and_seed(capsys):
    sample_rows(GROUPED, 50, seed=17, label="cases")
    printed = capsys.readouterr().out
    assert "50 of 490 cases" in printed
    assert "seed 17" in printed
