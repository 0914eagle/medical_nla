"""The two heldout rows that made the v2 rule unusable, and what replaced it.

Both are real first-run output from the L24 seed-17 adapter, and both are
correct or near-correct readings of a cue string that was never supervised.
The literal-containment rule scores both zero.
"""

from src.cue_readout_scoring import (
    FUNCTION_WORDS,
    exact_containment,
    observed_items,
    overlap_f1,
    score_readout,
)

PARAPHRASE = (
    "they feel that their eyes produce excessive tears",
    "their eyes produce too many tears",
)
PARTIAL = (
    "slightly dizzy or lightheaded",
    "they do feel like they are about to faint or are lightheaded",
)




def test_the_literal_rule_scores_a_correct_paraphrase_zero():
    """Why this module exists rather than a threshold on the old scorer."""
    gold, read = PARAPHRASE
    assert exact_containment(read, gold) is False
    assert overlap_f1(read, gold) > 0.5


def test_a_partial_reading_lands_between_right_and_wrong():
    """And lands squarely on any threshold anyone would pick.

    "lightheaded" is read and "dizzy" is not, while "about to faint" is added:
    one content word shared out of two on each side, which is 0.5 exactly. That
    is the case for not treating overlap as the metric -- whether this counts
    as reading the cue is a judgement about medicine, and no cutoff makes it
    for us. Overlap sorts the rows; something that understands the words has to
    decide them."""
    assert overlap_f1(*reversed(PARTIAL)) == 0.5
    assert overlap_f1("they have a cough", PARTIAL[0]) < 0.5 < overlap_f1(*reversed(PARAPHRASE))


def test_an_unrelated_reading_scores_zero():
    assert overlap_f1("they have a cough", "slightly dizzy or lightheaded") == 0.0


def test_only_grammatical_classes_are_discarded():
    """A frequency cutoff was tried first and would delete "pain" from a
    chest-pain corpus. Function words are safe to fix in advance; clinical
    words are never decided here."""
    for word in ("they", "do", "have", "the"):
        assert word in FUNCTION_WORDS
    for word in ("pain", "tears", "dizzy", "lightheaded", "fever", "cough"):
        assert word not in FUNCTION_WORDS


def test_padding_the_readout_is_not_rewarded():
    """F1, not recall: naming every cue in the corpus must not score well."""
    gold = "slightly dizzy or lightheaded"
    padded = "dizzy lightheaded cough fever chest pain tears vomiting rash"
    assert overlap_f1(padded, gold) < overlap_f1("dizzy lightheaded", gold)


def test_bullets_split_into_items_but_hyphenated_words_survive():
    assert observed_items("- fever\n- chest pain") == ["fever", "chest pain"]
    assert observed_items("- x-ray changes") == ["x-ray changes"]


def test_the_best_item_wins_when_a_readout_emits_several():
    result = score_readout("- a cough\n- their eyes produce too many tears", PARAPHRASE[0])
    assert result["n_items"] == 2
    assert result["best_item"] == "their eyes produce too many tears"
    assert result["f1"] > 0.5


def test_an_empty_readout_scores_zero_rather_than_raising():
    assert score_readout("", "slightly dizzy")["f1"] == 0.0


VANILLA = (
    "they feel that their eyes produce excessive tears",
    'Excessive lacrimation/tears\nThe phrase "eyes produce excessive tears" '
    "establishes a symptom description, likely continuing with dry eye syndrome "
    "or conjunctivitis context.\nNo additional elaboration or conclusion is "
    "needed; the sentence ends a factual observation about the physiological "
    "symptom, implying a structured medical description or a transition to the "
    'topic sentence. Final token "tears" immediately expects a continuation.',
)


def test_a_rambling_baseline_is_credited_for_reading_and_penalised_for_length():
    """Real vanilla AV output, which names the finding and then writes on about
    what token comes next. Judged by F1 alone it looks unable to read the
    vector; the two failures have to stay separate, because which one is true
    decides what the adapter is claimed to add."""
    gold, read = VANILLA
    result = score_readout(read, gold)
    assert result["output_recall"] > 0.5, "the finding is in the output"
    assert result["output_precision"] < 0.2, "and so is a great deal else"
    assert result["f1"] > 0.5, "one line of it is a clean reading"
    assert result["n_chars"] > 400


def test_the_adapter_output_scores_well_on_both():
    gold, read = PARAPHRASE
    result = score_readout(read, gold)
    assert result["output_recall"] > 0.5
    assert result["output_precision"] > 0.5
