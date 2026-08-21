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


# Real heldout rows, with the verdict a human gave them. The scorer's job is
# only to sort; these pin what it does to readings already known to be right.
HAND_LABELLED = [
    ("A", "a fever (either felt or measured with a thermometer)",
     "had a fever (defined as 100F or higher)"),
    ("A", "coughing up blood", "recently had a cough that produced blood"),
    ("A", "had chills or shivers",
     "they feel like they are shivering or have muscle spasms"),
    ("C", "an itchy nose or an itchy back of the throat",
     "they ever felt like there was something in the upper gum or bottom lip of the throat"),
    ("C", "find that their symptoms have worsened over the last 2 weeks and that "
          "progressively less effort is required to cause the symptoms",
     "their symptoms are worse when they are still and alleviated when moving"),
]


def test_the_questionnaire_gloss_no_longer_sinks_a_correct_reading():
    """Eleven of the twenty-five lowest-scoring heldout rows were this one cue,
    read correctly each time and scored 0.22 for not repeating the survey's
    own parenthetical."""
    _, gold, read = HAND_LABELLED[0]
    assert overlap_f1(read, gold) == 1.0


def test_inflection_no_longer_sinks_a_correct_reading():
    _, gold, read = HAND_LABELLED[1]
    assert overlap_f1(read, gold) > 0.5


def test_wrong_readings_stay_low():
    for verdict, gold, read in HAND_LABELLED:
        if verdict == "C":
            assert overlap_f1(read, gold) < 0.25


def test_no_threshold_separates_right_from_wrong():
    """The reason the labelling cannot be replaced by a cutoff, kept as an
    executable fact rather than a claim in prose: a correct reading of
    "undergo dialysis" as "have to dialyze" shares no stem with the gold and
    scores zero, below every wrong reading here."""
    correct_but_zero = overlap_f1("have to dialyze", "undergo dialysis")
    worst_wrong = max(
        overlap_f1(read, gold) for verdict, gold, read in HAND_LABELLED if verdict == "C"
    )
    assert correct_but_zero < worst_wrong
