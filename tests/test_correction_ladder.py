"""The second-pass prompts of the correction ladder."""

from scripts.make_correction_ladder_cases import readout_cues, rung_prompt
from src.case_prompts import DIRECT_INSTRUCTION

PRESENTATION = (
    "You are an expert physician. A 51-year-old woman presents with:\n"
    "- coughing up blood\n- a fever\n\nThe referring note suspects Bronchitis."
)


def test_every_rung_keeps_the_note_in_place():
    """Deployment does not get to delete the confounder; a rung that quietly
    removed the note would be measuring a different, easier problem."""
    for rung in (3, 4, 5, 6):
        prompt = rung_prompt(
            rung, PRESENTATION, "Silicosis", ["a fever"], "Tuberculosis", "",
            probe_class="Tuberculosis",
        )
        assert "The referring note suspects Bronchitis." in prompt
        assert "Your previous answer was: Silicosis." in prompt
        assert prompt.endswith(DIRECT_INSTRUCTION)


def test_the_rungs_differ_only_in_what_is_appended():
    r3 = rung_prompt(3, PRESENTATION, "Silicosis", ["a fever"], "Tuberculosis", "coughing blood")
    r4 = rung_prompt(4, PRESENTATION, "Silicosis", ["a fever"], "Tuberculosis", "coughing blood")
    r5 = rung_prompt(5, PRESENTATION, "Silicosis", ["a fever"], "Tuberculosis", "coughing blood")
    assert "patient's findings" in r4 and "patient's findings" not in r3
    assert "internal conclusion: Tuberculosis" in r5
    # r4 must NOT carry the readout and r5 must NOT carry the chart re-print:
    # each rung isolates one addition, or the comparison between them is mush.
    assert "internal conclusion" not in r4
    assert "patient's findings" not in r5


def test_rung_six_carries_the_probe_class_and_nothing_else():
    """r6 is the probe-content control: a class name with no phrasing and no
    grounds. It must use the probe's argmax, not the readout's conclusion --
    the two can disagree, and feeding the readout's word under a probe label
    would make r5 minus r6 measure nothing."""
    r6 = rung_prompt(
        6, PRESENTATION, "Silicosis", ["a fever"], "Tuberculosis",
        "coughing blood", probe_class="Pneumonia",
    )
    assert "classifier probe" in r6
    assert "predicts: Pneumonia" in r6
    assert "Tuberculosis" not in r6
    assert "internal conclusion" not in r6
    assert "coughing blood" not in r6
    assert "patient's findings" not in r6


def test_readout_cues_come_from_the_structured_field():
    text = "<readout><answer>TB</answer><supporting_cues>a; b</supporting_cues></readout>"
    assert readout_cues(text) == "a; b"
    assert readout_cues("no tags") == ""


def test_replacement_policy_swaps_only_where_flagged():
    """The free policy: conclusion replaces the answer only on flagged rows,
    so an unflagged case keeps its first-pass verdict even when the
    conclusion happens to be wrong."""
    from scripts.analyze_correction_ladder import conclusion_correct

    flagged_right = {
        "correction_flag": True,
        "first_correct": False,
        "readout_conclusion": "Tuberculosis",
        "diagnosis_name": "Tuberculosis",
        "diagnosis_aliases": ["TB"],
    }
    flagged_wrong = dict(flagged_right, readout_conclusion="Bronchitis")
    empty_conclusion = dict(flagged_right, readout_conclusion="")
    assert conclusion_correct(flagged_right)
    assert not conclusion_correct(flagged_wrong)
    # An empty readout must count as wrong, not as a vacuous containment match.
    assert not conclusion_correct(empty_conclusion)
    # Aliases go through the same containment matching as regular scoring.
    assert conclusion_correct(dict(flagged_right, readout_conclusion="TB"))


def test_mcnemar_counts_only_the_pairs_that_disagree():
    """A cell of 161 cases where the rungs differ on seven is seven
    observations, not 161. Treating it as 161 is how a null result gets
    reported as a finding."""
    from scripts.analyze_correction_ladder import mcnemar_exact

    assert mcnemar_exact(0, 0) == 1.0
    # Perfectly split discordance is maximally uninformative.
    assert mcnemar_exact(10, 10) == 1.0
    # A lopsided split is significant only once there are enough pairs.
    assert mcnemar_exact(0, 3) > 0.05
    assert mcnemar_exact(0, 6) < 0.05
    # Symmetric in its arguments: which rung won does not change the p-value.
    assert mcnemar_exact(2, 11) == mcnemar_exact(11, 2)
