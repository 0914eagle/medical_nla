"""Telling a laterality flip apart from a paraphrase.

The external judge graded 'swelling located thigh(L)' against 'thigh(R)' as B
-- right finding, wrong attribute -- where the hand pass said C. Both readings
are arguable under the rubric, which is exactly why the split has to be
counted: a scorer's margin made of laterality is a different claim from one
made of paraphrase.
"""

from __future__ import annotations

from scripts.analyze_readout_semantic_judgements import differs_only_by_site, norm


def test_laterality_flip_is_site_only():
    assert differs_only_by_site(
        "where is the swelling located thigh(l)",
        "where is the swelling located thigh(r)",
    )


def test_site_flip_is_site_only():
    assert differs_only_by_site(
        "where is the swelling located dorsal aspect of the foot(r)",
        "where is the swelling located lateral side of the foot(r)",
    )
    assert differs_only_by_site(
        "where is the affected region located bottom lip(r)",
        "where is the affected region located upper lip(r)",
    )


def test_a_different_finding_is_not_site_only():
    """Melena against hematochezia shares words but names another finding."""
    assert not differs_only_by_site(
        "recently had stools that were black like coal",
        "light red blood or blood clots in their stool which is black and tarry",
    )


def test_paraphrase_is_not_site_only():
    assert not differs_only_by_site(
        "bouts of choking or shortness of breath that wake the patient up at night",
        "are their symptoms more prominent at night",
    )


def test_identical_text_is_not_counted():
    """No difference at all is not a site difference; it is agreement."""
    assert not differs_only_by_site("how severe is the itching", "how severe is the itching")


def test_extra_clinical_word_defeats_the_rule():
    """One real word beside the site word means the finding also changed."""
    assert not differs_only_by_site(
        "where is the swelling located thigh(l)",
        "where is the numbness located thigh(r)",
    )


def test_norm_strips_the_bullet_the_hand_file_kept():
    assert norm("- how bad is the itching") == "how bad is the itching"
    assert norm("  how   bad  is it ") == "how bad is it"
