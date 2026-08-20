from src.answer_matching import is_correct
from src.ddxplus_aliases import aliases_for, diagnoses_without_aliases, unknown_alias_keys


def test_an_abbreviated_label_accepts_what_a_model_writes_out():
    """"URTI" and "upper respiratory tract infection" share no words, so
    containment alone scores a correct answer wrong."""
    assert is_correct("Upper respiratory tract infection", "URTI", aliases_for("URTI"))
    assert is_correct("Paroxysmal supraventricular tachycardia", "PSVT", aliases_for("PSVT"))


def test_the_typo_in_the_label_is_corrected():
    """DDXPlus stores "Larygospasm"; it cost 4 of 100 answers."""
    assert is_correct("Laryngospasm", "Larygospasm", aliases_for("Larygospasm"))


def test_the_neoplasm_labels_accept_the_malignancy():
    """The measured case: 30 of 100 answers were "Pancreatic Cancer" or
    "Pancreatic adenocarcinoma"."""
    for answer in ("Pancreatic Cancer", "Pancreatic adenocarcinoma"):
        assert is_correct(answer, "Pancreatic neoplasm", aliases_for("Pancreatic neoplasm"))


def test_an_adjacent_condition_is_not_accepted():
    """63 answers said "Heart failure" for acute pulmonary edema. It is a
    different DDXPlus class -- and "heart failure" is itself a cue string in
    these prompts, so accepting it would score reading a finding back out."""
    assert not is_correct(
        "Heart failure", "Acute pulmonary edema", aliases_for("Acute pulmonary edema")
    )
    assert not is_correct(
        "Allergic rhinitis", "Allergic sinusitis", aliases_for("Allergic sinusitis")
    )


def test_a_label_with_no_entry_gets_no_aliases():
    assert aliases_for("Croup") == []
    assert aliases_for("") == []


def test_a_key_matching_no_label_is_reported():
    """A key written from memory would otherwise read as coverage it does not
    provide."""
    assert "urti" in unknown_alias_keys(["Croup", "Pneumonia"])
    assert unknown_alias_keys(list(_all_keys())) == []


def _all_keys():
    from src.ddxplus_aliases import _ALIASES

    return _ALIASES.keys()


def test_labels_without_aliases_are_listed_for_review():
    assert diagnoses_without_aliases(["Croup", "URTI"]) == ["Croup"]
