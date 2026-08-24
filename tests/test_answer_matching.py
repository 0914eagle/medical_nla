from src.answer_matching import content_tokens, is_correct, normalize, token_f1


def test_normalize_folds_case_spacing_and_trailing_period():
    assert normalize("  Acute Otitis  Media. ") == "acute otitis media"


def test_is_correct_matches_containment_in_either_direction():
    # A model may answer more or less specifically than the gold label; both
    # name the same condition.
    assert is_correct("acute otitis media", "Otitis media", [])
    assert is_correct("Otitis media", "Acute otitis media", [])
    assert is_correct("URTI", "Viral URTI", ["URTI"])


def test_is_correct_rejects_a_different_condition_or_no_answer():
    assert not is_correct("Pneumonia", "Otitis media", [])
    assert not is_correct(None, "Otitis media", [])
    assert not is_correct("", "Otitis media", [])


def test_containment_misses_an_interposed_word():
    """The failure token_f1 exists to measure, stated as a test."""
    assert not is_correct("phototoxic drug reaction", "phototoxic reaction", [])
    assert token_f1("phototoxic drug reaction", "phototoxic reaction", []) > 0.7


def test_token_f1_penalizes_padding_and_truncation_alike():
    assert round(token_f1("reaction", "phototoxic reaction", []), 2) == 0.67
    assert round(token_f1("phototoxic drug allergic reaction", "phototoxic reaction", []), 2) == 0.67
    assert token_f1("phototoxic reaction", "phototoxic reaction", []) == 1.0
    assert token_f1("Pneumonia", "Otitis media", []) == 0.0


def test_token_f1_takes_the_best_accepted_name():
    assert token_f1("viral URTI", "Something else", ["viral urti"]) == 1.0


def test_content_tokens_keep_words_that_change_the_diagnosis():
    """Dropping 'acute' or 'left' to raise a match rate would score a
    different question, so only true fillers are removed."""
    assert content_tokens("acute renal failure") == {"acute", "renal", "failure"}
    assert content_tokens("inflammation of the left kidney") == {
        "inflammation",
        "left",
        "kidney",
    }


def test_token_f1_treats_a_hyphen_as_a_word_boundary():
    assert token_f1("drug-induced reaction", "drug induced reaction", []) == 1.0


def test_token_f1_without_an_answer_is_zero():
    assert token_f1(None, "Otitis media", []) == 0.0
    assert token_f1("the of", "Otitis media", []) == 0.0


def test_readable_label_recovers_names_stored_as_identifiers():
    """The forms actually seen in MedCaseReasoning's final_diagnosis field."""
    from src.answer_matching import readable_diagnosis_label as readable

    assert readable("Scleroderma_renal_crisis") == "Scleroderma renal crisis"
    assert readable("AmeloblasticFibroma") == "Ameloblastic Fibroma"
    assert readable("ThyroidFollicularRenalCellCarcinoma") == (
        "Thyroid Follicular Renal Cell Carcinoma"
    )
    assert readable("PFAPAsyndrome") == "PFAPA syndrome"


def test_readable_label_leaves_prose_and_short_suffixes_alone():
    from src.answer_matching import readable_diagnosis_label as readable

    assert readable("SBP-101 induced retinal toxicity") == "SBP-101 induced retinal toxicity"
    assert readable("Guillain-Barre syndrome") == "Guillain-Barre syndrome"
    assert readable("HDR syndrome") == "HDR syndrome"
    # "PEC oma" would be etymology, not a name.
    assert readable("PEComa") == "PEComa"


def test_a_recovered_label_scores_the_answer_it_should():
    """Underscores fold in normalization; a CamelCase run does not, which is
    what the label recovery is for."""
    from src.answer_matching import readable_diagnosis_label as readable

    assert is_correct("scleroderma renal crisis", "Scleroderma_renal_crisis", [])
    assert not is_correct("ameloblastic fibroma", "AmeloblasticFibroma", [])
    assert is_correct("ameloblastic fibroma", readable("AmeloblasticFibroma"), [])


def test_markup_typography_and_accents_do_not_cost_a_match():
    assert is_correct("**Erythema Multiforme**", "erythema multiforme", [])
    assert is_correct("Guillain-Barre syndrome", "Guillain-Barré syndrome", [])
    assert is_correct("Whipple's disease", "Whipple’s disease", [])
    assert is_correct("vitamin D-dependent rickets", "vitamin D–dependent rickets", [])


def test_the_possessive_on_an_eponym_is_house_style_not_a_claim():
    """Both spellings are in use and name one condition; the apostrophe was
    surviving normalization as a stray "s" token and blocking the match."""
    assert is_correct("Eagle syndrome", "Eagle’s syndrome", [])
    assert is_correct("Gitelman syndrome", "Gitelman's syndrome", [])
    assert is_correct("Wernicke's encephalopathy", "Wernicke encephalopathy", [])
    assert normalize("Eagle’s syndrome") == "eagle syndrome"


def test_stripping_the_possessive_does_not_merge_different_eponyms():
    assert not is_correct("Down syndrome", "Turner's syndrome", [])


def test_british_spelling_is_orthography_not_disagreement():
    """A published case report writes "oedema"; a US-trained model writes
    "edema". Neither is a claim about the diagnosis."""
    assert is_correct("Acute pulmonary edema", "Acute pulmonary oedema", [])
    assert is_correct("anemia", "anaemia", [])
    assert is_correct("esophageal rupture", "oesophageal rupture", [])


def test_spelling_folding_does_not_maul_unrelated_words():
    """Folding "ae"/"oe" wholesale would rewrite these; the list is explicit."""
    assert normalize("aerosol") == "aerosol"
    assert normalize("anaesthesia") == "anaesthesia"


def test_a_short_alias_does_not_match_inside_a_longer_word():
    """"pe" is an alias of pulmonary embolism and sits inside "pericarditis".

    Plain containment scored every pulmonary-embolism case answered
    "Pericarditis" as correct, and put 26 of them into the adoption count. The
    same collision was fixed once in gold_is_written_in, where "PE" matched
    inside "the posterior as-pe-ct of the ankle", and was missed here -- in the
    function that scores every answer in the paper.
    """
    from src.ddxplus_aliases import aliases_for

    assert not is_correct("Pericarditis", "Pulmonary embolism",
                          aliases_for("Pulmonary embolism"))
    assert is_correct("PE", "Pulmonary embolism", aliases_for("Pulmonary embolism"))


def test_containment_does_not_cross_a_clinical_opposite():
    """"stable angina" sits inside "unstable angina", which is the other disease."""
    assert not is_correct("Unstable angina", "Stable angina", [])
    assert is_correct("Stable angina", "Stable angina", [])
    # Croup is not bronchitis, however the letters fall.
    assert not is_correct("Laryngotracheobronchitis (Croup)", "Bronchitis", [])


def test_legitimate_containment_survives_the_boundary_rule():
    assert is_correct("Acute bronchitis", "Bronchitis", [])
    assert is_correct("Otitis media", "Acute otitis media", [])
    assert is_correct("Iron Deficiency Anemia", "Anemia", [])
