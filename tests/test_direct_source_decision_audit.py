from scripts.audit_direct_source_decision_labels import coverage, matches, ontology


def test_ontology_mapping_is_unique_or_explicitly_ambiguous() -> None:
    rows = [
        {
            "canonical_pdd": "NSTEMI",
            "annotation_root_diagnosis": "NSTE-ACS",
            "folder_pdd": "NSTEMI",
            "disease_category": "Acute Coronary Syndrome",
        },
        {
            "canonical_pdd": "UA",
            "annotation_root_diagnosis": "UA",
            "folder_pdd": "UA",
            "disease_category": "Acute Coronary Syndrome",
        },
    ]
    pdds = ontology(rows, "canonical_pdd")
    categories = ontology(rows, "disease_category")

    assert matches("The answer is NSTE-ACS.", pdds) == ["NSTEMI"]
    assert matches("Acute coronary syndrome", categories) == [
        "Acute Coronary Syndrome"
    ]


def test_coverage_uses_only_unique_ontology_matches() -> None:
    train = [
        {
            "normalized_answer": "alpha",
            "pdd_matches": ["A"],
            "category_matches": ["X"],
        },
        {
            "normalized_answer": "ambiguous",
            "pdd_matches": ["A", "B"],
            "category_matches": [],
        },
    ]
    validation = [
        {
            "normalized_answer": "alpha",
            "pdd_matches": ["A"],
            "category_matches": ["X"],
        },
        {
            "normalized_answer": "new",
            "pdd_matches": ["B"],
            "category_matches": [],
        },
    ]

    assert coverage(train, validation, "normalized_answer")["rate"] == 0.5
    assert coverage(train, validation, "pdd") == {
        "covered": 1,
        "n": 2,
        "rate": 0.5,
    }
    assert coverage(train, validation, "category") == {
        "covered": 1,
        "n": 1,
        "rate": 1.0,
    }
