from scripts.audit_clinical_span_cases import audit_prose_cases


def case(case_id, cues, *, presentation=None, diagnosis="Pneumonia"):
    text = presentation or " ".join(cues)
    return {
        "id": case_id,
        "prompt": f"You are an expert physician. A patient presents as follows:\n\n{text}\n\nWhat?",
        "presentation": text,
        "diagnosis_name": diagnosis,
        "cue_targets": list(cues),
        "cue_is_boilerplate": ["unremarkable" in cue or "normal" in cue for cue in cues],
    }


def test_clean_corpus_passes_and_reports_diversity():
    cases = [case(f"c{i}", [f"a finding number {i}", f"another finding {i}"]) for i in range(20)]
    failures, summary = audit_prose_cases(cases)
    assert failures == []
    assert summary["distinct_cues"] == 40
    assert summary["cues_seen_once_rate"] == 1.0


def test_presentation_naming_the_diagnosis_is_a_hard_failure():
    cases = [case("c1", ["a finding here"], presentation="known Pneumonia with cough")]
    failures, _ = audit_prose_cases(cases)
    assert any("names the gold diagnosis" in failure for failure in failures)


def test_near_constant_cue_is_a_hard_failure():
    # The failure that removed DDXPlus's negatives: one cue in almost every case.
    cases = [case(f"c{i}", ["has not traveled recently", f"a finding number {i}"]) for i in range(20)]
    failures, summary = audit_prose_cases(cases)
    assert any("near-constant cue" in failure for failure in failures)
    assert summary["most_common_cue_rate"] == 1.0


def test_boilerplate_share_is_reported_not_failed():
    cases = [case(f"c{i}", [f"a finding number {i}", "examination was unremarkable"]) for i in range(20)]
    failures, summary = audit_prose_cases(cases)
    # Frequent, but reported rather than failed: a normal exam is a real finding.
    assert summary["boilerplate_cue_rate"] == 0.5
    assert not any("boilerplate" in failure for failure in failures)


def test_composition_checks_are_inherited_from_the_ddxplus_gate():
    broken = case("c1", ["a cue absent from the prompt"])
    broken["prompt"] = "You are an expert physician. A patient presents as follows:\n\nsomething else"
    failures, _ = audit_prose_cases([broken])
    assert any("not verbatim" in failure for failure in failures)
