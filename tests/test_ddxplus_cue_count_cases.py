import pytest

from scripts.make_ddxplus_cue_count_cases import make_prompt, parse_cue_count


def test_parse_cue_count_reads_integers():
    assert parse_cue_count("3") == 3
    assert parse_cue_count("12") == 12


def test_parse_cue_count_maps_all_forms_to_none():
    # None means "take every cue the case has", which imposes no minimum.
    for token in ("all", "ALL", " full ", "max"):
        assert parse_cue_count(token) is None


def test_parse_cue_count_rejects_non_positive():
    with pytest.raises(ValueError):
        parse_cue_count("0")
    with pytest.raises(ValueError):
        parse_cue_count("-2")


def test_min_required_has_no_floor_when_only_all_is_requested():
    # Regression: `--cue-counts all` used to raise on max() of an empty
    # sequence, since "all" contributes no explicit count.
    cue_counts = [parse_cue_count("all")]
    explicit = [count for count in cue_counts if count is not None]
    assert explicit == []
    assert (max(explicit) if explicit else 1) == 1


def test_min_required_uses_the_largest_explicit_count():
    cue_counts = [parse_cue_count(token) for token in ("3", "5", "all")]
    explicit = [count for count in cue_counts if count is not None]
    assert (max(explicit) if explicit else 1) == 5


def test_make_prompt_lists_each_finding_on_its_own_line():
    # A list frame takes clauses ("the rash is swollen") as well as noun
    # phrases, which an inline "presents with X, Y and Z" sentence cannot, and
    # the line break keeps cues containing commas from blurring the boundary.
    prompt = make_prompt(["fever", "the rash is swollen", "chest pain, worse at night"])
    assert "A patient presents with the following findings:\n" in prompt
    assert "\n- fever\n" in prompt
    assert "\n- the rash is swollen\n" in prompt
    assert "\n- chest pain, worse at night\n" in prompt
    assert prompt.rstrip().endswith('You MUST end your response with exactly "The answer is <diagnosis>."')


def test_make_prompt_keeps_every_cue_verbatim():
    # Extraction resolves cues by substring, so the frame must not alter them.
    cues = ["fever", "has not traveled out of the country", "the pain is located in the chest"]
    prompt = make_prompt(cues)
    assert all(cue in prompt for cue in cues)


def test_make_prompt_puts_the_question_after_the_findings():
    # Causal attention means cue positions cannot see what follows them, so the
    # same activations serve every instruction that is appended here.
    prompt = make_prompt(["fever"])
    assert prompt.index("- fever") < prompt.index("What is the single most likely diagnosis?")
    # Both conditions must share the presentation byte for byte, or the two arms
    # cannot share one extraction.
    cot = make_prompt(["fever"], condition="cot")
    shared = prompt.split("\n\nWhat is")[0]
    assert cot.startswith(shared)


def _patient_csv(tmp_path, rows):
    import csv

    path = tmp_path / "patients.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["PATHOLOGY", "EVIDENCES"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def _evidences(tmp_path):
    import json

    path = tmp_path / "evidences.json"
    path.write_text(
        json.dumps(
            {
                "E_COUGH": {"question_en": "Do you have a cough?", "is_antecedent": False},
                "E_FEVER": {"question_en": "Do you have a fever?", "is_antecedent": False},
            }
        ),
        encoding="utf-8",
    )
    return path


def _run(tmp_path, extra):
    import subprocess
    import sys

    tmp_path.mkdir(parents=True, exist_ok=True)
    out = tmp_path / "cases.jsonl"
    patients = _patient_csv(
        tmp_path,
        [{"PATHOLOGY": "Pneumonia", "EVIDENCES": "['E_COUGH', 'E_FEVER']"} for _ in range(50)],
    )
    result = subprocess.run(
        [
            sys.executable,
            "scripts/make_ddxplus_cue_count_cases.py",
            "--patients", str(patients),
            "--evidences", str(_evidences(tmp_path)),
            "--output", str(out),
            "--cue-counts", "all",
            "--max-diagnoses", "1",
            "--examples-per-diagnosis", "5",
            *extra,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.read_text(encoding="utf-8"), result.stdout


def test_stop_when_full_matches_the_full_scan(tmp_path):
    # Same output, without rendering the rows that cannot change it.
    full, _ = _run(tmp_path / "a", [])
    early, stdout = _run(tmp_path / "b", ["--stop-when-full"])
    assert early == full
    assert "stopping at row" in stdout


def test_full_scan_is_the_default(tmp_path):
    _, stdout = _run(tmp_path / "c", [])
    assert "stopping at row" not in stdout
