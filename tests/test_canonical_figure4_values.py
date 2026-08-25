import json
import sys

from scripts.build_canonical_figure4_values import main


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return str(path)


def test_figure4_builder_uses_one_canonical_cohort_for_both_panels(tmp_path, monkeypatch):
    answers = []
    channels = []
    probes = []
    monitors = []
    # Four eligible cases provide a within-diagnosis contrast. The fifth is
    # present in every downstream artifact but must be removed because its
    # canonical no-note answer is wrong.
    for index, moved in enumerate((False, False, True, True, True)):
        base_id = f"case-{index}"
        eligible = index < 4
        for arm in ("none", "wrong", "correct"):
            answers.append({
                "id": f"{base_id}-{arm}",
                "base_id": base_id,
                "hint_variant": arm,
                "diagnosis_name": "Pneumonia",
                "hint_diagnosis_name": "Bronchitis" if arm == "wrong" else None,
                "answer": "Pneumonia" if arm != "wrong" or not moved else "Tuberculosis",
                "source_correct": eligible if arm == "none" else not moved,
            })
        signal = float(moved)
        channels.append({
            "base_id": base_id,
            "diagnosis_name": "Pneumonia",
            "moved": moved,
            "answer_is_suggestion": False,
            "answer states the suspicion (output-only)": signal,
            "chain dwells on the suspicion": signal,
            "answer omits the internal conclusion (containment)": signal,
        })
        probes.append({
            "base_id": base_id,
            "diagnosis_name": "Pneumonia",
            "moved": moved,
            "answer_is_suggestion": False,
            "probe_flag": moved,
        })
        monitors.append({
            "id": base_id,
            "response": f"P = {0.9 if moved else 0.1}",
        })

    answer_path = write_jsonl(tmp_path / "answers.jsonl", answers)
    channel_path = write_jsonl(tmp_path / "channels.jsonl", channels)
    probe_path = write_jsonl(tmp_path / "probe.jsonl", probes)
    monitor_path = write_jsonl(tmp_path / "monitor.jsonl", monitors)
    rung_paths = []
    for rung in (3, 4, 5, 6):
        rows = []
        for index, moved in enumerate((False, False, True, True, True)):
            rows.append({
                "base_id": f"case-{index}",
                "ladder_rung": rung,
                "moved": moved,
                "first_correct": not moved,
                "source_correct": rung >= 5 if moved else rung <= 4,
            })
        rung_paths.append(write_jsonl(tmp_path / f"r{rung}.jsonl", rows))

    output = tmp_path / "values.json"
    summary = tmp_path / "summary.md"
    monkeypatch.setattr(sys, "argv", [
        "build_canonical_figure4_values.py",
        "--answers", answer_path,
        "--channel-scores", channel_path,
        "--monitor", monitor_path,
        "--probe-verdicts", probe_path,
        "--rungs", *rung_paths,
        "--output-json", str(output),
        "--summary-md", str(summary),
    ])
    main()

    values = json.loads(output.read_text(encoding="utf-8"))
    assert values["cohort"]["source_rows"] == 5
    assert values["cohort"]["eligible_rows"] == 4
    assert values["detection"]["n_all"] == 4
    assert values["detection"]["n_moved"] == 2
    assert values["detection"]["n_silent_moved"] == 2
    assert values["correction"]["n_all"] == 4
    assert values["correction"]["n_moved"] == 2
    assert values["detection"]["all"] == [1.0] * 5
    assert values["correction"]["overall"] == [0.5, 0.5, 0.5, 0.5, 0.5]
    assert values["correction"]["moved"] == [0.0, 0.0, 0.0, 1.0, 1.0]
