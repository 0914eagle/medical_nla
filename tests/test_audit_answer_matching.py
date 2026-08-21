"""The audit recomputes accuracy; these pin what it does when it disagrees.

It read DDXPlus at 0.2920 against the 0.3724 the run had recorded, and the gap
looked like a scoring finding until it turned out the answer rows carried no
diagnosis_aliases and the audit had simply been scoring without the alias
table. A second plausible number is worse than no number, so the disagreement
is now fatal.
"""

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "audit_answer_matching.py"


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return str(path)


def run(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True
    )


ANSWERS = [
    {
        "id": f"c{i}",
        "diagnosis_name": "Acute COPD exacerbation / infection",
        "answer": "Acute Exacerbation of COPD",
        "source_correct": True,
    }
    for i in range(5)
]
CASES = [
    {"id": f"c{i}", "diagnosis_aliases": ["Acute exacerbation of COPD"]} for i in range(5)
]


def test_it_refuses_when_it_disagrees_with_the_run_that_wrote_the_file(tmp_path):
    result = run("--answers", write_jsonl(tmp_path / "a.jsonl", ANSWERS))
    assert result.returncode != 0
    assert "score differently here than the run recorded" in result.stderr
    # The remedy has to be in the message, or the next reader re-derives it.
    assert "--cases" in result.stderr


def test_aliases_join_from_the_case_file_and_restore_agreement(tmp_path):
    result = run(
        "--answers",
        write_jsonl(tmp_path / "a.jsonl", ANSWERS),
        "--cases",
        write_jsonl(tmp_path / "c.jsonl", CASES),
    )
    assert result.returncode == 0, result.stderr
    assert "strict_accuracy: 1.0000" in result.stdout


def test_aliases_carried_on_the_answer_row_need_no_case_file(tmp_path):
    rows = [dict(row, diagnosis_aliases=["Acute exacerbation of COPD"]) for row in ANSWERS]
    result = run("--answers", write_jsonl(tmp_path / "a.jsonl", rows))
    assert result.returncode == 0, result.stderr
    assert "rows with an alias list: 5 / 5" in result.stdout


def test_a_repeated_disagreement_prints_once_with_its_count(tmp_path):
    """DDXPlus's 49 names mean one pair can be hundreds of rows. Printed flat,
    a --show of 60 spent all sixty lines on a single pair."""
    rows = [
        {
            "id": f"c{i}",
            "diagnosis_name": "Acute dystonic reactions",
            "answer": "Serotonin Syndrome",
            "diagnosis_aliases": [],
        }
        for i in range(40)
    ]
    result = run("--answers", write_jsonl(tmp_path / "a.jsonl", rows), "--show", "10")
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("Serotonin Syndrome") == 1
    assert "40 rows" in result.stdout
    assert "1 distinct (gold, answer) pairs" in result.stdout
