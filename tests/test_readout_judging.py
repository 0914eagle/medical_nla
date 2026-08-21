"""The judge is reached by pasting, so its reply is prose and must be checked.

A rate computed over part of a pool, or over numbers nobody asked about, is
not the pool's rate. These pin the refusals rather than the parsing.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "apply_readout_judgements.py"

from scripts.apply_readout_judgements import cohens_kappa, parse_verdicts

INDEX = [
    {"n": 1, "gold": "a cough", "read": "a cough", "ids": ["r1", "r2"]},
    {"n": 2, "gold": "coughing up blood", "read": "blood or clots", "ids": ["r3"]},
    {"n": 3, "gold": "a fever", "read": "intense muscle spasms", "ids": ["r4"]},
]


def write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def run(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True
    )


def test_verdicts_are_found_inside_whatever_prose_arrives():
    text = "Sure, here are the verdicts:\n\n1=A\n2 = B\n3. C\n\nLet me know if..."
    assert parse_verdicts(text) == {1: "A", 2: "B", 3: "C"}


def test_a_number_answered_twice_differently_is_fatal():
    with pytest.raises(SystemExit):
        parse_verdicts("1=A\n1=C\n")


def test_an_unanswered_request_refuses_rather_than_reporting_a_partial_rate(tmp_path):
    index = write(tmp_path / "i.jsonl", INDEX)
    verdicts = tmp_path / "v.txt"
    verdicts.write_text("1=A\n2=B\n")
    result = run("--index", index, "--verdicts", str(verdicts))
    assert result.returncode != 0
    assert "unanswered" in result.stderr


def test_a_verdict_for_something_never_asked_is_fatal(tmp_path):
    index = write(tmp_path / "i.jsonl", INDEX)
    verdicts = tmp_path / "v.txt"
    verdicts.write_text("1=A\n2=B\n3=C\n9=A\n")
    result = run("--index", index, "--verdicts", str(verdicts))
    assert result.returncode != 0
    assert "not asked" in result.stderr


def test_a_pair_verdict_reaches_every_row_that_pair_covers(tmp_path):
    index = write(tmp_path / "i.jsonl", INDEX)
    verdicts = tmp_path / "v.txt"
    verdicts.write_text("1=A\n2=B\n3=C\n")
    out = tmp_path / "out.jsonl"
    result = run("--index", index, "--verdicts", str(verdicts), "--output", str(out))
    assert result.returncode == 0, result.stderr
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == 4  # pair 1 covers two rows
    assert [r["verdict"] for r in rows if r["id"] in ("r1", "r2")] == ["A", "A"]
    assert "A + B    0.7500" in result.stdout


def test_kappa_is_zero_when_agreement_is_only_what_the_marginals_give():
    assert cohens_kappa([("A", "A")] * 10) != cohens_kappa([("A", "A"), ("B", "C")])
    perfect = cohens_kappa([("A", "A"), ("B", "B"), ("C", "C"), ("A", "A")])
    assert perfect == pytest.approx(1.0)


def test_the_judge_being_softer_than_the_human_is_counted(tmp_path):
    """A lenient judge inflates exactly the rate the paper claims."""
    index = write(tmp_path / "i.jsonl", INDEX)
    (tmp_path / "v.txt").write_text("1=A\n2=A\n3=A\n")
    labels = tmp_path / "hand.tsv"
    labels.write_text(
        "verdict\tpool\tgold\tread\n"
        "A\theldout\ta cough\ta cough\n"
        "B\theldout\tcoughing up blood\tblood or clots\n"
        "C\theldout\ta fever\tintense muscle spasms\n"
    )
    result = run(
        "--index", index,
        "--verdicts", str(tmp_path / "v.txt"),
        "--hand-labels", str(labels),
    )
    assert result.returncode == 0, result.stderr
    assert "judge harsher    0   judge softer 2" in result.stdout
