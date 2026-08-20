import pytest

from src.config import load_config


def write(tmp_path, text):
    path = tmp_path / "c.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_a_placeholder_resolves_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDICAL_NLA_DATA_ROOT", "/data9/somebody")
    cfg = load_config(write(tmp_path, "paths:\n  result_dir: ${MEDICAL_NLA_DATA_ROOT}/x/results\n"))
    assert cfg["paths"]["result_dir"] == "/data9/somebody/x/results"


def test_a_known_placeholder_has_a_default(tmp_path, monkeypatch):
    monkeypatch.delenv("MEDICAL_NLA_DATA_ROOT", raising=False)
    cfg = load_config(write(tmp_path, "paths:\n  result_dir: ${MEDICAL_NLA_DATA_ROOT}/x\n"))
    assert cfg["paths"]["result_dir"] == "/data1/heejae/x"


def test_an_unknown_placeholder_raises_rather_than_becoming_a_directory(tmp_path):
    """Left literal it becomes a directory named "${VAR}" that a run writes
    into and nobody finds again."""
    with pytest.raises(KeyError, match="NO_SUCH_VAR"):
        load_config(write(tmp_path, "paths:\n  result_dir: ${NO_SUCH_VAR}/x\n"))


def test_substitution_reaches_nested_values(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDICAL_NLA_DATA_ROOT", "/d")
    cfg = load_config(
        write(tmp_path, "a:\n  b:\n    - ${MEDICAL_NLA_DATA_ROOT}/one\n    - plain\n  n: 3\n")
    )
    assert cfg["a"]["b"] == ["/d/one", "plain"]
    assert cfg["a"]["n"] == 3


def test_the_shipped_configs_all_resolve(monkeypatch):
    """A config that cannot be loaded is found at the start of a long job."""
    import glob

    monkeypatch.setenv("MEDICAL_NLA_DATA_ROOT", "/data1/heejae")
    for path in sorted(glob.glob("configs/*.yaml")):
        cfg = load_config(path)
        assert "${" not in str(cfg["paths"]), path
