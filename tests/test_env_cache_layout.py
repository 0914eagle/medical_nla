import subprocess
from pathlib import Path

ENV_SH = Path("scripts/env.sh")
BOOTSTRAP = Path("scripts/bootstrap_server.sh")


def sourced(script: Path, extra: str = "") -> dict[str, str]:
    out = subprocess.run(
        ["bash", "-c", f"{extra} source {script} >/dev/null 2>&1; env"],
        capture_output=True, text=True, check=True,
    ).stdout
    return dict(line.split("=", 1) for line in out.splitlines() if "=" in line)


def test_only_hf_home_names_the_cache():
    """TRANSFORMERS_CACHE *is* the hub directory while HF_HOME is its parent,
    so setting both to one path gives two caches: snapshot_download writes to
    hub/ and from_pretrained to the root. That cost 46GB of duplicate
    checkpoints before it was noticed."""
    env = sourced(ENV_SH)
    assert env["HF_HOME"].endswith("/hf_cache")
    assert "TRANSFORMERS_CACHE" not in env
    assert "HF_DATASETS_CACHE" not in env


def test_the_bootstrap_agrees_with_the_session_env():
    """A bootstrap that cached elsewhere would leave a session re-downloading."""
    for script in (ENV_SH, BOOTSTRAP):
        text = script.read_text(encoding="utf-8")
        assert "unset TRANSFORMERS_CACHE" in text, script
        assert 'export TRANSFORMERS_CACHE' not in text, script


def test_the_config_cache_dir_is_the_hub_directory_not_its_parent():
    """`from_pretrained(cache_dir=X)` treats X as the cache, the way the
    deprecated TRANSFORMERS_CACHE did, while HF_HOME names X's parent. Pointing
    the config at $HF_HOME therefore made our loaders miss everything
    snapshot_download had already fetched, and download it again."""
    import glob

    from src.config import load_config

    for path in sorted(glob.glob("configs/*.yaml")):
        cache_dir = load_config(path)["paths"]["cache_dir"]
        assert cache_dir.endswith("/hf_cache/hub"), (path, cache_dir)


def test_the_config_cache_dir_agrees_with_the_session_hf_home():
    """Two locations means two copies, and the second is only noticed when the
    disk fills or a run stalls re-fetching what is already there."""
    from src.config import load_config

    env = sourced(ENV_SH, "MEDICAL_NLA_DATA_ROOT=/data1/heejae")
    import os

    os.environ["MEDICAL_NLA_DATA_ROOT"] = "/data1/heejae"
    cache_dir = load_config("configs/default.yaml")["paths"]["cache_dir"]
    assert cache_dir == f"{env['HF_HOME']}/hub"
