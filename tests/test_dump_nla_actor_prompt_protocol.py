import pytest

from scripts.dump_nla_actor_prompt_protocol import snapshot_revision


def test_snapshot_revision_from_huggingface_cache_path():
    assert (
        snapshot_revision(
            "/cache/models--kitft--nla-gemma3-12b-L32-av/snapshots/abc123/nla_meta.yaml"
        )
        == "abc123"
    )


def test_snapshot_revision_rejects_unversioned_path():
    with pytest.raises(ValueError, match="Cannot resolve"):
        snapshot_revision("/models/nla_meta.yaml")
