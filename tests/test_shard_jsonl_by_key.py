from scripts.shard_jsonl_by_key import shard_for


def test_same_case_variants_always_share_a_shard() -> None:
    assert shard_for("case-a", 2) == shard_for("case-a", 2)
    assert shard_for("case-a", 7) == shard_for("case-a", 7)


def test_shard_index_is_in_range() -> None:
    for count in (1, 2, 7):
        assert 0 <= shard_for("case-a", count) < count
