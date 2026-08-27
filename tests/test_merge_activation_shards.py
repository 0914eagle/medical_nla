from scripts.merge_activation_shards import parse_path_maps, remap_path


def test_activation_path_is_remapped_by_prefix() -> None:
    mappings = parse_path_maps(["/data1/heejae=/data/heejae"])
    assert remap_path("/data1/heejae/a.pt", mappings) == "/data/heejae/a.pt"
    assert remap_path("/other/a.pt", mappings) == "/other/a.pt"
