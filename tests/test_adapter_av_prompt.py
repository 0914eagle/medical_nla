import pytest

pytest.importorskip("torch")

from src.nla import AV_PROMPT_FILENAME, adapter_av_prompt
from src.run_nla import manifest_output_context


def test_an_adapter_directory_supplies_the_prompt_it_was_trained_under(tmp_path):
    """Training and generation drifting apart is the pilot defect this closes."""
    (tmp_path / AV_PROMPT_FILENAME).write_text("<concept>{injection_char}</concept>\n")
    assert adapter_av_prompt(str(tmp_path)).startswith("<concept>")


def test_a_directory_without_the_file_leaves_resolution_to_the_caller(tmp_path):
    assert adapter_av_prompt(str(tmp_path)) is None


def test_a_hub_id_or_no_adapter_is_not_treated_as_a_path():
    assert adapter_av_prompt("kitft/nla-gemma3-12b-L32-av") is None
    assert adapter_av_prompt(None) is None
    assert adapter_av_prompt("") is None


def test_activation_only_manifest_does_not_require_prompt_or_position():
    context = manifest_output_context(
        {
            "id": "case__direct_e3_p0",
            "base_id": "case",
            "activation_path": "/tmp/case.pt",
            "layer": 32,
            "position_family": "P0",
        }
    )
    assert context["prompt"] is None
    assert context["position"] is None
    assert context["layer"] == 32
