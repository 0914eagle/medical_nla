from src.extract_activations import PASSTHROUGH_FIELDS


def test_ddxplus_e5_value_and_counterfactual_metadata_are_preserved() -> None:
    required = {
        "cue_value_ids",
        "cue_value_labels",
        "cue_polarities",
        "official_split",
        "cf_target_index",
        "cf_original_evidence_id",
        "cf_original_value_id",
        "cf_replacement_evidence_id",
        "cf_replacement_value_id",
    }
    assert required <= set(PASSTHROUGH_FIELDS)
