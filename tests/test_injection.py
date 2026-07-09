import pytest
import torch

from src.injection import find_token_positions, replace_placeholder_embeddings


def test_find_token_positions():
    input_ids = torch.tensor([1, 99, 2, 99, 3])
    assert find_token_positions(input_ids, 99) == [1, 3]


def test_replace_single_placeholder():
    input_ids = torch.tensor([[1, 99, 2]])
    embeds = torch.zeros((1, 3, 4), dtype=torch.float32)
    activation = torch.tensor([1.0, 2.0, 3.0, 4.0])

    result = replace_placeholder_embeddings(
        input_ids=input_ids,
        base_embeds=embeds,
        placeholder_token_id=99,
        activation=activation,
    )

    assert result.placeholder_positions == [1]
    assert result.inputs_embeds.shape == (1, 3, 4)
    assert torch.equal(result.inputs_embeds[0, 1], activation)
    assert torch.equal(result.inputs_embeds[0, 0], torch.zeros(4))


def test_replace_span_placeholders():
    input_ids = torch.tensor([99, 99, 2])
    embeds = torch.zeros((3, 2), dtype=torch.float32)
    activation = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    result = replace_placeholder_embeddings(
        input_ids=input_ids,
        base_embeds=embeds,
        placeholder_token_id=99,
        activation=activation,
    )

    assert result.placeholder_positions == [0, 1]
    assert torch.equal(result.inputs_embeds[:2], activation)


def test_placeholder_count_must_match_activation_span():
    input_ids = torch.tensor([99, 2])
    embeds = torch.zeros((2, 2))
    activation = torch.zeros((2, 2))

    with pytest.raises(ValueError, match="Placeholder count"):
        replace_placeholder_embeddings(
            input_ids=input_ids,
            base_embeds=embeds,
            placeholder_token_id=99,
            activation=activation,
        )


def test_embedding_dim_must_match_activation_dim():
    input_ids = torch.tensor([99])
    embeds = torch.zeros((1, 2))
    activation = torch.zeros(3)

    with pytest.raises(ValueError, match="Embedding dim"):
        replace_placeholder_embeddings(
            input_ids=input_ids,
            base_embeds=embeds,
            placeholder_token_id=99,
            activation=activation,
        )


def test_l2_normalization():
    input_ids = torch.tensor([99])
    embeds = torch.zeros((1, 2))
    activation = torch.tensor([3.0, 4.0])

    result = replace_placeholder_embeddings(
        input_ids=input_ids,
        base_embeds=embeds,
        placeholder_token_id=99,
        activation=activation,
        normalization="l2",
    )

    assert torch.allclose(result.inputs_embeds[0], torch.tensor([0.6, 0.8]))
