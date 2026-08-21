import glob

from src.config import load_config

MODELS = ("source_model", "nla_model")


def test_every_12b_model_is_allowed_to_span_two_cards():
    """A 12B checkpoint in bfloat16 is 23.6GB and a card here holds 24GB. On one
    card the AV model loaded at 100% of cuda:0 and the first training forward
    died asking for 84MB; the backbone had the same failure earlier."""
    for path in sorted(glob.glob("configs/*.yaml")):
        cfg = load_config(path)
        for name in MODELS:
            assert cfg[name]["device_map"] == "auto", (path, name)


def test_the_per_card_budget_is_stated_rather_than_inferred():
    """Left to its own heuristic accelerate put a fifth of the backbone on meta
    with every card empty, which fails later inside cuBLAS naming neither."""
    for path in sorted(glob.glob("configs/*.yaml")):
        cfg = load_config(path)
        for name in MODELS:
            budget = cfg[name].get("max_memory")
            assert budget, (path, name)
            assert set(budget) == {0, 1}, (path, name, budget)
            assert all(str(v).endswith("GiB") for v in budget.values()), (path, name)
