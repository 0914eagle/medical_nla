from collections import Counter

from scripts.train_medical_nla_lora import (
    EvalMetrics,
    aggregate_eval_metrics,
    source_temperature_epoch_rows,
)


def test_aggregate_eval_metrics_uses_token_denominators() -> None:
    combined = aggregate_eval_metrics(
        [
            EvalMetrics(1.0, 2.0, 0.5, 10, 20),
            EvalMetrics(3.0, 4.0, 1.5, 30, 10),
        ]
    )
    assert combined.loss == (1.0 * 30 + 3.0 * 40) / 70
    assert combined.content_loss == (2.0 * 10 + 4.0 * 30) / 40
    assert combined.scaffold_loss == (0.5 * 20 + 1.5 * 10) / 30
    assert combined.content_tokens == 40
    assert combined.scaffold_tokens == 30


def test_source_temperature_schedule_keeps_all_rows_and_sqrt_replays() -> None:
    rows = [
        {"id": f"large-{index}", "source_dataset": "large"} for index in range(9)
    ] + [{"id": "small-0", "source_dataset": "small"}]
    scheduled = source_temperature_epoch_rows(rows, alpha=0.5, seed=17, epoch=1)
    counts = Counter(row["source_dataset"] for row in scheduled)
    assert counts == {"large": 9, "small": 3}
    assert {row["id"] for row in rows} <= {row["id"] for row in scheduled}
    assert scheduled == source_temperature_epoch_rows(rows, alpha=0.5, seed=17, epoch=1)


def test_source_temperature_alpha_one_preserves_natural_counts() -> None:
    rows = [
        {"id": f"a-{index}", "source_dataset": "a"} for index in range(4)
    ] + [{"id": "b-0", "source_dataset": "b"}]
    scheduled = source_temperature_epoch_rows(rows, alpha=1.0, seed=17, epoch=1)
    assert Counter(row["source_dataset"] for row in scheduled) == {"a": 4, "b": 1}
