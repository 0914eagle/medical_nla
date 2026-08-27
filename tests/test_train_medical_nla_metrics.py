from scripts.train_medical_nla_lora import EvalMetrics, aggregate_eval_metrics


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
