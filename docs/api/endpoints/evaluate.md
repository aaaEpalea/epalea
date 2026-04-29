# POST /api/v1/evaluate

Run batch inference + compute evaluation metrics over labeled test data.

Maximum evaluation set: 500 entities.

## Request

```json
{
  "model_id": "compliance-v1",
  "test_data": [
    {
      "entity_id": "C0001",
      "true_label": "high",
      "evidence": [
        {
          "evidence_id": "E001",
          "text_content": "Company passed all audits with zero findings.",
          "credibility": 0.95,
          "evidence_type": "audit_report"
        }
      ]
    }
  ],
  "options": {
    "mode": "both",
    "top_k": 5
  }
}
```

## Response

```json
{
  "model_id": "compliance-v1",
  "mode": "both",
  "metrics": {
    "accuracy": 0.947,
    "macro_f1": 0.941,
    "weighted_f1": 0.943,
    "nll": 0.182,
    "brier": 0.073,
    "ece": 0.004
  },
  "per_entity": [
    {
      "entity_id": "C0001",
      "true_label": "high",
      "prediction": "high",
      "correct": true,
      "confidence": 0.84,
      "distribution": { "low": 0.05, "medium": 0.11, "high": 0.84 },
      "uncertainty": {
        "spn": {
          "epistemic": 0.12,
          "aleatoric": 0.08,
          "total": 0.20,
          "weights_source": "uniform"
        },
        "aggregator": {
          "epistemic": 0.09,
          "aleatoric": 0.11,
          "total": 0.20,
          "weights_source": "aggregator"
        }
      }
    }
  ],
  "n_evaluated": 135,
  "n_failed": 0,
  "execution_time_ms": 412.3
}
```

The `per_entity` array always includes `uncertainty.spn` and `uncertainty.aggregator` when `mode="both"`, so you can inspect per-entity uncertainty alongside prediction correctness. Use this to identify entities where:
- High epistemic uncertainty → model was unsure (more evidence would help)
- High aleatoric uncertainty → evidence itself was ambiguous (human review warranted)
- SPN and aggregator disagree → worth investigating that entity further
