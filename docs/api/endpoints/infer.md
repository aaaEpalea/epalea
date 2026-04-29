# POST /api/v1/infer

Run inference for a single entity.

## Request

```json
{
  "model_id": "compliance-v1",
  "entity_id": "C0042",
  "evidence": [
    {
      "evidence_id": "C0042_E001",
      "text_content": "Company demonstrates strong compliance with timely filings.",
      "credibility": 0.9,
      "evidence_type": "audit_report"
    }
  ],
  "options": {
    "mode": "spn",
    "output_format": "nested",
    "top_k": 5,
    "n_samples": 4,
    "temperature": 0.8,
    "alpha": 0.1
  }
}
```

## Response — SPN mode (live)

```json
{
  "entity_id": "C0042",
  "model_id": "compliance-v1",
  "mode": "spn",
  "results": {
    "spn": {
      "prediction": "high",
      "confidence": 0.84,
      "distribution": {"low": 0.06, "medium": 0.10, "high": 0.84}
    }
  },
  "uncertainty": {
    "epistemic": 0.12,
    "aleatoric": 0.08,
    "total": 0.20,
    "decomposition_error": 0.003,
    "weights_source": "uniform"
  },
  "n_evidence_used": 5,
  "execution_time_ms": 23.4
}
```

## Response — `mode: both` (coming soon)

When `mode: both` is requested but the aggregator is not yet live, the SPN result
is returned alongside a coming-soon notice. The call is not wasted.

```json
{
  "entity_id": "C0042",
  "mode": "both",
  "status": "coming_soon",
  "message": "Aggregator mode is not yet available via API. Use --mode spn, or run locally: pip install epalea && epalea infer --mode both",
  "docs": "https://epalea.ai/docs/api/endpoints/infer",
  "spn_result": { "results": {...}, "uncertainty": {...} }
}
```
