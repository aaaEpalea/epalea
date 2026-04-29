# Tax Compliance Domain

**Predicate:** `compliance_level`  
**Classes:** `low`, `medium`, `high`  
**Model ID:** `compliance-v1`

## Description

Predicts the tax compliance risk level of a company given textual and structured evidence —
audit reports, filing records, financial data, and news.

## Pretrained model

```bash
epalea models download compliance-v1
epalea infer --entity-id C0001 --mode both
```

Or via API:

```bash
curl https://epalea.ai/api/v1/infer \
  -H "X-API-Key: ek_live_..." \
  -d '{"model_id": "compliance-v1", "entity_id": "C0001", "evidence": [...]}'
```

## Benchmark

Trained on 900 company-year records (300 companies × 3 years).

| System | Accuracy | NLL | Brier | ECE |
|---|---|---|---|---|
| LPF-SPN | 0.947 | 0.182 | 0.073 | 0.004 |
| LPF-Agg | 0.951 | 0.174 | 0.069 | 0.003 |
