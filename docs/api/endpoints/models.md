# GET /api/v1/models

List all available pretrained models.

## Response

```json
{
  "models": [
    {
      "model_id": "compliance-v1",
      "domain": "compliance",
      "predicate": "compliance_level",
      "domain_values": ["low", "medium", "high"],
      "has_aggregator": true,
      "has_evidence_index": true,
      "description": "Tax compliance risk classification",
      "released": "2025-01-01"
    }
  ]
}
```

# GET /api/v1/models/{model_id}

Get metadata for a specific model.

```bash
curl https://epalea.ai/api/v1/models/compliance-v1 \
  -H "X-API-Key: ek_live_..."
```
