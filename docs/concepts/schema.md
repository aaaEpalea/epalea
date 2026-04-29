# Schema

The schema defines the structure of predicates and variables for SPN reasoning.

## schema.json

Every pretrained model includes a `schema.json`:

```json
{
  "model_id": "compliance-v1",
  "version": "1.0.0",
  "domain": "compliance",
  "predicate": "compliance_level",
  "domain_values": ["low", "medium", "high"],
  "variable_name": "compliance",
  "embedding_dim": 384,
  "latent_dim": 64,
  "has_aggregator": true,
  "has_evidence_index": true,
  "description": "Tax compliance risk classification",
  "released": "2025-01-01"
}
```

## Key fields

- `has_aggregator` — whether `aggregator.pt` exists; determines if `aggregator`/`both` modes are available
- `has_evidence_index` — whether `evidence_index/` exists; required for all inference
- `variable_name` — name of the latent variable in the SPN (distinct from `predicate`)

## Python API

```python
from models.schema import Schema

schema = Schema()
schema.add_variable("compliance", ["low", "medium", "high"])
schema.add_predicate("compliance_level", ["compliance"], ["low", "medium", "high"])

domain = schema.get_predicate_domain("compliance_level")  # ["low", "medium", "high"]
```
