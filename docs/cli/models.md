# epalea models

Manage pretrained models.

## epalea models list

```bash
epalea models list
```

```
ID                     Domain         Classes    Aggregator   Index    Released
compliance-v1          compliance     3          ✓            ✓        2025-01-01
```

## epalea models download

```bash
epalea models download compliance-v1
# ✓ Downloaded to ./pretrained/compliance-v1/
```

Downloads all assets: `best_model.pt`, `aggregator.pt`, `evidence_index/`, `schema.json`, `sample_data/`.
