# epalea infer

Run inference for a single entity.

## Usage

```bash
# Config file + entity flag
epalea infer --config ./user_workspace/configs/infer.yaml --entity-id [C0001]

# Inline flags — pretrained model
epalea infer \
  --checkpoint ./pretrained/compliance-v1/best_model.pt \
  --aggregator-checkpoint ./pretrained/compliance-v1/aggregator.pt \
  --schema ./pretrained/compliance-v1/schema.json \
  --index-dir ./pretrained/compliance-v1/evidence_index \
  --entity-id [C0001] \
  --mode both

# Inline flags — custom-trained model
epalea infer \
  --checkpoint [./user_workspace/checkpoints/domain/best_model.pt] \
  --aggregator-checkpoint [./user_workspace/checkpoints/domain/aggregator.pt] \
  --schema [./user_workspace/checkpoints/domain/schema.json] \
  --index-dir [./user_workspace/data/domain] \
  --entity-id [C0001] \
  --mode both
```

`--entity-id` cannot be set in a config file and must always be passed inline.

## Config file — `infer.yaml`

```yaml
# Pretrained model
checkpoint: ./pretrained/compliance-v1/best_model.pt
aggregator_checkpoint: ./pretrained/compliance-v1/aggregator.pt
schema: ./pretrained/compliance-v1/schema.json
index_dir: ./pretrained/compliance-v1/evidence_index

# Custom-trained model — replace the four lines above with:
# checkpoint: ./user_workspace/checkpoints/[domain]/best_model.pt
# aggregator_checkpoint: ./user_workspace/checkpoints/[domain]/aggregator.pt
# schema: ./user_workspace/checkpoints/[domain]/schema.json
# index_dir: ./user_workspace/data/[domain]

mode: both
output_format: nested
top_k: 5
n_samples: 4
temperature: 0.8
alpha: 0.1
```

## Prerequisites

- `best_model.pt` + evidence index (test evidence must be indexed via `epalea index`)
- For `--mode aggregator` or `--mode both`: also `aggregator.pt`

## Options

| Flag | Type | Default | Description |
|---|---|---|---|
| `--config` | path | — | YAML config file |
| `--checkpoint` | path | *required* | `best_model.pt` |
| `--aggregator-checkpoint` | path | — | `aggregator.pt` — required for `aggregator` / `both` |
| `--schema` | path | — | `schema.json` — auto-detected from checkpoint dir |
| `--index-dir` | path | *required* | Evidence index directory |
| `--entity-id` | str | *required* | Entity to query — always passed inline |
| `--mode` | str | `spn` | `spn` \| `aggregator` \| `both` |
| `--output-format` | str | `nested` | `nested` \| `flat` |
| `--top-k` | int | `5` | Evidence items to retrieve |
| `--n-samples` | int | `4` | Monte Carlo samples |
| `--temperature` | float | `0.8` | Temperature scaling |
| `--alpha` | float | `0.1` | Weight penalty α |

## Output — `--mode both --output-format nested`

```json
{
  "entity_id": "C0001",
  "mode": "both",
  "results": {
    "spn": {
      "prediction": "high",
      "confidence": 0.84,
      "distribution": { "low": 0.06, "medium": 0.10, "high": 0.84 }
    },
    "aggregator": {
      "prediction": "high",
      "confidence": 0.81,
      "distribution": { "low": 0.08, "medium": 0.11, "high": 0.81 }
    }
  },
  "uncertainty": {
    "spn":        { "epistemic": 0.12, "aleatoric": 0.08, "total": 0.20, "weights_source": "uniform" },
    "aggregator": { "epistemic": 0.09, "aleatoric": 0.11, "total": 0.20, "weights_source": "aggregator" }
  },
  "n_evidence_used": 5,
  "execution_time_ms": 31.2
}
```

Both `uncertainty.spn` and `uncertainty.aggregator` are returned independently — neither is suppressed.

## Output — `--mode both --output-format flat`

```json
{
  "entity_id": "C0001",
  "mode": "both",
  "spn_prediction": "high",
  "spn_confidence": 0.84,
  "aggregator_prediction": "high",
  "aggregator_confidence": 0.81,
  "spn_epistemic": 0.12,
  "spn_aleatoric": 0.08,
  "spn_total_uncertainty": 0.20,
  "aggregator_epistemic": 0.09,
  "aggregator_aleatoric": 0.11,
  "aggregator_total_uncertainty": 0.20,
  "n_evidence_used": 5,
  "execution_time_ms": 31.2
}
```

## Output — `--mode spn`

```json
{
  "entity_id": "C0001",
  "mode": "spn",
  "results": {
    "spn": { "prediction": "high", "confidence": 0.84, "distribution": { ... } }
  },
  "uncertainty": {
    "spn": { "epistemic": 0.12, "aleatoric": 0.08, "total": 0.20, "weights_source": "uniform" }
  },
  "n_evidence_used": 5,
  "execution_time_ms": 18.1
}
```
