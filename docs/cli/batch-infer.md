# epalea batch-infer

Run inference for every entity in a companies JSON file.

## Usage

```bash
# Config file + companies flag
epalea batch-infer \
  --config ./user_workspace/configs/infer.yaml \
  --companies [./user_workspace/data/domain/test_companies.json] \
  --output-dir [./user_workspace/results/domain]

# Inline flags — pretrained model
epalea batch-infer \
  --checkpoint ./pretrained/compliance-v1/best_model.pt \
  --aggregator-checkpoint ./pretrained/compliance-v1/aggregator.pt \
  --schema ./pretrained/compliance-v1/schema.json \
  --index-dir ./pretrained/compliance-v1/evidence_index \
  --companies ./pretrained/compliance-v1/sample_data/test_companies.json \
  --mode both \
  --output-dir ./user_workspace/results/compliance

# Inline flags — custom-trained model
epalea batch-infer \
  --checkpoint [./user_workspace/checkpoints/domain/best_model.pt] \
  --aggregator-checkpoint [./user_workspace/checkpoints/domain/aggregator.pt] \
  --schema [./user_workspace/checkpoints/domain/schema.json] \
  --index-dir [./user_workspace/data/domain] \
  --companies [./user_workspace/data/domain/test_companies.json] \
  --mode both \
  --output-dir [./user_workspace/results/domain]
```

`batch-infer` inherits all inference flags from `epalea infer`. The config file is the same `infer.yaml` — pass `--companies` and `--output-dir` inline on top.

## Options

All options from [`epalea infer`](./infer.md) apply, plus:

| Flag | Type | Default | Description |
|---|---|---|---|
| `--companies` | path | *required* | Companies JSON file to run batch inference over |
| `--output-dir` | path | `./user_workspace/results` | Output directory |

## Output

Writes `{output-dir}/batch_predictions.json` — an array of per-entity results in the same format as `epalea infer`.

```json
[
  {
    "entity_id": "C0001",
    "mode": "both",
    "results": {
      "spn":        { "prediction": "high", "confidence": 0.84, "distribution": { ... } },
      "aggregator": { "prediction": "high", "confidence": 0.81, "distribution": { ... } }
    },
    "uncertainty": {
      "spn":        { "epistemic": 0.12, "aleatoric": 0.08, "total": 0.20 },
      "aggregator": { "epistemic": 0.09, "aleatoric": 0.11, "total": 0.20 }
    },
    "n_evidence_used": 5,
    "execution_time_ms": 31.2
  },
  ...
]
```

Use `--output-format flat` for a flat array suitable for loading into pandas or a spreadsheet.
