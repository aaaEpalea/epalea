# epalea evaluate

Run batch inference and compute full evaluation metrics over labeled test data.

## Usage

```bash
# Config file (recommended)
epalea evaluate --config ./user_workspace/configs/evaluate.yaml

# Inline flags — pretrained model
epalea evaluate \
  --checkpoint ./pretrained/compliance-v1/best_model.pt \
  --aggregator-checkpoint ./pretrained/compliance-v1/aggregator.pt \
  --schema ./pretrained/compliance-v1/schema.json \
  --index-dir ./pretrained/compliance-v1/evidence_index \
  --test-companies ./pretrained/compliance-v1/sample_data/test_companies.json \
  --mode both \
  --output-dir ./user_workspace/results/compliance

# Inline flags — custom-trained model
epalea evaluate \
  --checkpoint [./user_workspace/checkpoints/domain/best_model.pt] \
  --aggregator-checkpoint [./user_workspace/checkpoints/domain/aggregator.pt] \
  --schema [./user_workspace/checkpoints/domain/schema.json] \
  --index-dir [./user_workspace/data/domain] \
  --test-companies [./user_workspace/data/domain/test_companies.json] \
  --mode both \
  --output-dir [./user_workspace/results/domain]
```

## Config file — `evaluate.yaml`

```yaml
# Pretrained model
checkpoint: ./pretrained/compliance-v1/best_model.pt
aggregator_checkpoint: ./pretrained/compliance-v1/aggregator.pt
schema: ./pretrained/compliance-v1/schema.json
index_dir: ./pretrained/compliance-v1/evidence_index
test_companies: ./pretrained/compliance-v1/sample_data/test_companies.json

# Custom-trained model — replace the five lines above with:
# checkpoint: ./user_workspace/checkpoints/[domain]/best_model.pt
# aggregator_checkpoint: ./user_workspace/checkpoints/[domain]/aggregator.pt
# schema: ./user_workspace/checkpoints/[domain]/schema.json
# index_dir: ./user_workspace/data/[domain]
# test_companies: ./user_workspace/data/[domain]/test_companies.json

output_dir: ./user_workspace/results/compliance
mode: both
output_format: nested
top_k: 5
n_samples: 4
```

## Prerequisites

Same as `epalea infer` — test evidence must be indexed via `epalea index` before evaluating.

## Options

All options from [`epalea infer`](./infer.md) apply, plus:

| Flag | Type | Default | Description |
|---|---|---|---|
| `--test-companies` | path | *required* | Test companies JSON with ground-truth labels |
| `--output-dir` | path | `./user_workspace/results` | Output directory |

## Terminal output

```
── mode: both ────────────────────────────────────────────────────────────────────
System             Acc   Macro F1      NLL    Brier      ECE   Ep.Unc   Al.Unc
─────────────────────────────────────────────────────────────────────────────────
LPF-SPN          0.956      0.946    0.230    0.024    0.032    0.106    0.053
LPF-Agg          0.896      0.881    0.271    0.040    0.066    0.097    0.053
Saved: [output-dir]/evaluation.json
```

`Ep.Unc` and `Al.Unc` show mean epistemic and aleatoric uncertainty across the test set per mode.

## Output file — `evaluation.json`

```json
{
  "mode": "both",
  "metrics": {
    "spn":        { "accuracy": 0.956, "macro_f1": 0.946, "nll": 0.230, "brier": 0.024, "ece": 0.032 },
    "aggregator": { "accuracy": 0.896, "macro_f1": 0.881, "nll": 0.271, "brier": 0.040, "ece": 0.066 }
  },
  "n_evaluated": 135,
  "per_entity": [
    {
      "entity_id": "C0001",
      "true_label": "high",
      "spn_prediction": "high",
      "spn_confidence": 0.84,
      "aggregator_prediction": "high",
      "aggregator_confidence": 0.87,
      "uncertainty": {
        "spn":        { "epistemic": 0.12, "aleatoric": 0.08, "total": 0.20 },
        "aggregator": { "epistemic": 0.09, "aleatoric": 0.11, "total": 0.20 }
      }
    }
  ]
}
```

`per_entity` includes per-mode uncertainty for every entity — use it to find entities with high epistemic uncertainty (needs more evidence) or high aleatoric uncertainty (genuinely ambiguous, escalate for human review).

## Metrics computed

Accuracy, macro F1, weighted F1, NLL, Brier score, ECE.
