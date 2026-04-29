# Key Concepts

## LPF — Latent Factor Posteriors

LPF encodes evidence items into Gaussian latent posteriors via a VAE encoder, decodes each posterior into a soft likelihood factor, and performs exact marginal inference over a Sum-Product Network.

## The Pipeline

```
Step 1: epalea generate-data     → train/val/test JSON files
Step 2: epalea train             → best_model.pt  +  train/val evidence indexed automatically
Step 3: epalea train-aggregator  → aggregator.pt  (uses index from Step 2)
Step 4: epalea index             → adds test evidence to existing index
Step 5: epalea infer / batch-infer / evaluate
```

Steps must run in order. `epalea train` builds the evidence index for train and val splits automatically — you do **not** run a separate index step for those. Step 4 is needed only to add test evidence before running inference or evaluate.

> **Shortcut:** `epalea train-full` runs Steps 2 and 3 together. You still run Step 4 separately before inference.

## SPN vs Aggregator

| Mode | Description | Requires |
|---|---|---|
| `spn` | Exact SPN inference with uniform evidence weights | `best_model.pt` + index |
| `aggregator` | Learned per-evidence quality weights | + `aggregator.pt` |
| `both` | Both paths, side-by-side | + `aggregator.pt` |

## Uncertainty Decomposition

Returned per-mode in `mode="both"` — neither signal is suppressed:

```json
{
  "uncertainty": {
    "spn":        { "epistemic": 0.12, "aleatoric": 0.08, "total": 0.20, "weights_source": "uniform" },
    "aggregator": { "epistemic": 0.09, "aleatoric": 0.11, "total": 0.20, "weights_source": "aggregator" }
  }
}
```

- `epistemic` — model uncertainty, reducible with more data
- `aleatoric` — data uncertainty, irreducible

## Schema

```json
{
  "model_id": "compliance-v1",
  "predicate": "compliance_level",
  "domain_values": ["low", "medium", "high"],
  "has_aggregator": true,
  "has_evidence_index": true
}
```

Written to `checkpoints/{domain}/schema.json` after training and updated automatically after each step.

## Directory Conventions

| Path | Contents |
|---|---|
| `pretrained/{model-id}/` | Shipped pretrained assets (read-only) |
| `user_workspace/data/{domain}/` | Generated data + evidence index |
| `user_workspace/checkpoints/{domain}/` | Trained model weights |
| `user_workspace/results/{domain}/` | Evaluation outputs |
