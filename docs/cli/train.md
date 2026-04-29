# epalea train

Train the LPF encoder + decoder. Produces `best_model.pt` and automatically indexes train and val evidence.

## Usage

```bash
# Config file (recommended)
epalea train --config ./user_workspace/configs/train.yaml

# Inline flags
epalea train \
  --domain [domain] \
  --predicate [predicate_name] \
  --domain-values [value1] --domain-values [value2] --domain-values [value3] \
  --data-dir [./user_workspace/data/domain] \
  --checkpoint-dir [./user_workspace/checkpoints/domain]
```

## Config file — `train.yaml`

```yaml
domain: compliance
predicate: compliance_level
domain_values: [low, medium, high]
data_dir: ./user_workspace/data/compliance
checkpoint_dir: ./user_workspace/checkpoints/compliance
n_seeds: 7
embedding_dim: 384
latent_dim: 64
epochs: 50
early_stopping_patience: 15
learning_rate: 0.001
kl_weight: 0.01
```

## Options

| Flag | Type | Default | Description |
|---|---|---|---|
| `--config` | path | — | YAML config file |
| `--domain` | str | *required* | Domain name |
| `--predicate` | str | *required* | Predicate name |
| `--domain-values` | list | *required* | Class values — pass once per value |
| `--data-dir` | path | *required* | Directory containing train/val/test JSON files |
| `--checkpoint-dir` | path | `./user_workspace/checkpoints/` | Output directory |
| `--n-seeds` | int | `7` | Seeds for seed search |
| `--single-seed` | int | — | Fixed seed — skips seed search |
| `--label-key` | str | predicate name | Label field name in entity JSON |
| `--embedding-dim` | int | `384` | Embedding dimension |
| `--latent-dim` | int | `64` | Latent dimension |
| `--epochs` | int | `50` | Max training epochs |
| `--early-stopping-patience` | int | `15` | Early stopping patience |
| `--learning-rate` | float | `0.001` | Adam learning rate |
| `--kl-weight` | float | `0.01` | KL regularisation weight β |

## `--label-key`

If your entity JSON uses a different field name for the label than the predicate name:

```bash
epalea train --predicate risk_tier --label-key risk_category ...
```

## What it produces

```
[checkpoint-dir]/
├── best_model.pt       ← best checkpoint across all seeds
├── schema.json         ← domain/predicate/model metadata
└── seed_*/
    └── best_model.pt   ← per-seed checkpoints

[data-dir]/
├── vector_store.faiss  ← train + val evidence indexed automatically
└── metadata.jsonl
```

> The evidence index is written into `data-dir` during training. There is no separate index step for train and val splits.

## Output

```
Seed search: 7 seeds
  Seed 1/7 — val_acc=0.871
  Seed 2/7 — val_acc=0.923  ← best
  ...
✓ Best seed: 2 (val_acc=0.923)
✓ Saved: ./user_workspace/checkpoints/compliance/best_model.pt
✓ Indexed 1200 train + 300 val evidence items → ./user_workspace/data/compliance/
```
