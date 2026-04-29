# epalea train-aggregator

Train the `EvidenceAggregator` network. Enables `--mode aggregator` and `--mode both`.

## Usage

```bash
# Config file (recommended)
epalea train-aggregator --config ./user_workspace/configs/train_aggregator.yaml

# Inline flags
epalea train-aggregator \
  --checkpoint [./user_workspace/checkpoints/domain/best_model.pt] \
  --schema [./user_workspace/checkpoints/domain/schema.json] \
  --index-dir [./user_workspace/data/domain] \
  --data-dir [./user_workspace/data/domain] \
  --aggregator-checkpoint-dir [./user_workspace/checkpoints/domain]
```

## Config file — `train_aggregator.yaml`

```yaml
checkpoint: ./user_workspace/checkpoints/compliance/best_model.pt
schema: ./user_workspace/checkpoints/compliance/schema.json

# index_dir and data_dir are the same: epalea train writes the index into data_dir.
index_dir: ./user_workspace/data/compliance
data_dir: ./user_workspace/data/compliance

aggregator_checkpoint_dir: ./user_workspace/checkpoints/compliance
latent_dim: 64
hidden_dim: 128
dropout: 0.1
epochs: 30
learning_rate: 0.001
```

## Prerequisites

- `best_model.pt` from `epalea train`
- Evidence index built automatically by `epalea train` (train + val splits)

## Options

| Flag | Type | Default | Description |
|---|---|---|---|
| `--config` | path | — | YAML config file |
| `--checkpoint` | path | *required* | `best_model.pt` |
| `--schema` | path | — | `schema.json` — auto-detected from checkpoint dir |
| `--index-dir` | path | *required* | Evidence index directory (same as `--data-dir`) |
| `--data-dir` | path | *required* | Training data directory |
| `--aggregator-checkpoint-dir` | path | checkpoint parent dir | Output directory |
| `--latent-dim` | int | `64` | Must match LPF training |
| `--hidden-dim` | int | `128` | Aggregator hidden dimension |
| `--dropout` | float | `0.1` | Dropout rate |
| `--epochs` | int | `30` | Training epochs |
| `--learning-rate` | float | `0.001` | Adam learning rate |

> `--index-dir` and `--data-dir` point to the same directory because `epalea train` writes the evidence index into `data-dir`. This will be simplified in a future release.

## Output

```
Training aggregator...
  Preparing training data: 240 entities
  Epoch 10/30, Loss: 0.412
  Epoch 20/30, Loss: 0.287
  Epoch 30/30, Loss: 0.241

✓ Saved: ./user_workspace/checkpoints/compliance/aggregator.pt
--mode aggregator and --mode both are now available.
```

After training, `schema.json` is updated with `has_aggregator: true`.
