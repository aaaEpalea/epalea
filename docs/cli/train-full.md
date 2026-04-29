# epalea train-full

Run Steps 2 and 3 of the pipeline together: LPF base model + aggregator.

## Usage

```bash
epalea train-full --config ./user_workspace/configs/train_full.yaml
```

`--config` is required for `train-full`. All stage parameters are in the config.

## Config file — `train_full.yaml`

```yaml
# Stage 1: LPF base model training
# Also indexes train + val evidence automatically into data_dir.
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

# Stage 2: Aggregator training
# Uses the index written to data_dir during Stage 1.
aggregator_hidden_dim: 128
aggregator_dropout: 0.1
aggregator_epochs: 30
aggregator_learning_rate: 0.001
```

## What it does

```
Stage 1/2  epalea train            → best_model.pt  +  train/val evidence indexed
Stage 2/2  epalea train-aggregator → aggregator.pt
```

**It does not index test evidence.** After `train-full` completes, run `epalea index` to add test evidence before inference or evaluate.

## Output

```
══ Stage 1/2: LPF Base Model Training ════════════════════
  Seed search: 7 seeds
  ✓ Best seed: 2 (val_acc=0.923)
  ✓ Saved: [checkpoint-dir]/best_model.pt
  ✓ Train/val evidence indexed at [data-dir]/

══ Stage 2/2: Aggregator Training ════════════════════════
  Epoch 10/30, Loss: 0.412
  Epoch 30/30, Loss: 0.241
  ✓ Saved: [checkpoint-dir]/aggregator.pt

══ train-full complete ════════════════════════════════════
✓ Base model and aggregator trained.

Next step: index test evidence before inference:
  epalea index --config ./user_workspace/configs/index.yaml
```

If any stage fails, the pipeline stops immediately with exit code 1.

## Full workflow using train-full

```bash
# Step 1
epalea generate-data \
  --domain [domain] \
  --n-entities [300] \
  --output-dir [./user_workspace/data]

# Steps 2 + 3
epalea train-full --config ./user_workspace/configs/train_full.yaml

# Step 4
epalea index --config ./user_workspace/configs/index.yaml

# Step 5
epalea evaluate --config ./user_workspace/configs/evaluate.yaml
```
