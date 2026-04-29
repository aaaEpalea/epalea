# Pipeline Architecture

The LPF pipeline has five steps in strict order.

## Step 1 — Generate Data

```bash
epalea generate-data \
  --domain compliance \
  --n-entities 300 \
  --years 2020 --years 2021 --years 2022 \
  --evidence-per-entity 5 \
  --noise-level 0.1 \
  --contradictory-rate 0.05 \
  --output-dir ./user_workspace/data
```

**Produces** in `./user_workspace/data/compliance/`:
```
train_companies.json    val_companies.json    test_companies.json
train_evidence.json     val_evidence.json     test_evidence.json
```

## Step 2 — Train LPF Base Model

```bash
epalea train \
  --domain compliance \
  --predicate compliance_level \
  --domain-values low --domain-values medium --domain-values high \
  --data-dir ./user_workspace/data/compliance \
  --checkpoint-dir ./user_workspace/checkpoints/compliance
```

**Requires:** data files from Step 1

**Produces:**
- `./user_workspace/checkpoints/compliance/best_model.pt`
- `./user_workspace/checkpoints/compliance/schema.json`
- `./user_workspace/data/compliance/vector_store.faiss` ← train + val evidence indexed automatically
- `./user_workspace/data/compliance/metadata.jsonl`

> The evidence index for train and val splits is built automatically during training. There is no separate index step for these splits.

## Step 3 — Train Aggregator

```bash
epalea train-aggregator \
  --checkpoint ./user_workspace/checkpoints/compliance/best_model.pt \
  --schema ./user_workspace/checkpoints/compliance/schema.json \
  --index-dir ./user_workspace/data/compliance \
  --data-dir ./user_workspace/data/compliance \
  --aggregator-checkpoint-dir ./user_workspace/checkpoints/compliance
```

**Requires:** `best_model.pt` + index from Step 2

**Produces:** `./user_workspace/checkpoints/compliance/aggregator.pt`

> `--index-dir` and `--data-dir` point to the same directory because that is where `epalea train` wrote the index. This will be simplified in a future release.

## Step 4 — Index Test Evidence

```bash
epalea index \
  --checkpoint ./user_workspace/checkpoints/compliance/best_model.pt \
  --schema ./user_workspace/checkpoints/compliance/schema.json \
  --evidence ./user_workspace/data/compliance/test_evidence.json \
  --predicate compliance_level \
  --index-dir ./user_workspace/data/compliance
```

**Requires:** `best_model.pt` + existing index from Step 2

**Produces:** appends test entity vectors into the existing index at `./user_workspace/data/compliance/`

> This step adds test evidence into the existing index. It must run before `epalea infer` or `epalea evaluate` on test entities.

## Step 5 — Infer / Evaluate

```bash
# Single inference
epalea infer \
  --checkpoint ./user_workspace/checkpoints/compliance/best_model.pt \
  --aggregator-checkpoint ./user_workspace/checkpoints/compliance/aggregator.pt \
  --schema ./user_workspace/checkpoints/compliance/schema.json \
  --index-dir ./user_workspace/data/compliance \
  --entity-id C0001 \
  --mode both

# Full evaluation
epalea evaluate \
  --checkpoint ./user_workspace/checkpoints/compliance/best_model.pt \
  --aggregator-checkpoint ./user_workspace/checkpoints/compliance/aggregator.pt \
  --schema ./user_workspace/checkpoints/compliance/schema.json \
  --index-dir ./user_workspace/data/compliance \
  --test-companies ./user_workspace/data/compliance/test_companies.json \
  --mode both \
  --output-dir ./user_workspace/results/compliance
```

## Shortcut: train-full

`epalea train-full` runs Steps 2 and 3 together from a single config:

```bash
epalea train-full --config ./user_workspace/configs/train_full.yaml
```

**You still need to run Step 4** (`epalea index`) separately before inference.

## Pipeline Summary

```
generate-data
    ↓
train  ←─────────────── also builds train/val index
    ↓
train-aggregator  ←───── uses existing index
    ↓
index  ←─────────────── adds test evidence
    ↓
infer / batch-infer / evaluate
```
