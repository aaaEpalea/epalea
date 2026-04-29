# Try It Now — Full Pipeline Walkthrough

This guide walks through all five steps of the LPF pipeline from scratch using the compliance domain.
Every command is exact and runnable. Paths are shown in the form `[./path/to/thing]` where you should
substitute your own locations — the compliance example uses the exact paths shown.

**Time to complete:** ~15 minutes (depending on hardware)

**What you'll build:** A trained LPF-SPN + Aggregator model that classifies company compliance
level as `low`, `medium`, or `high` with full epistemic/aleatoric uncertainty decomposition.

---

## Before you start

```bash
pip install epalea
epalea --version   # should print 1.0.0
epalea info        # confirms PyTorch, device, and workspace status
```

---

## Step 1 — Generate data

Generate 300 synthetic companies with 5 evidence items each across 3 fiscal years.

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

**Produces** `./user_workspace/data/compliance/`:

```
train_companies.json   (210 companies)
train_evidence.json    (1050 evidence items)
val_companies.json     (45 companies)
val_evidence.json      (225 evidence items)
test_companies.json    (45 companies)
test_evidence.json     (225 evidence items)
```

**Verify:**
```bash
ls ./user_workspace/data/compliance/
```

---

## Step 2 — Train the LPF base model

Train the VAE encoder + decoder using seed search across 7 seeds.
This step also **automatically indexes train and val evidence** — no separate index step needed for those.

```bash
epalea train \
  --domain compliance \
  --predicate compliance_level \
  --domain-values low --domain-values medium --domain-values high \
  --data-dir ./user_workspace/data/compliance \
  --checkpoint-dir ./user_workspace/checkpoints/compliance
```

Or using a config file:

```bash
epalea train --config ./user_workspace/configs/train.yaml
```

**Produces:**

```
./user_workspace/checkpoints/compliance/
├── best_model.pt       ← best checkpoint across all seeds
└── schema.json

./user_workspace/data/compliance/
├── vector_store.faiss  ← train + val evidence indexed here
└── metadata.jsonl
```

**Verify:**
```bash
ls ./user_workspace/checkpoints/compliance/
ls ./user_workspace/data/compliance/*.faiss ./user_workspace/data/compliance/*.jsonl
```

Expected output in terminal:
```
Seed search: 7 seeds
  Seed 2/7 — val_acc=0.923  ← best
✓ Saved: ./user_workspace/checkpoints/compliance/best_model.pt
✓ Indexed 1050 train + 225 val evidence items
```

---

## Step 3 — Train the aggregator

Train the `EvidenceAggregator` network using the existing index from Step 2.

> `--index-dir` and `--data-dir` both point to `./user_workspace/data/compliance` — this is correct.
> The index lives in the data directory.

```bash
epalea train-aggregator \
  --checkpoint ./user_workspace/checkpoints/compliance/best_model.pt \
  --schema ./user_workspace/checkpoints/compliance/schema.json \
  --index-dir ./user_workspace/data/compliance \
  --data-dir ./user_workspace/data/compliance \
  --aggregator-checkpoint-dir ./user_workspace/checkpoints/compliance
```

Or using a config file:

```bash
epalea train-aggregator --config ./user_workspace/configs/train_aggregator.yaml
```

**Produces:**

```
./user_workspace/checkpoints/compliance/
└── aggregator.pt
```

Expected terminal output:
```
Training aggregator...
  Preparing training data: 210 entities
  Epoch 10/30, Loss: 0.412
  Epoch 20/30, Loss: 0.287
  Epoch 30/30, Loss: 0.241
✓ Saved: ./user_workspace/checkpoints/compliance/aggregator.pt
```

**Verify:**
```bash
ls ./user_workspace/checkpoints/compliance/
# best_model.pt  aggregator.pt  schema.json
```

---

## Step 4 — Index test evidence

Test entities are not yet in the index. Add them now before inference.

```bash
epalea index \
  --checkpoint ./user_workspace/checkpoints/compliance/best_model.pt \
  --schema ./user_workspace/checkpoints/compliance/schema.json \
  --evidence ./user_workspace/data/compliance/test_evidence.json \
  --predicate compliance_level \
  --index-dir ./user_workspace/data/compliance
```

Or using a config file:

```bash
epalea index --config ./user_workspace/configs/index.yaml
```

Expected terminal output:
```
Indexing evidence...
  File 1/1: test_evidence.json → 225 items
Rebuilding entity-predicate lookup...
✓ Entities indexed: 225
✓ Evidence items:   225
✓ Saved: ./user_workspace/data/compliance/vector_store.faiss
```

---

## Step 5 — Infer and evaluate

Everything is ready. Run inference on a single entity, then evaluate the full test set.

### Single entity inference

```bash
epalea infer \
  --checkpoint ./user_workspace/checkpoints/compliance/best_model.pt \
  --aggregator-checkpoint ./user_workspace/checkpoints/compliance/aggregator.pt \
  --schema ./user_workspace/checkpoints/compliance/schema.json \
  --index-dir ./user_workspace/data/compliance \
  --entity-id C0001 \
  --mode both
```

Or using a config file + entity flag:

```bash
epalea infer --config ./user_workspace/configs/infer.yaml --entity-id C0001
```

Expected output:

```json
{
  "entity_id": "C0001",
  "mode": "both",
  "results": {
    "spn":        { "prediction": "high", "confidence": 0.84 },
    "aggregator": { "prediction": "high", "confidence": 0.81 }
  },
  "uncertainty": {
    "spn":        { "epistemic": 0.12, "aleatoric": 0.08, "total": 0.20, "weights_source": "uniform" },
    "aggregator": { "epistemic": 0.09, "aleatoric": 0.11, "total": 0.20, "weights_source": "aggregator" }
  },
  "n_evidence_used": 5,
  "execution_time_ms": 31.2
}
```

### Batch inference

```bash
epalea batch-infer \
  --checkpoint ./user_workspace/checkpoints/compliance/best_model.pt \
  --aggregator-checkpoint ./user_workspace/checkpoints/compliance/aggregator.pt \
  --schema ./user_workspace/checkpoints/compliance/schema.json \
  --index-dir ./user_workspace/data/compliance \
  --companies ./user_workspace/data/compliance/test_companies.json \
  --mode both \
  --output-dir ./user_workspace/results/compliance
```

Writes `./user_workspace/results/compliance/batch_predictions.json`.

### Full evaluation

```bash
epalea evaluate \
  --checkpoint ./user_workspace/checkpoints/compliance/best_model.pt \
  --aggregator-checkpoint ./user_workspace/checkpoints/compliance/aggregator.pt \
  --schema ./user_workspace/checkpoints/compliance/schema.json \
  --index-dir ./user_workspace/data/compliance \
  --test-companies ./user_workspace/data/compliance/test_companies.json \
  --mode both \
  --output-dir ./user_workspace/results/compliance
```

Or using a config file:

```bash
epalea evaluate --config ./user_workspace/configs/evaluate.yaml
```

Expected terminal output:

```
── mode: both ────────────────────────────────────────────────────────────────────
System             Acc   Macro F1      NLL    Brier      ECE   Ep.Unc   Al.Unc
─────────────────────────────────────────────────────────────────────────────────
LPF-SPN          0.956      0.946    0.230    0.024    0.032    0.106    0.053
LPF-Agg          0.896      0.881    0.271    0.040    0.066    0.097    0.053
Saved: ./user_workspace/results/compliance/evaluation.json
```

---

## Summary of what you built

```
user_workspace/
├── data/compliance/
│   ├── train_companies.json        ← generated data
│   ├── train_evidence.json
│   ├── val_companies.json
│   ├── val_evidence.json
│   ├── test_companies.json
│   ├── test_evidence.json
│   ├── vector_store.faiss          ← full index (train + val + test)
│   └── metadata.jsonl
├── checkpoints/compliance/
│   ├── best_model.pt               ← LPF encoder + decoder
│   ├── aggregator.pt               ← EvidenceAggregator
│   └── schema.json
└── results/compliance/
    ├── batch_predictions.json
    └── evaluation.json
```

---

## Shortcut: train-full

Steps 2 and 3 can be combined into a single call:

```bash
epalea train-full --config ./user_workspace/configs/train_full.yaml
```

You still need to run Step 4 (`epalea index`) separately before inference.

---

## Next steps

- **Custom domain** — Run the same pipeline with your own predicate and data. See [Custom Domains](./custom-domain.md).
- **Pretrained model** — Skip training entirely. See [Quickstart](./quickstart.md).
- **Notebooks** — Run the interactive version of this walkthrough in [Notebook 02](../notebooks/environment.md).
- **API** — Run inference via HTTP. See [API Overview](../api/overview.md).
