# epalea index

Add evidence to the existing FAISS index.

## When to use this

`epalea train` automatically indexes train and val evidence during training. You only need `epalea index` to **add test evidence** before running inference or evaluate.

## Usage

```bash
# Config file (recommended)
epalea index --config ./user_workspace/configs/index.yaml

# Inline flags
epalea index \
  --checkpoint [./user_workspace/checkpoints/domain/best_model.pt] \
  --schema [./user_workspace/checkpoints/domain/schema.json] \
  --evidence [./user_workspace/data/domain/test_evidence.json] \
  --predicate [predicate_name] \
  --index-dir [./user_workspace/data/domain]
```

## Config file — `index.yaml`

```yaml
checkpoint: ./user_workspace/checkpoints/compliance/best_model.pt
schema: ./user_workspace/checkpoints/compliance/schema.json
evidence_files:
  - ./user_workspace/data/compliance/test_evidence.json
predicate: compliance_level

# index_dir must match the data_dir used during training —
# that is where epalea train wrote the index.
index_dir: ./user_workspace/data/compliance
embedding_dim: 384
```

## Prerequisites

- `best_model.pt` from `epalea train`
- Existing index at `index-dir` created by `epalea train`

## Options

| Flag | Type | Default | Description |
|---|---|---|---|
| `--config` | path | — | YAML config file |
| `--checkpoint` | path | *required* | `best_model.pt` |
| `--schema` | path | — | `schema.json` — auto-detected from checkpoint dir |
| `--evidence` | paths | *required* | One or more evidence JSON files |
| `--predicate` | str | *required* | Predicate name |
| `--index-dir` | path | *required* | Index directory — must already exist from training |
| `--embedding-dim` | int | `384` | Must match training |

## Output

```
Indexing evidence...
  File 1/1: test_evidence.json → 450 items
Rebuilding entity-predicate lookup...
Saving index...

✓ Entities indexed: 450
✓ Evidence items:   450
✓ Sample lookup: C0201 → compliance_level → 5 items ✓
✓ Saved: [index-dir]/vector_store.faiss
✓ Saved: [index-dir]/metadata.jsonl
✓ Updated schema.json (has_evidence_index=true)
```

This appends to the existing index — it does not recreate it from scratch.
