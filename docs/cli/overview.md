# CLI Overview

```bash
epalea --help
epalea --version   # 1.0.0
epalea info        # system info, pretrained models, workspace status
```

## Two ways to run every command

Every command accepts either a config file or inline flags. CLI flags always override config values.

```bash
# Config file
epalea train --config ./user_workspace/configs/train.yaml

# Inline flags
epalea train \
  --domain compliance \
  --predicate compliance_level \
  --domain-values low --domain-values medium --domain-values high \
  --data-dir [./user_workspace/data/domain] \
  --checkpoint-dir [./user_workspace/checkpoints/domain]

# Mix — config base, override one flag
epalea infer --config ./user_workspace/configs/infer.yaml --entity-id C0042
```

Config files live in `./user_workspace/configs/`. See the [config reference](../configs/) for all available keys.

## The pipeline

```
Step 1  epalea generate-data
            ↓
Step 2  epalea train         ← produces best_model.pt
                               also indexes train + val evidence into data_dir
            ↓
Step 3  epalea train-aggregator   ← uses the index from Step 2
                                    produces aggregator.pt
            ↓
Step 4  epalea index         ← adds test evidence to the existing index
            ↓
Step 5  epalea infer / batch-infer / evaluate
```

`epalea train-full` combines Steps 2 and 3 into one call. You still run Step 4 separately before inference.

Failures show the exact fix command:

```
✗  Evidence index not found at ./user_workspace/data/compliance/vector_store.faiss
   Run first:
     epalea index --checkpoint [./user_workspace/checkpoints/domain/best_model.pt] ...
```

## Output path protection

All output must go into `./user_workspace/`. Writing into `pretrained/` raises a hard error.

## Command reference

| Command | Description |
|---|---|
| `epalea --version` | Print version and exit |
| `epalea info` | System info, models, workspace status |
| `epalea models list` | List pretrained models |
| `epalea models download <id>` | Download a pretrained model |
| `epalea generate-data` | Generate synthetic training data |
| `epalea train` | Train LPF encoder + decoder (also indexes train/val) |
| `epalea train-aggregator` | Train aggregator (uses existing index) |
| `epalea train-full` | Run train + train-aggregator together |
| `epalea index` | Add evidence to existing index (use for test split) |
| `epalea infer` | Single entity inference |
| `epalea batch-infer` | Batch inference over a companies file |
| `epalea evaluate` | Batch inference + full metrics |
