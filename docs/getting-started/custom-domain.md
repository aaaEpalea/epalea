# Custom Domains

LPF is domain-agnostic. You can train a model on any classification problem where multiple pieces of evidence inform a single label. No changes to the framework are needed — you provide a domain name, a predicate, and class values, and the pipeline handles the rest.

## What is a domain?

A domain is a classification task defined by:

| Concept | Description | Example |
|---|---|---|
| **Domain** | The application area | `loan_risk` |
| **Predicate** | The question being answered | `risk_tier` |
| **Domain values** | Possible answers | `low`, `medium`, `high` |

The compliance domain that ships with `compliance-v1` is one example. You can define your own.

## Built-in domains vs custom domains

| | Built-in (`compliance`) | Custom |
|---|---|---|
| `generate-data` | Pass `--domain compliance` | Pass `--domain [name] --predicate [p] --domain-values [...]` |
| Training | Identical | Identical |
| Pretrained weights | Available (`compliance-v1`) | You train from scratch |

## Defining a custom domain

You do not need a config file for the domain itself — you pass the domain definition directly as flags.

### Step 1 — Generate data

```bash
epalea generate-data \
  --domain [your_domain] \
  --predicate [your_predicate] \
  --domain-values [value1] --domain-values [value2] --domain-values [value3] \
  --n-entities [300] \
  --years [2022] --years [2023] \
  --evidence-per-entity [5] \
  --noise-level [0.1] \
  --output-dir [./user_workspace/data]
```

Output lands in `./user_workspace/data/[your_domain]/`.

### Using your own data

If you have real data, skip `generate-data` entirely. Place your files in the expected structure:

```
./user_workspace/data/[your_domain]/
├── train_companies.json    ← list of {"company_id": ..., "[predicate]": "value", ...}
├── train_evidence.json     ← list of {"evidence_id": ..., "company_id": ..., "text_content": ..., ...}
├── val_companies.json
├── val_evidence.json
├── test_companies.json
└── test_evidence.json
```

### Step 2 — Train

Provide your domain, predicate, and domain values. Everything else is identical to the compliance workflow:

```bash
epalea train \
  --domain [your_domain] \
  --predicate [your_predicate] \
  --domain-values [value1] --domain-values [value2] --domain-values [value3] \
  --data-dir [./user_workspace/data/your_domain] \
  --checkpoint-dir [./user_workspace/checkpoints/your_domain]
```

### Steps 3–5

Identical to any other domain — just swap in your paths. See [Try It Now](./tutorial.md) for a complete walkthrough of all five steps.

## Example — Loan risk

```bash
# Step 1: Generate data
epalea generate-data \
  --domain loan_risk \
  --predicate risk_tier \
  --domain-values low --domain-values medium --domain-values high \
  --n-entities 500 \
  --years 2022 --years 2023 \
  --output-dir ./user_workspace/data

# Step 2: Train
epalea train \
  --domain loan_risk \
  --predicate risk_tier \
  --domain-values low --domain-values medium --domain-values high \
  --data-dir ./user_workspace/data/loan_risk \
  --checkpoint-dir ./user_workspace/checkpoints/loan_risk

# Step 3: Train aggregator
epalea train-aggregator \
  --checkpoint ./user_workspace/checkpoints/loan_risk/best_model.pt \
  --schema ./user_workspace/checkpoints/loan_risk/schema.json \
  --index-dir ./user_workspace/data/loan_risk \
  --data-dir ./user_workspace/data/loan_risk \
  --aggregator-checkpoint-dir ./user_workspace/checkpoints/loan_risk

# Step 4: Index test evidence
epalea index \
  --checkpoint ./user_workspace/checkpoints/loan_risk/best_model.pt \
  --schema ./user_workspace/checkpoints/loan_risk/schema.json \
  --evidence ./user_workspace/data/loan_risk/test_evidence.json \
  --predicate risk_tier \
  --index-dir ./user_workspace/data/loan_risk

# Step 5: Infer
epalea infer \
  --checkpoint ./user_workspace/checkpoints/loan_risk/best_model.pt \
  --aggregator-checkpoint ./user_workspace/checkpoints/loan_risk/aggregator.pt \
  --schema ./user_workspace/checkpoints/loan_risk/schema.json \
  --index-dir ./user_workspace/data/loan_risk \
  --entity-id C0001 \
  --mode both
```

---

Ready to follow a complete walkthrough? The next page walks through all five steps on the compliance domain with exact commands you can run immediately.

→ **[Try It Now — Full Pipeline Walkthrough](./tutorial.md)**
