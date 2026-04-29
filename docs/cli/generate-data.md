# epalea generate-data

Generate synthetic training data for a domain.

## Usage

```bash
# Inline
epalea generate-data \
  --domain [domain] \
  --n-entities [300] \
  --years [2020] --years [2021] --years [2022] \
  --evidence-per-entity [5] \
  --noise-level [0.1] \
  --contradictory-rate [0.05] \
  --output-dir [./user_workspace/data]
```

`generate-data` does not accept a `--config` file. All parameters are passed inline.

## Options

| Flag | Type | Default | Description |
|---|---|---|---|
| `--domain` | str | *required* | Domain name — e.g. `compliance`, `loan_risk` |
| `--predicate` | str | — | Predicate name (custom domains) |
| `--domain-values` | list | — | Class values (custom domains) |
| `--n-entities` | int | `300` | Number of entities to generate |
| `--years` | list | `[2022]` | Fiscal years — pass once per year |
| `--evidence-per-entity` | int | `5` | Evidence items per entity |
| `--noise-level` | float | `0.1` | Label noise level `[0, 1]` |
| `--contradictory-rate` | float | `0.05` | Fraction of contradictory evidence items |
| `--output-dir` | path | `./user_workspace/data` | Root output directory |

## Output structure

```
[output-dir]/[domain]/
├── train_companies.json
├── train_evidence.json
├── val_companies.json
├── val_evidence.json
├── test_companies.json
└── test_evidence.json
```

## Built-in domain: compliance

```bash
epalea generate-data \
  --domain compliance \
  --n-entities 300 \
  --years 2020 --years 2021 --years 2022 \
  --output-dir ./user_workspace/data
```

## Custom domain

Pass `--predicate` and `--domain-values` for any domain not built-in:

```bash
epalea generate-data \
  --domain loan_risk \
  --predicate risk_tier \
  --domain-values low --domain-values medium --domain-values high \
  --n-entities 500 \
  --years 2022 --years 2023 \
  --output-dir ./user_workspace/data
```

Output lands in `./user_workspace/data/loan_risk/`.

> For fully custom domains with your own data, skip `generate-data` entirely and place your own `train_companies.json`, `train_evidence.json`, etc. directly in `./user_workspace/data/[domain]/`. See [Custom Domain](../domains/custom.md).
