# Uncertainty Decomposition

Every inference result includes a full uncertainty decomposition. When `mode="both"`, uncertainty is reported **separately for each architecture** — LPF-SPN and LPF-Learned — so neither signal is suppressed and you can compare them directly.

## Fields — `mode="both"`

```json
{
  "uncertainty": {
    "spn": {
      "epistemic": 0.12,
      "aleatoric": 0.08,
      "total": 0.20,
      "decomposition_error": 0.001,
      "weights_source": "uniform"
    },
    "aggregator": {
      "epistemic": 0.09,
      "aleatoric": 0.11,
      "total": 0.20,
      "decomposition_error": 0.001,
      "weights_source": "aggregator"
    }
  }
}
```

## Fields — `mode="spn"` or `mode="aggregator"`

When running a single mode, only the relevant key is present:

```json
{ "uncertainty": { "spn": { "epistemic": 0.12, "aleatoric": 0.08, "total": 0.20, ... } } }
```

## Field reference

| Field | Meaning |
|---|---|
| `epistemic` | Model uncertainty — reducible with more data or better training |
| `aleatoric` | Data uncertainty — irreducible noise inherent in the evidence |
| `total` | `epistemic + aleatoric` (always exact to <0.002%) |
| `decomposition_error` | Numerical error from Monte Carlo sampling |
| `weights_source` | `"uniform"` for SPN, `"aggregator"` when learned weights were used |

## Interpreting the numbers

| Pattern | Meaning | Action |
|---|---|---|
| High `spn.epistemic` | SPN has sparse / weak evidence | Request more evidence |
| High `spn.aleatoric` | Evidence itself is genuinely contradictory | Escalate to human review |
| High `aggregator.epistemic` | Neural aggregator hasn't seen cases like this | Retrain / adapt with more domain data |
| `spn` and `aggregator` agree | High confidence in the decomposition | Trust the signal |
| `spn` and `aggregator` disagree | One architecture fits this input distribution better | Run an ablation study |

## Python API

```python
result = model.infer(entity_id="C0001", mode="both")

# Access per-mode uncertainty
print(result.uncertainty.spn.epistemic)          # 0.12
print(result.uncertainty.spn.aleatoric)          # 0.08
print(result.uncertainty.aggregator.epistemic)   # 0.09
print(result.uncertainty.aggregator.aleatoric)   # 0.11
```

## Flat output format

When `output_format="flat"`, per-mode uncertainty is expanded into prefixed columns:

```json
{
  "spn_epistemic": 0.12,
  "spn_aleatoric": 0.08,
  "spn_total_uncertainty": 0.20,
  "aggregator_epistemic": 0.09,
  "aggregator_aleatoric": 0.11,
  "aggregator_total_uncertainty": 0.20
}
```
