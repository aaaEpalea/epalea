# Inference Modes

The `--mode` flag controls which inference path is used.

## `--mode spn` (default)

Uses the Sum-Product Network path (`use_spn=True` in the orchestrator).

- **When to use:** Default choice. The primary research contribution.
- **Requires:** `best_model.pt` + `evidence_index/`
- **Aggregator needed?** No
- `uncertainty.weights_source` will be `"uniform"`

## `--mode aggregator`

Uses the learned aggregation path (`use_spn=False` with monkey-patched `learned_aggregate`).

- **When to use:** When you have trained the aggregator and want maximum predictive quality.
- **Requires:** `best_model.pt` + `evidence_index/` + `aggregator.pt`
- `uncertainty.weights_source` will be `"aggregator"`

## `--mode both`

Runs both paths and returns side-by-side results.

- **When to use:** Comparison, ablation studies, research.
- **Requires:** `best_model.pt` + `evidence_index/` + `aggregator.pt`
- Returns `results.spn` and `results.aggregator` in the same response

## Output Format

Use `--output-format nested` (default) or `--output-format flat`:

**nested:**
```json
{
  "results": {
    "spn": {"prediction": "high", "confidence": 0.84, "distribution": {...}},
    "aggregator": {"prediction": "high", "confidence": 0.81, "distribution": {...}}
  }
}
```

**flat:**
```json
{
  "spn_prediction": "high",
  "spn_confidence": 0.84,
  "aggregator_prediction": "high",
  "aggregator_confidence": 0.81
}
```
