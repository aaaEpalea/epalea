# Quickstart — 5 minutes

Load `compliance-v1` and run inference with full uncertainty decomposition.

## 1. Install

```bash
pip install epalea
```

## 2. Verify

```bash
epalea --version   # 1.0.0
epalea info        # system info + model status
```

## 3. Run inference with the pretrained model

```bash
epalea infer \
  --checkpoint ./pretrained/compliance-v1/best_model.pt \
  --aggregator-checkpoint ./pretrained/compliance-v1/aggregator.pt \
  --schema ./pretrained/compliance-v1/schema.json \
  --index-dir ./pretrained/compliance-v1/evidence_index \
  --entity-id C0001 \
  --mode both
```

Output:
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
  "execution_time_ms": 23.4
}
```

Both `uncertainty.spn` and `uncertainty.aggregator` are always returned independently — neither is suppressed.

## 4. Evaluate on the test set

```bash
epalea evaluate \
  --checkpoint ./pretrained/compliance-v1/best_model.pt \
  --aggregator-checkpoint ./pretrained/compliance-v1/aggregator.pt \
  --schema ./pretrained/compliance-v1/schema.json \
  --index-dir ./pretrained/compliance-v1/evidence_index \
  --test-companies ./pretrained/compliance-v1/sample_data/test_companies.json \
  --mode both \
  --output-dir ./user_workspace/results/compliance
```

Output:
```
── mode: both ────────────────────────────────────────────────────────────────────
System             Acc   Macro F1      NLL    Brier      ECE   Ep.Unc   Al.Unc
─────────────────────────────────────────────────────────────────────────────────
LPF-SPN          0.956      0.946    0.230    0.024    0.032    0.106    0.053
LPF-Agg          0.896      0.881    0.271    0.040    0.066    0.097    0.053
```

`Ep.Unc` and `Al.Unc` show mean epistemic and aleatoric uncertainty across the test set per mode.

## 5. Python API

```python
import epalea

model = epalea.load_model("compliance-v1")

result = model.infer(entity_id="C0001", mode="both")
print(result.results.spn.prediction)            # "high"
print(result.uncertainty.spn.epistemic)         # 0.12
print(result.uncertainty.aggregator.epistemic)  # 0.09
```

## 6. Train your own model

See the [full pipeline](../concepts/pipeline.md) or run [Notebook 02](../notebooks/environment.md).
