# Introduction to Epalea and LPF

> **Epalea** (from *epistemic* + *aleatoric*) is AI trust infrastructure — a framework that helps AI systems know *what they don't know*, and *why*.

---

## The Problem: AI Systems Without Uncertainty

Modern AI systems are either *confident* or *silent*. A language model asserts facts with 94% confidence — even when those facts are hallucinated. A medical classifier returns probabilities — but doesn't tell you whether that uncertainty reflects a data gap (something more evidence could fix) or irreducible ambiguity (something no evidence can resolve).

This distinction matters enormously in practice:

| High epistemic uncertainty | High aleatoric uncertainty |
|---|---|
| *"I haven't seen enough cases like this."* | *"The evidence genuinely conflicts."* |
| ✅ Collect more data | ✅ Escalate to human review |
| ✅ Reducible by more information | ❌ Cannot be resolved by more data |

Without this decomposition, AI systems act on false confidence — or refuse to act at all.

---

## Our Approach: Latent Posterior Factors (LPF)

LPF is a probabilistic inference framework that takes **multiple pieces of evidence** about an entity and produces a **calibrated, decomposed uncertainty estimate** alongside a prediction.

### Core innovation

The key insight is to route each evidence item through a **Variational Autoencoder (VAE)**, converting raw text or structured data into a probability distribution over latent states. These distributions are then converted into **soft factors** — probabilistic constraints that attach to a reasoning engine.

```
Evidence e ──► VAE Encoder ──► q(z|e) ~ N(μ, σ²)
                                   │
                                   │  Monte Carlo sampling
                                   ▼
                              z⁽¹⁾, z⁽²⁾, ..., z⁽ᴹ⁾
                                   │
                                   │  Decode each sample
                                   ▼
                         p(y|z⁽¹⁾), p(y|z⁽²⁾), ..., p(y|z⁽ᴹ⁾)
                                   │
                                   │  Aggregate into soft factor
                                   ▼
                              Φₑ(y) = soft factor for evidence e
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
              LPF-SPN                       LPF-Learned
    (SPN reasoning over factors)      (Neural aggregation)
                    │                             │
                    └──────────────┬──────────────┘
                                   ▼
                         P(y | all evidence)
                    + epistemic / aleatoric decomposition
```

The Monte Carlo sampling step is what makes the uncertainty decomposition **exact** rather than heuristic:

- **Epistemic uncertainty** = variance in the posterior *mean* across evidence items (what the model doesn't know — reducible)
- **Aleatoric uncertainty** = average variance *within* each posterior (what the data disagrees about — irreducible)

---

## Two Architectures

LPF ships two complementary architectures that share the same encoder and soft-factor machinery but differ in how they aggregate across evidence.

### LPF-SPN: Structured probabilistic reasoning

```
Soft factors Φ₁(y), Φ₂(y), ..., Φₙ(y)
           │
           ▼
   Sum-Product Network
   (exact marginal inference)
           │
           ▼
   P(y | e₁, e₂, ..., eₙ)
```

- Converts each posterior into a soft factor attached to an **SPN** (a tractable graphical model)
- Performs **exact marginal inference** — no approximation errors
- **Advantages**: principled, interpretable, best calibration, provable guarantees
- **Best for**: high-stakes decisions requiring audit trails, regulatory compliance

### LPF-Learned: Neural evidence aggregation

```
Posteriors q(z|e₁), q(z|e₂), ..., q(z|eₙ)
           │
           ▼
   Quality scores + consistency scores
   (learned neural weights)
           │
           ▼
   Weighted aggregation in latent space
           │
           ▼  single decoder pass
   P(y | e₁, e₂, ..., eₙ)
```

- Learns **quality** (is this evidence reliable?) and **consistency** (does it agree with other evidence?) scores
- Aggregates posteriors in latent space using learned weights, then decodes *once*
- **Advantages**: simpler to deploy, competitive accuracy, better cross-domain transfer
- **Best for**: production scenarios prioritising simplicity, transfer learning targets

---

## The Uncertainty Decomposition

Both architectures produce the same decomposed uncertainty output after inference:

```json
{
  "uncertainty": {
    "spn": {
      "epistemic": 0.12,
      "aleatoric": 0.08,
      "total": 0.20,
      "weights_source": "uniform"
    },
    "aggregator": {
      "epistemic": 0.09,
      "aleatoric": 0.11,
      "total": 0.20,
      "weights_source": "aggregator"
    }
  }
}
```

Both SPN and Aggregator uncertainty are **always reported separately** — neither is suppressed — so you can compare the two signals and understand where they agree or diverge.

### Interpreting the numbers

| Signal | Meaning | Action |
|---|---|---|
| High SPN epistemic | The structured reasoning engine is uncertain due to sparse or weak evidence | Request more evidence |
| High SPN aleatoric | The evidence itself is genuinely contradictory | Escalate to human review |
| High Aggregator epistemic | The neural aggregator is uncertain — model may not have seen similar cases | Retrain / adapt with more domain data |
| Agreement between SPN and Aggregator | High confidence in the decomposition itself | Trust the signal |
| Disagreement | One architecture is more appropriate for this input distribution | Run an ablation study |

---

## Formal Guarantees

LPF is not a heuristic. It comes with seven published formal theorems:

| Theorem | Guarantee |
|---|---|
| **Calibration preservation** | If each VAE posterior is ε-calibrated, LPF's aggregate is also ε-calibrated |
| **Monte Carlo error control** | ECE ≤ ε + C/√K where K is the number of Monte Carlo samples |
| **PAC-Bayes generalisation** | Non-vacuous generalisation bound on unseen entities |
| **Information-theoretic optimality** | LPF minimises KL divergence from the true posterior under mild assumptions |
| **Corruption robustness** | Performance degrades gracefully at up to 50% corrupted evidence |
| **Sample complexity** | n = O(d log d / ε²) samples sufficient for ε-accurate decomposition |
| **Exact decomposition** | Epistemic + aleatoric = total uncertainty, with <0.002% numerical error |

---

## Validated Domains

LPF has been validated across 10 domains in published research:

| Domain | Best result | Paper |
|---|---|---|
| Regulatory compliance | 94.7% accuracy, ECE 0.4% | LPF v1 |
| LLM fact verification | 64% hallucination reduction | LPF vs LLM paper |
| Autonomous vehicle safety | 100% accuracy, ECE 0.0004% | AV safety paper |
| Quantum state tomography | 50% fewer errors vs MLE | Quantum paper |
| Healthcare triage | Validated on synthetic data | LPF v1 |
| Financial risk | Validated on synthetic data | LPF v1 |
| Legal outcome | Validated on synthetic data | Transfer paper |
| Construction risk | Validated on synthetic data | LPF v1 |
| Grant assessment | Validated on synthetic data | LPF v1 |
| Materials science | Validated on synthetic data | Transfer paper |

---

## Quick Start

### Installation

```bash
pip install epalea
```

### Your first inference

```python
from epalea import LPFPipeline

pipeline = LPFPipeline.from_pretrained("epalea/compliance-v1")

result = pipeline.infer(
    entity_id="company_001",
    evidence=[
        {
            "text_content": "ISO 27001 certified with zero audit findings.",
            "credibility": 0.95,
            "evidence_type": "audit_report",
            "year": 2023,
        },
        {
            "text_content": "Minor GDPR delay remediated within 30 days.",
            "credibility": 0.85,
            "evidence_type": "regulatory_filing",
            "year": 2023,
        },
    ],
    mode="both",  # run both LPF-SPN and LPF-Learned
)

print(result.prediction)               # "high"
print(result.confidence)               # 0.84
print(result.uncertainty.spn)          # ModeUncertainty(epistemic=0.12, aleatoric=0.08, total=0.20)
print(result.uncertainty.aggregator)   # ModeUncertainty(epistemic=0.09, aleatoric=0.11, total=0.20)
```

### Using the CLI

```bash
# Single inference
epalea infer \
  --model compliance-v1 \
  --entity company_001 \
  --evidence evidence.json \
  --mode both

# Evaluate on labeled test data
epalea evaluate \
  --model compliance-v1 \
  --test-data test.json \
  --mode both \
  --output results.csv
```

### Using the REST API

```bash
curl -X POST https://epalea.ai/api/v1/infer \
  -H "X-API-Key: your_key" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "compliance-v1",
    "entity_id": "company_001",
    "evidence": [...],
    "options": {"mode": "both"}
  }'
```

---

## Next Steps

- **[Live API Explorer](/docs/api/live-testing)** — test the API without leaving your browser
- **[Tutorials](/docs/tutorials)** — domain-specific walkthroughs
- **[Research Papers](/docs/research)** — the full publication list
- **[CLI Reference](/docs/cli)** — all commands documented
- **[API Reference](/api/redoc)** — OpenAPI spec with all endpoints
- **[Contributing](/docs/contributing)** — build a domain adapter

---

*Epalea is open-source under the MIT licence. The LPF research is available on arXiv (preprints) and under peer review.*
