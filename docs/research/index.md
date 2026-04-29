# Research

Epalea is built on a body of peer-reviewed research spanning ML theory, probabilistic reasoning, autonomous systems, quantum computing, clinical AI, and legal reasoning. All papers are available as preprints; peer review is ongoing.

---

## Publication List

### Paper 1 — Core Architecture

**I Know What I Don't Know: Latent Posterior Factor Models for Multi-Evidence Probabilistic Reasoning**

*Architecture · Neuro-Symbolic*

Introduces LPF and its two complementary architectures — **LPF-SPN** (structured factor-based inference) and **LPF-Learned** (end-to-end neural aggregation). Evaluated across eight domains including FEVER fact verification, achieving up to **97.8% accuracy** and **ECE 1.4%**.

**Key results**: 97.8% accuracy · ECE 1.4% · 8 domains including FEVER

[↗ arXiv preprint](https://arxiv.org/abs/coming-soon)

---

### Paper 2 — Theoretical Foundations

**Theoretical Foundations of Latent Posterior Factors: Formal Guarantees for Multi-Evidence Reasoning**

*Theory · Formal Guarantees*

Proves seven formal theorems establishing LPF on rigorous mathematical ground: calibration preservation, Monte Carlo error control (ECE ≤ ε + C/√K), PAC-Bayes non-vacuous generalisation bounds, information-theoretic optimality, corruption robustness, sample complexity, and exact uncertainty decomposition.

**Key results**: 7 formal theorems · ECE ≤ ε + C/√K · PAC-Bayes non-vacuous bound

[↗ arXiv preprint](https://arxiv.org/abs/coming-soon)

---

### Paper 3 — Architectural Ablation

**Dissecting Hybrid Neuro-Symbolic Systems: An Architectural Ablation Study of the Latent Probabilistic Framework**

*Ablation · Neuro-Symbolic*

Systematic ablation across 15 random seeds (n=135 each) identifies the SPN reasoning module as the most critical component. Removing the SPN causes a **6.1pp accuracy drop** and nearly **triples calibration error** (ECE 2.3% → 6.1%). The VAE latent averaging baseline achieves 95.6%, confirming the value of the full pipeline.

**Key results**: 15 seeds · n=135 each · SPN removal: ECE 2.3% → 6.1% · VAE latent avg: 95.6%

[↗ arXiv preprint](https://arxiv.org/abs/coming-soon)

---

### Paper 4 — LLM Comparison

**Exact Uncertainty Decomposition for Multi-Evidence Fact Verification: A Formal Alternative to LLM-Based Reasoning**

*Fact Verification · LLM Comparison*

Benchmarks LPF against Qwen-2.5-3B and Llama-3.2-3B on conflicting-evidence datasets. LPF achieves a **64% hallucination reduction** vs Qwen and **54%** vs Llama — with **zero LLM calls** at inference time. Throughput: **1.03 queries/sec** vs baselines' 0.5–0.8 q/s.

**Key results**: 64% hallucination reduction · 0 LLM calls at inference · 1.03 q/s throughput

[↗ arXiv preprint](https://arxiv.org/abs/coming-soon)

---

### Paper 5 — Autonomous Systems

**Safety-Critical Decision Making for Autonomous Machines: A Probabilistic Framework with Uncertainty-Based Human Handoff**

*Autonomous Systems · Safety*

Deploys LPF for autonomous vehicles, surgical robotics, drones, and warehouse automation. Achieves **100% accuracy** with **ECE 0.0004%**. In a critical scenario (fog + night + conflicting sensors), LPF-enabled vehicles detect high epistemic uncertainty and stop 300 units before an ambiguous obstacle — while baselines crash at 50 units. **Zero missed human escalations** across 200 test scenarios.

**Key results**: ECE 0.0004% · 100% accuracy · Zero missed humans · Stops 300 units vs 50

[↗ arXiv preprint](https://arxiv.org/abs/coming-soon)

---

### Paper 6 — Quantum Computing

**Uncertainty-Aware Quantum State Tomography via Latent Posterior Factors: A Multi-Evidence Aggregation Approach**

*Quantum Computing · Physics*

Applies LPF to single-qubit state tomography. Achieves **99.8% classification accuracy** and **0.979 fidelity** while making **50% fewer errors** than maximum likelihood estimation. The epistemic/aleatoric decomposition reveals that 46% of total uncertainty is reducible through additional measurements — enabling adaptive protocols that cut required measurements by 10%.

**Key results**: 50% fewer errors vs MLE · 10% measurement reduction · ECE 0.17% · Fidelity 0.979

[↗ arXiv preprint](https://arxiv.org/abs/coming-soon)

---

### Paper 7 — Robustness

**Robustness Under Noise: A Comprehensive Analysis of Latent Factor Posterior Models in Noisy Compliance Environments**

*Robustness · Noise Analysis*

Evaluates LPF across five noise configurations — from clean data to 70% feature noise and 40% contradictory evidence — each run across 15 random seeds. Establishes noise tolerance thresholds and characterises how epistemic vs aleatoric uncertainty respond differently to noise type and magnitude.

**Key results**: 15 seeds per configuration · 5 noise types · 70% feature noise tolerance · 40% contradictory evidence

[↗ arXiv preprint](https://arxiv.org/abs/coming-soon)

---

### Paper 8 — Transfer Learning

**Cross-Domain Transfer and Calibration Analysis for Latent Probabilistic Frameworks**

*Transfer Learning · Cross-Domain*

Tests zero-shot and few-shot transfer from the compliance domain to six targets: healthcare, academic, construction, legal, finance, and materials science. **LPF-Learned** achieves **42.5% zero-shot average** vs LPF-SPN's 29.0%, confirming the neural aggregator as the better transfer architecture. 100-shot adaptation brings both to >90% across all targets.

**Key results**: 6 target domains · 42.5% zero-shot avg (Learned) · 100-shot adaptation · Calibration maintained across domains

[↗ arXiv preprint](https://arxiv.org/abs/coming-soon)

---

### Paper 9 — Scalability

**Scalability of Latent Factor Posteriors to Varying Evidence Pool Sizes**

*Scalability · Evidence Pools*

Empirical study spanning evidence pools from **10 to 500 items per entity** — a 50× range. LPF-SPN maintains ECE 0.050–0.163 with 14–15ms inference throughout. LPF-Learned reaches **98.5–100% accuracy** at the top of the range. Establishes practical guidance on evidence pool sizing for production deployments.

**Key results**: 10–500 evidence items (50× range) · 98.5–100% accuracy · 14–15ms inference at scale

[↗ arXiv preprint](https://arxiv.org/abs/coming-soon)

---

### Paper 10 — Complete Characterisation

**Complete Theoretical Characterization of Latent Probabilistic Fusion (LPF)**

*Theory · Complete Characterization*

Validates all seven formal guarantees on 4,200 training and 900 test examples. LPF achieves **<0.002% decomposition error** — making reported uncertainty statistically rigorous rather than a heuristic. Performance is maintained at 88% even at 50% corruption. The ECE bound is achieved with 82% margin below the theoretical limit.

**Key results**: <0.002% decomposition error · 82% margin below ECE bound · 88% performance at 50% corruption

[↗ arXiv preprint](https://arxiv.org/abs/coming-soon)

---

## Research Themes

| Theme | Papers |
|---|---|
| Formal guarantees and theory | Papers 2, 10 |
| Core architecture | Papers 1, 3 |
| LLM / hallucination | Paper 4 |
| Safety-critical systems | Paper 5 |
| Quantum computing | Paper 6 |
| Robustness and noise | Paper 7 |
| Transfer and generalisation | Papers 8, 9 |

---

## Citation

If you use Epalea or LPF in your research, please cite:

```bibtex
@misc{epalea2025lpf,
  title  = {I Know What I Don't Know: Latent Posterior Factor Models for Multi-Evidence Probabilistic Reasoning},
  author = {Epalea Research Team},
  year   = {2025},
  eprint = {coming-soon},
  archivePrefix = {arXiv},
}
```

---

## Contributing to Research

We welcome research contributions — new formal results, novel applications, architectural improvements. Please read our [Contributing Guide](/docs/contributing#research-extensions) before submitting.
