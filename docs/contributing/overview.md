# Contributing to Epalea

We welcome contributions across three categories: **domain adapters**, **core improvements**, and **research extensions**. All contributions require a brief issue discussion before implementation.

## What We Welcome

| ✅ Welcome | ⚠️ Needs Discussion | ❌ Not Accepted |
|---|---|---|
| Bug fixes in `core/` and `models/` | New domain presets | Changes to `pretrained/` |
| Documentation improvements | New CLI commands | Personal model weights |
| Notebook improvements | Architecture changes | `user_workspace/` changes |
| CLI usability fixes | | |
| Test coverage | | |

## CI Enforcement

PRs that modify `pretrained/` or `user_workspace/` will **automatically fail CI**.
This is enforced in `.github/workflows/protect_pretrained.yml`.

## License

All contributions are automatically licensed under Apache 2.0. No CLA required.

---

## How to Contribute

1. **Open an issue first.** Describe what you want to build. We'll confirm scope and avoid duplicate work.
2. **Fork the repo** and create a branch: `git checkout -b feat/my-contribution`
3. **Implement and test.** All contributions must include tests and pass `pytest` cleanly.
4. **Open a PR** referencing the issue.

---

## Domain Adapters — Primary Contribution Path

A domain adapter lets LPF operate in a new application domain without modifying the core framework.

### Currently available

| Domain | Status |
|---|---|
| Compliance (EU AI Act) | ✅ Production |
| LLM Hallucination | ✅ Production |
| Quantum Tomography | ✅ Production |
| AV Safety | ✅ Production |
| Healthcare Triage | 🔬 Pilot |
| Financial Risk | 🔬 Pilot |
| Legal Outcome | 🔬 Evaluation |
| Grants Assessment | 🔬 Evaluation |
| Construction Risk | 🔬 Evaluation |
| Materials Science | 🔬 Evaluation |

### Domains we'd particularly welcome

- Climate / environmental risk
- Education assessment
- Supply chain disruption
- Cybersecurity threat classification
- Drug discovery / biomarker classification

### Checklist

- [ ] `schema.json` with `domain_values`, `predicate`, `description`, `released`
- [ ] `EvidenceAdapter` subclass with `preprocess_evidence` and `get_prompt_template`
- [ ] `SyntheticDataGenerator` subclass
- [ ] `EvalConfig` specifying primary metric and thresholds
- [ ] Working notebook in `adapters/my_domain/notebooks/`
- [ ] Unit tests covering edge cases
- [ ] `README.md` in the adapter directory

**Safety-critical domains** (medical, financial, autonomous systems) require a scoping discussion before implementation. Open an issue tagged `[domain-adapter]`.

---

## Core Improvements

We welcome improvements to the shared LPF machinery — after issue discussion. Areas include:

**Calibration** — temperature scaling, post-hoc recalibration, new ECE estimators.

**Uncertainty decomposition** — alternative epistemic/aleatoric decomposition methods, ensemble-based epistemic estimation, conformal prediction wrappers.

**Evidence retrieval** — new embedding models, hybrid dense/sparse retrieval, cross-lingual retrieval.

**Architecture** — new SPN structure types, alternative VAE architectures, efficient batch inference kernels.

Open an issue tagged `[core]` with your motivation and proposed approach.

---

## Research Extensions

To extend LPF's theoretical foundations — new formal guarantees, novel applications, new decomposition methods — open an issue tagged `[research]` with a 1–2 page write-up describing the research question, method, evaluation plan, and citations. We respond within two weeks.

---

## Development Setup

```bash
git clone https://github.com/epalea/epalea.git
cd epalea
pip install -e ".[dev]"
pytest tests/ -v
ruff check . && ruff format .
```

---

## PR Guidelines

- One concern per PR
- Tests required — PRs without tests will not be merged
- Add a `CHANGELOG.md` entry under `[Unreleased]`
- Add docs for any new public API or adapter

### PR title format

```
feat(adapter): add climate risk domain adapter
fix(core): correct aleatoric uncertainty in zero-evidence case
docs(api): update evaluate endpoint for per-mode uncertainty
```

---

## Questions?

Open an issue or email `research@epalea.ai` or `team@epalea.ai`. We respond within 48 hours.
