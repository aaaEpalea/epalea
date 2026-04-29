# Changelog

## v1.0.0 — 2025-01-01

### Added

**Package and CLI**
- `epalea` Python package installable via `pip install epalea`
- `epalea.load_model()` — loads all assets, sets `model.available_modes`
- CLI: `info`, `generate-data`, `train`, `index`, `train-aggregator`, `train-full`, `infer`, `batch-infer`, `evaluate`, `models list/download`
- `--config` YAML support for all commands (CLI flags override config values)
- `--mode spn | aggregator | both` on `infer`, `batch-infer`, `evaluate`
- `--output-format nested | flat` on all inference commands
- Prerequisite checking with hard exits and exact fix commands
- `train-full` — single-config combined pipeline (LPF → index → aggregator)
- `--label-key` flag for custom domains where entity field name differs from predicate name

**Models and Data**
- Pretrained: `compliance-v1` — `best_model.pt` + `aggregator.pt` + `evidence_index/`
- `schema.json` with `has_aggregator` and `has_evidence_index` fields
- Config templates: `train.yaml`, `index.yaml`, `train_aggregator.yaml`, `train_full.yaml`, `infer.yaml`, `evaluate.yaml`
- `user_workspace/` isolation with output path guard

**Inference**
- `uncertainty.weights_source` field (`"aggregator"` or `"uniform"`) — transparent decomposition
- Uncertainty decomposition always returned, regardless of mode

**Notebooks**
- `01_quickstart.ipynb` — load model, infer with `--mode both`, understand uncertainty
- `02_training.ipynb` — all four pipeline stages + `train-full` shortcut
- `03_inference.ipynb` — all three modes, metrics comparison table, calibration curves
- `04_custom_domain.ipynb` — full custom domain with `train-full`

**API**
- Live: `POST /api/v1/infer` (SPN mode), `POST /api/v1/infer/batch`, `POST /api/v1/evaluate`
- Live: `GET /api/v1/models`, `GET /api/v1/models/{id}`
- `mode` and `output_format` fields accepted on live endpoints
- Aggregator/both modes return coming-soon JSON + SPN fallback
- Coming-soon stubs: `train`, `index`, `train-aggregator`, `train-full`
- Rate limiting: 20 rpm / 500 rpd (free tier)

**Infrastructure**
- CI: blocks PRs to `pretrained/` and `user_workspace/`
- Apache License 2.0 (`LICENSE`)
- Epalea Model Weight License (`pretrained/WEIGHT_LICENSE.md`)
- `environment.yml` for conda

**Documentation**
- Full Docusaurus docs at `epalea.ai/docs`
- Docs: getting-started, concepts (pipeline, inference modes, uncertainty, evidence index, schema)
- Docs: full CLI reference for all 10 commands
- Docs: API reference with coming-soon endpoint stubs
- Docs: all 10 domain pages
- Interactive API testing at `epalea.ai/docs/api/live-testing`

### Domains

- **Tax Compliance** — full CLI + API + Notebooks
- **All other domains** — web demo at `epalea.ai` only

---

## v1.1.0 — Upcoming

- API: aggregator mode live (`mode: aggregator` and `mode: both`)
- API: `POST /api/v1/models/train`, `/index`, `/train-aggregator`, `/train-full`
- API: account management and API key self-service
- Quantum Tomography domain — CLI + API + all modes
- Healthcare domain — CLI + API + all modes
