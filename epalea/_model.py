"""
LoadedModel — High-level interface for pretrained models.
Used by load_model() and notebooks.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class InferenceResult:
    """Result of a single inference call."""

    def __init__(self, data: Dict[str, Any]):
        self._data = data

    @property
    def entity_id(self) -> str:
        return self._data["entity_id"]

    @property
    def mode(self) -> str:
        return self._data["mode"]

    @property
    def results(self):
        return _Namespace(self._data.get("results", {}))

    @property
    def uncertainty(self):
        return _Namespace(self._data.get("uncertainty", {}))

    @property
    def n_evidence_used(self) -> int:
        return self._data.get("n_evidence_used", 0)

    @property
    def execution_time_ms(self) -> float:
        return self._data.get("execution_time_ms", 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return self._data

    def __repr__(self):
        return f"InferenceResult(entity_id={self.entity_id!r}, mode={self.mode!r})"


class _Namespace:
    def __init__(self, d: dict):
        self._d = d
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, _Namespace(v))
            else:
                setattr(self, k, v)

    def __repr__(self):
        return repr(self._d)

    def __getitem__(self, key):
        return self._d[key]

    def get(self, key, default=None):
        return self._d.get(key, default)


class LoadedModel:
    """
    High-level interface for a loaded pretrained model.

    Loaded via epalea.load_model('compliance-v1').
    """

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self.model_id = model_dir.name

        # Load schema
        schema_path = self.model_dir / "schema.json"
        if not schema_path.exists():
            raise FileNotFoundError(f"schema.json not found in {model_dir}")
        with open(schema_path) as f:
            self._schema_data = json.load(f)

        # Determine available modes
        self._has_lpf = (self.model_dir / "best_model.pt").exists()
        self._has_aggregator = (self.model_dir / "aggregator.pt").exists()
        self._has_index = (self.model_dir / "evidence_index" / "vector_store.faiss").exists()

        self._orchestrator = None
        self._loaded = False

        print(f"epalea: Found model '{self.model_id}'")
        print(f"  LPF checkpoint:  {'✓' if self._has_lpf else '✗'}")
        print(f"  Aggregator:      {'✓' if self._has_aggregator else '✗'}")
        print(f"  Evidence index:  {'✓' if self._has_index else '✗'}")
        print(f"  Available modes: {self.available_modes}")

    @property
    def available_modes(self) -> List[str]:
        """Return list of available inference modes."""
        if not (self._has_lpf and self._has_index):
            return []
        modes = ["spn"]
        if self._has_aggregator:
            modes += ["aggregator", "both"]
        return modes

    @property
    def schema(self) -> Dict[str, Any]:
        return self._schema_data

    def _ensure_loaded(self):
        """Lazy-load the full model on first inference call."""
        if self._loaded:
            return
        import torch
        from epalea.models.schema import Schema
        from epalea.models.vae_encoder import VAEEncoderNetwork, VAEEncoder
        from epalea.models.decoder_network import DecoderNetwork
        from epalea.core.evidence_index import EvidenceIndex
        from epalea.core.orchestrator import Orchestrator
        from epalea.core.canonical_db import CanonicalDB
        from epalea.core.provenance_ledger import ProvenanceLedger

        schema = Schema()
        predicate = self._schema_data["predicate"]
        domain_values = self._schema_data["domain_values"]
        variable_name = self._schema_data.get("variable_name", predicate.replace("_level", "").replace("_type", ""))
        schema.add_variable(variable_name, domain_values)
        schema.add_predicate(predicate, [variable_name], domain_values)

        checkpoint = torch.load(self.model_dir / "best_model.pt", map_location="cpu")
        embedding_dim = self._schema_data.get("embedding_dim", 384)
        latent_dim = self._schema_data.get("latent_dim", 64)

        encoder_network = VAEEncoderNetwork(
            embedding_dim=embedding_dim,
            latent_dim=latent_dim,
            hidden_dims=[256, 128]
        )
        encoder_network.load_state_dict(checkpoint["encoder_state_dict"])

        decoder_network = DecoderNetwork(
            latent_dim=latent_dim,
            schema=schema,
            hidden_dims=[128, 64]
        )
        decoder_network.load_state_dict(checkpoint["decoder_state_dict"])

        vae_encoder = VAEEncoder(encoder_network=encoder_network, device="cpu")

        index_dir = self.model_dir / "evidence_index"
        evidence_index = EvidenceIndex(
            embedding_dim=embedding_dim,
            vector_store_path=str(index_dir / "vector_store.faiss"),
            metadata_store_path=str(index_dir / "metadata.jsonl"),
        )
        evidence_index._rebuild_entity_index()

        canonical_db = CanonicalDB()
        ledger = ProvenanceLedger()

        self._orchestrator = Orchestrator(
            schema=schema,
            canonical_db=canonical_db,
            evidence_index=evidence_index,
            vae_encoder=vae_encoder,
            decoder_network=decoder_network,
            provenance_ledger=ledger,
            device="cpu",
        )

        if self._has_aggregator:
            from epalea.models.learned_aggregator import add_learned_aggregation_to_orchestrator
            add_learned_aggregation_to_orchestrator(self._orchestrator, latent_dim=latent_dim)
            trainer = getattr(self._orchestrator, "aggregator_trainer", None)
            if trainer:
                trainer.load_checkpoint(str(self.model_dir / "aggregator.pt"))

        self._predicate = predicate
        self._loaded = True
        print(f"epalea: Model '{self.model_id}' loaded successfully.")

    def infer(
        self,
        entity_id: str,
        evidence_file: Optional[str] = None,
        mode: str = "spn",
        output_format: str = "nested",
        top_k: int = 5,
        n_samples: int = 16,
        temperature: float = 0.8,
        alpha: float = 0.1,
    ) -> InferenceResult:
        import sys
        from pathlib import Path
        # Use the same _run_single_infer that the CLI uses — known working path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from epalea.cli import _run_single_infer

        p = {
            "checkpoint": str(self.model_dir / "best_model.pt"),
            "aggregator_checkpoint": str(self.model_dir / "aggregator.pt") if self._has_aggregator else None,
            "schema": str(self.model_dir / "schema.json"),
            "index_dir": str(self.model_dir / "evidence_index"),
            "entity_id": entity_id,
            "mode": mode,
            "output_format": output_format,
            "top_k": top_k,
            "n_samples": n_samples,
            "temperature": temperature,
            "alpha": alpha,
        }

        data = _run_single_infer(p)
        # LoadedModel adds model_id to the result, CLI doesn't
        data["model_id"] = self.model_id
        return InferenceResult(data)

    def batch_infer(
        self,
        companies_file: str,
        evidence_file: Optional[str] = None,
        mode: str = "spn",
        output_format: str = "nested",
        top_k: int = 5,
        n_samples: int = 4,
    ) -> List[InferenceResult]:
        """Run inference for all entities in a companies JSON file."""
        import json

        with open(companies_file) as f:
            companies = json.load(f)

        results = []
        for company in companies:
            entity_id = company.get("company_id") or company.get("entity_id")
            try:
                result = self.infer(
                    entity_id=entity_id,
                    mode=mode,
                    output_format=output_format,
                    top_k=top_k,
                    n_samples=n_samples,
                )
                results.append(result)
            except Exception as e:
                print(f"  Warning: failed on {entity_id}: {e}")

        return results

    def plot_calibration(self, results: List[InferenceResult], mode: str = "spn"):
        """Plot calibration curve for inference results (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("matplotlib required: pip install matplotlib")
            return

        print(f"Calibration plot for mode={mode} — {len(results)} entities")
        print("(Implement full calibration curve with evaluation ground truth labels)")


def _flatten_result(data: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested result to flat format."""
    flat = {
        "entity_id": data["entity_id"],
        "model_id": data.get("model_id", ""),
        "mode": data["mode"],
    }
    for mode_key, mode_data in data.get("results", {}).items():
        flat[f"{mode_key}_prediction"] = mode_data.get("prediction")
        flat[f"{mode_key}_confidence"] = mode_data.get("confidence")
        flat[f"{mode_key}_distribution"] = mode_data.get("distribution")
    unc = data.get("uncertainty", {})
    # New per-mode format: {"spn": {...}, "aggregator": {...}}
    if "spn" in unc or "aggregator" in unc:
        spn_u = unc.get("spn") or {}
        agg_u = unc.get("aggregator") or {}
        flat["spn_epistemic"] = spn_u.get("epistemic")
        flat["spn_aleatoric"] = spn_u.get("aleatoric")
        flat["spn_total_uncertainty"] = spn_u.get("total")
        flat["spn_decomposition_error"] = spn_u.get("decomposition_error")
        flat["aggregator_epistemic"] = agg_u.get("epistemic")
        flat["aggregator_aleatoric"] = agg_u.get("aleatoric")
        flat["aggregator_total_uncertainty"] = agg_u.get("total")
        flat["aggregator_decomposition_error"] = agg_u.get("decomposition_error")
    else:
        # Legacy flat uncertainty — backward compat
        flat["epistemic_uncertainty"] = unc.get("epistemic")
        flat["aleatoric_uncertainty"] = unc.get("aleatoric")
        flat["total_uncertainty"] = unc.get("total")
        flat["decomposition_error"] = unc.get("decomposition_error")
        flat["weights_source"] = unc.get("weights_source")
    flat["n_evidence_used"] = data.get("n_evidence_used")
    flat["execution_time_ms"] = data.get("execution_time_ms")
    return flat
