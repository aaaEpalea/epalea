"""
Shared inference logic used by both the Python API and CLI.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional


def run_inference(
    orchestrator,
    entity_id: str,
    predicate: str,
    mode: str = "spn",
    output_format: str = "nested",
    top_k: int = 5,
    n_samples: int = 4,
    temperature: float = 0.8,
    alpha: float = 0.1,
) -> Dict[str, Any]:
    """
    Core inference function. Runs one or both inference modes and
    assembles the output dict in nested or flat format.

    Uncertainty is always decomposed per mode:
      - uncertainty["spn"]        → epistemic/aleatoric from LPF-SPN
      - uncertainty["aggregator"] → epistemic/aleatoric from LPF-Learned
    Neither signal is suppressed when mode='both'.
    """
    from epalea.core.orchestrator import QueryOptions

    def _query(use_spn: bool):
        options = QueryOptions(
            top_k=top_k, n_samples=n_samples, temperature=temperature,
            alpha=alpha, use_canonical=True, use_spn=use_spn,
        )
        t0 = time.perf_counter()
        result = orchestrator.query(entity_id, predicate, options, return_uncertainty=True)
        elapsed = (time.perf_counter() - t0) * 1000
        return result, elapsed

    n_ev = len(orchestrator.evidence_index.search(entity_id, predicate, top_k=top_k))

    if mode == "spn":
        result, elapsed = _query(use_spn=True)
        per_mode_unc = {"spn": _extract_uncertainty(result, "uniform"), "aggregator": None}
        return _format(entity_id, mode, output_format,
                       spn=_extract_result(result), aggregator=None,
                       uncertainty=per_mode_unc, n_evidence=n_ev, elapsed=elapsed)

    elif mode == "aggregator":
        result, elapsed = _query(use_spn=False)
        per_mode_unc = {"spn": None, "aggregator": _extract_uncertainty(result, "aggregator")}
        return _format(entity_id, mode, output_format,
                       spn=None, aggregator=_extract_result(result),
                       uncertainty=per_mode_unc, n_evidence=n_ev, elapsed=elapsed)

    elif mode == "both":
        t0 = time.perf_counter()
        spn_raw, _ = _query(use_spn=True)
        agg_raw, _ = _query(use_spn=False)
        elapsed = (time.perf_counter() - t0) * 1000
        # Both modes: return uncertainty for each independently
        per_mode_unc = {
            "spn": _extract_uncertainty(spn_raw, "uniform"),
            "aggregator": _extract_uncertainty(agg_raw, "aggregator"),
        }
        return _format(entity_id, mode, output_format,
                       spn=_extract_result(spn_raw), aggregator=_extract_result(agg_raw),
                       uncertainty=per_mode_unc, n_evidence=n_ev, elapsed=elapsed)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")


def _extract_result(raw: Dict) -> Dict[str, Any]:
    return {
        "prediction": raw.get("top_value", "unknown"),
        "confidence": round(float(raw.get("confidence", 0.0)), 4),
        "distribution": {k: round(float(v), 4) for k, v in raw.get("distribution", {}).items()},
    }


def _extract_uncertainty(raw: Dict, weights_source: str) -> Dict[str, Any]:
    ep = float(raw.get("epistemic_uncertainty", 0.0))
    al = float(raw.get("aleatoric_uncertainty", 0.0))
    return {
        "epistemic": round(ep, 4),
        "aleatoric": round(al, 4),
        "total": round(ep + al, 4),
        "decomposition_error": round(float(raw.get("decomposition_error", 0.0)), 4),
        "weights_source": weights_source,
    }


def _format(
    entity_id: str,
    mode: str,
    output_format: str,
    spn: Optional[Dict],
    aggregator: Optional[Dict],
    uncertainty: Dict,   # {"spn": {...}|None, "aggregator": {...}|None}
    n_evidence: int,
    elapsed: float,
) -> Dict[str, Any]:
    spn_u = uncertainty.get("spn") or {}
    agg_u = uncertainty.get("aggregator") or {}

    if output_format == "flat":
        out: Dict[str, Any] = {"entity_id": entity_id, "mode": mode}
        if spn:
            out.update({"spn_prediction": spn["prediction"], "spn_confidence": spn["confidence"],
                        "spn_distribution": spn["distribution"]})
        if aggregator:
            out.update({"aggregator_prediction": aggregator["prediction"],
                        "aggregator_confidence": aggregator["confidence"],
                        "aggregator_distribution": aggregator["distribution"]})
        if spn_u:
            out.update({"spn_epistemic": spn_u.get("epistemic"), "spn_aleatoric": spn_u.get("aleatoric"),
                        "spn_total_uncertainty": spn_u.get("total"),
                        "spn_decomposition_error": spn_u.get("decomposition_error")})
        if agg_u:
            out.update({"aggregator_epistemic": agg_u.get("epistemic"),
                        "aggregator_aleatoric": agg_u.get("aleatoric"),
                        "aggregator_total_uncertainty": agg_u.get("total"),
                        "aggregator_decomposition_error": agg_u.get("decomposition_error")})
        out.update({"n_evidence_used": n_evidence, "execution_time_ms": round(elapsed, 1)})
        return out
    else:  # nested
        results: Dict[str, Any] = {}
        if spn:
            results["spn"] = spn
        if aggregator:
            results["aggregator"] = aggregator
        unc_out: Dict[str, Any] = {}
        if spn_u:
            unc_out["spn"] = spn_u
        if agg_u:
            unc_out["aggregator"] = agg_u
        return {
            "entity_id": entity_id,
            "mode": mode,
            "results": results,
            "uncertainty": unc_out,
            "n_evidence_used": n_evidence,
            "execution_time_ms": round(elapsed, 1),
        }
