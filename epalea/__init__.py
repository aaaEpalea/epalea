"""
epalea — Latent Factor Posteriors
Open source platform for evidential probabilistic inference.

Apache License 2.0 — see LICENSE
Pretrained weights: see pretrained/WEIGHT_LICENSE.md
"""

__version__ = "1.0.0"
__author__ = "Epalea Team"
__license__ = "Apache-2.0"

from pathlib import Path

_PACKAGE_ROOT = Path(__file__).parent
_REPO_ROOT = _PACKAGE_ROOT.parent
_PRETRAINED_ROOT = _REPO_ROOT / "pretrained"
_HF_ORG = "aaaEpalea"
_MODEL_REGISTRY = {
    "compliance-v1": f"{_HF_ORG}/epalea-compliance-v1",
}

def list_models():
    """List all available pretrained models."""
    import json
    models = []
    if not _PRETRAINED_ROOT.exists():
        return models
    for model_dir in sorted(_PRETRAINED_ROOT.iterdir()):
        if not model_dir.is_dir():
            continue
        schema_path = model_dir / "schema.json"
        if schema_path.exists():
            with open(schema_path) as f:
                schema = json.load(f)
            models.append(schema)
    return models


def load_model(model_id: str, cache_dir: str = ""):
    """
    Load a pretrained model by ID.

    Downloads from Hugging Face automatically on first use,
    then uses the local cache. No manual download needed.

    Args:
        model_id: Model identifier, e.g. 'compliance-v1'.
                  Can also be a local directory path.
        cache_dir: Optional custom cache directory.
                   Defaults to ~/.cache/huggingface/hub/

    Returns:
        LoadedModel instance ready for inference.

    Example:
        model = epalea.load_model("compliance-v1")
        result = model.infer(entity_id="C0001", mode="both")
    """
    from epalea._model import LoadedModel

    # 1. Treat as a local path if it looks like one
    local_path = Path(model_id)
    if local_path.exists() and (local_path / "schema.json").exists():
        return LoadedModel(local_path)

    # 2. Check pretrained/ folder (manually placed weights)
    bundled = _PRETRAINED_ROOT / model_id
    if bundled.exists() and (bundled / "schema.json").exists():
        return LoadedModel(bundled)

    # 3. Download from Hugging Face
    if model_id not in _MODEL_REGISTRY:
        available = list(_MODEL_REGISTRY.keys())
        raise ValueError(
            f"Model '{model_id}' not found locally and not in registry.\n"
            f"Available pretrained models: {available}\n"
            f"You can also pass a local directory path directly."
        )

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required to download pretrained models.\n"
            "Install it with: pip install huggingface_hub"
        )

    repo_id = _MODEL_REGISTRY[model_id]
    print(f"epalea: Downloading '{model_id}' from Hugging Face ({repo_id})...")
    print(f"  This only happens once — weights are cached locally after download.")

    local_dir = snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        ignore_patterns=["*.md", "*.txt"],
    )

    return LoadedModel(Path(local_dir))


__all__ = ["load_model", "list_models", "__version__"]
