"""
epalea CLI — Complete command-line interface.
Built with Typer. All commands follow strict prerequisite checking.
"""

import sys
from pathlib import Path
from typing import List, Optional

# ─────────────────────────────────────────────────────────────
# Epalea CLI bootstrap (FIRST thing in cli.py)
# ─────────────────────────────────────────────────────────────
import os

# Suppress HF Hub warnings & progress bars
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# Prevent tokenizer multiprocessing warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Optional: suppress transformers loader messages
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()
# ─────────────────────────────────────────────────────────────

import typer

app = typer.Typer(
    name="epalea",
    help="Epalea — Latent Factor Posteriors. Evidential probabilistic inference.",
    add_completion=False,
    pretty_exceptions_enable=False,
)

def _version_callback(value: bool):
    if value:
        from epalea import __version__
        typer.echo(f"epalea {__version__}")
        raise typer.Exit()

@app.callback()
def main_callback(
    version: bool = typer.Option(
        None, "--version", "-v",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
):
    pass

models_app = typer.Typer(help="Manage pretrained models.")
app.add_typer(models_app, name="models")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _repo_root() -> Path:
    return Path(__file__).parent.parent


def _guard_output_path(path: str) -> str:
    resolved = Path(path).resolve()
    pretrained = (_repo_root() / "pretrained").resolve()
    if resolved == pretrained or pretrained in resolved.parents:
        typer.echo(
            typer.style("✗  Output path cannot be inside pretrained/.", fg=typer.colors.RED)
            + "\n   Use ./user_workspace/ instead."
        )
        raise typer.Exit(code=1)
    return path


def _require(path: Path, message: str):
    if not path.exists():
        typer.echo(typer.style(f"✗  {message}", fg=typer.colors.RED))
        raise typer.Exit(code=1)


def _load_config(config_path: Optional[str]) -> dict:
    if config_path is None:
        return {}
    try:
        import yaml
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        import json as _json
        with open(config_path) as f:
            return _json.load(f)


def _merge(config: dict, **kwargs) -> dict:
    """CLI flags override config file values."""
    merged = dict(config)
    for k, v in kwargs.items():
        if v is not None:
            merged[k] = v
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# epalea info
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def info():
    """Show system info, available models, and workspace status."""
    import platform
    from epalea import __version__, list_models

    try:
        import torch
        torch_ver = torch.__version__
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        torch_ver = "not installed"
        device = "cpu"

    ws = _repo_root() / "user_workspace"
    ws_status = "✓" if ws.exists() else "✗ (run: mkdir user_workspace)"

    typer.echo(f"\nepalea {__version__} — Latent Factor Posteriors")
    typer.echo("─" * 45)
    typer.echo(f"Python         {platform.python_version()}")
    typer.echo(f"PyTorch        {torch_ver}")
    typer.echo(f"Device         {device}")
    typer.echo(f"user_workspace ./user_workspace  {ws_status}")

    typer.echo("\nPretrained models:")
    models = list_models()
    if not models:
        typer.echo("  (none found — run: epalea models download compliance-v1)")
    for m in models:
        model_dir = _repo_root() / "pretrained" / m["model_id"]
        has_agg = "✓" if (model_dir / "aggregator.pt").exists() else "✗"
        has_idx = "✓" if (model_dir / "evidence_index" / "vector_store.faiss").exists() else "✗"
        typer.echo(f"  {m['model_id']:<20} ({m.get('embedding_dim',384)}d → {len(m.get('domain_values',[]))} classes)")
        typer.echo(f"    best_model.pt  {'✓' if (model_dir/'best_model.pt').exists() else '✗'}")
        typer.echo(f"    aggregator.pt  {has_agg}")
        typer.echo(f"    evidence_index {has_idx}")
    typer.echo("")


# ─────────────────────────────────────────────────────────────────────────────
# epalea models
# ─────────────────────────────────────────────────────────────────────────────

@models_app.command("list")
def models_list():
    """List all available pretrained models."""
    from epalea import list_models

    models = list_models()
    if not models:
        typer.echo("No pretrained models found.")
        typer.echo("Run: epalea models download compliance-v1")
        return

    header = f"{'ID':<22} {'Domain':<14} {'Classes':<10} {'Aggregator':<12} {'Index':<8} {'Released'}"
    typer.echo(header)
    typer.echo("─" * len(header))
    for m in models:
        model_dir = _repo_root() / "pretrained" / m["model_id"]
        has_agg = "✓" if (model_dir / "aggregator.pt").exists() else "✗"
        has_idx = "✓" if (model_dir / "evidence_index" / "vector_store.faiss").exists() else "✗"
        typer.echo(
            f"{m['model_id']:<22} {m.get('domain',''):<14} {len(m.get('domain_values',[])):<10} "
            f"{has_agg:<12} {has_idx:<8} {m.get('released','')}"
        )


@models_app.command("download")
def models_download(
    model_id: str = typer.Argument(..., help="Model ID to download (e.g. compliance-v1)"),
    cache_dir: Optional[str] = typer.Option(None, help="Custom cache directory"),
):
    """Download a pretrained model from Hugging Face."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        typer.echo(typer.style(
            "✗  huggingface_hub not installed. Run: pip install huggingface_hub",
            fg=typer.colors.RED
        ))
        raise typer.Exit(1)

    from epalea import _MODEL_REGISTRY

    if model_id not in _MODEL_REGISTRY:
        typer.echo(typer.style(
            f"✗  Unknown model '{model_id}'. Available: {list(_MODEL_REGISTRY.keys())}",
            fg=typer.colors.RED
        ))
        raise typer.Exit(1)

    repo_id = _MODEL_REGISTRY[model_id]
    typer.echo(f"\nDownloading '{model_id}' from {repo_id}...")
    typer.echo("Weights are cached in ~/.cache/huggingface/hub/ after first download.\n")

    try:
        local_dir = snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            ignore_patterns=["*.md", "*.txt"],
        )
        typer.echo(typer.style(f"\n✓ Downloaded to: {local_dir}", fg=typer.colors.GREEN))
        typer.echo(f"\nLoad it with:")
        typer.echo(f"  python -c \"import epalea; m = epalea.load_model('{model_id}')\"")
    except Exception as e:
        typer.echo(typer.style(f"✗  Download failed: {e}", fg=typer.colors.RED))
        raise typer.Exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# epalea generate-data
# ─────────────────────────────────────────────────────────────────────────────

@app.command("generate-data")
def generate_data(
    domain: str = typer.Option(..., help="Domain name: 'compliance' or 'custom'"),
    config: Optional[str] = typer.Option(None, help="YAML config file"),
    predicate: Optional[str] = typer.Option(None, help="Predicate name (custom mode)"),
    domain_values: Optional[List[str]] = typer.Option(None, help="Class values (custom mode)"),
    n_entities: int = typer.Option(300, help="Number of entities"),
    years: Optional[List[int]] = typer.Option(None, help="Fiscal years"),
    evidence_per_entity: int = typer.Option(5, help="Evidence items per entity"),
    noise_level: float = typer.Option(0.1, help="Noise level [0,1]"),
    contradictory_rate: float = typer.Option(0.05, help="Contradictory evidence fraction"),
    output_dir: str = typer.Option("./user_workspace/data", help="Output directory"),
):
    """Generate synthetic training data for any domain."""
    cfg = _load_config(config)
    params = _merge(cfg,
        domain=domain, predicate=predicate, n_entities=n_entities,
        evidence_per_entity=evidence_per_entity, noise_level=noise_level,
        contradictory_rate=contradictory_rate, output_dir=output_dir
    )
    if years:
        params["years"] = years
    if domain_values:
        params["domain_values"] = domain_values

    _guard_output_path(params["output_dir"])

    years_list = params.get("years", [2022])
    if isinstance(years_list, int):
        years_list = [years_list]

    typer.echo(f"\nGenerating {params['domain']} data...")
    typer.echo(f"  Entities: {params['n_entities']} × {len(years_list)} years")
    typer.echo(f"  Evidence per entity: {params['evidence_per_entity']}")
    typer.echo(f"  Noise level: {params['noise_level']}")

    # Import the synthetic data generator
    sys.path.insert(0, str(_repo_root()))
    from epalea.data.synthetic_data import SyntheticDataGenerator

    gen = SyntheticDataGenerator(seed=42)
    companies, evidence = gen.generate_dataset(
        n_companies=params["n_entities"],
        years=years_list,
        evidence_per_company=params["evidence_per_entity"],
        noise_level=params["noise_level"],
        contradictory_rate=params["contradictory_rate"],
    )

    splits = gen.create_splits(companies, evidence)
    out = Path(params["output_dir"]) / params["domain"]
    out.mkdir(parents=True, exist_ok=True)

    for split_name, (split_companies, split_evidence) in splits.items():
        gen.save_dataset(split_companies, split_evidence, str(out), prefix=f"{split_name}_")

    typer.echo(f"\n✓ Generated {len(companies)} company-year records")
    typer.echo(f"✓ Generated {len(evidence)} evidence items")
    typer.echo(f"✓ Saved to {out}")
    typer.echo(f"\nNext step:")
    typer.echo(f"  epalea train --domain {params['domain']} --data-dir {out} ...")


# ─────────────────────────────────────────────────────────────────────────────
# epalea train
# ─────────────────────────────────────────────────────────────────────────────

@app.command("train")
def train(
    config: Optional[str] = typer.Option(None, help="YAML config file (recommended)"),
    domain: Optional[str] = typer.Option(None, help="Domain name"),
    predicate: Optional[str] = typer.Option(None, help="Predicate name"),
    domain_values: Optional[List[str]] = typer.Option(None, help="Class values"),
    data_dir: Optional[str] = typer.Option(None, help="Input data directory"),
    checkpoint_dir: Optional[str] = typer.Option(None, help="Output checkpoint directory"),
    n_seeds: int = typer.Option(7, help="Seeds for seed search"),
    single_seed: Optional[int] = typer.Option(None, help="Fixed seed (skips seed search)"),
    embedding_dim: int = typer.Option(384, help="Embedding dimension"),
    latent_dim: int = typer.Option(64, help="Latent dimension"),
    epochs: int = typer.Option(50, help="Max epochs"),
    learning_rate: float = typer.Option(1e-3, help="Adam learning rate"),
    kl_weight: float = typer.Option(0.01, help="KL regularisation weight β"),
    label_key: Optional[str] = typer.Option(None, help="Label field in entity JSON"),
):
    """Train LPF encoder + decoder. Produces best_model.pt."""
    cfg = _load_config(config)
    p = _merge(cfg,
        domain=domain, predicate=predicate, data_dir=data_dir,
        checkpoint_dir=checkpoint_dir, embedding_dim=embedding_dim,
        latent_dim=latent_dim, epochs=epochs, learning_rate=learning_rate,
        kl_weight=kl_weight,
    )
    if domain_values:
        p["domain_values"] = domain_values
    if single_seed is not None:
        p["single_seed"] = single_seed
    if label_key:
        p["label_key"] = label_key

    for required in ["domain", "predicate", "domain_values", "data_dir"]:
        if not p.get(required):
            typer.echo(typer.style(f"✗  --{required.replace('_','-')} is required", fg=typer.colors.RED))
            raise typer.Exit(1)

    data_path = Path(p["data_dir"])
    _require(data_path / "train_companies.json",
             f"Training data not found at {data_path}/train_companies.json\n"
             f"   Run first:\n     epalea generate-data --domain {p['domain']} --output-dir {data_path.parent}")

    chk_dir = p.get("checkpoint_dir") or f"./user_workspace/checkpoints/{p['domain']}"
    _guard_output_path(chk_dir)

    results_dir = f"./user_workspace/results/{p['domain']}"

    sys.path.insert(0, str(_repo_root()))
    from epalea.training.train_generic_lpf import (
        create_schema_from_args, train_with_seed_search, set_seed,
        LPFTrainer, LPFModel, prepare_training_data_from_files
    )
    from epalea.models.vae_encoder import VAEEncoderNetwork
    from epalea.models.decoder_network import DecoderNetwork
    from torch.utils.data import DataLoader

    variable_name = p.get("variable_name", p["domain"])
    pred_name = p["predicate"]
    dom_values = p["domain_values"]
    lkey = p.get("label_key", pred_name)

    schema = create_schema_from_args(variable_name, pred_name, dom_values)

    if p.get("single_seed") is not None:
        seed = p["single_seed"]
        typer.echo(f"\nTraining LPF: {p['domain']} — {pred_name} → {dom_values}")
        typer.echo(f"Fixed seed: {seed}")

        set_seed(seed)

        # Setup evidence index
        from epalea.core.evidence_index import EvidenceIndex
        from epalea.training.train_generic_lpf import load_and_index_data
        idx_path = data_path / "vector_store.faiss"
        meta_path = data_path / "metadata.jsonl"
        evidence_index = EvidenceIndex(
            embedding_dim=p["embedding_dim"],
            vector_store_path=str(idx_path),
            metadata_store_path=str(meta_path),
        )
        if len(evidence_index) == 0:
            typer.echo("Indexing evidence (one-time)...")
            load_and_index_data(str(data_path), evidence_index, pred_name, split="train")
            load_and_index_data(str(data_path), evidence_index, pred_name, split="val")
            evidence_index._rebuild_entity_index()
            evidence_index.save()

        train_ds = prepare_training_data_from_files(str(data_path), evidence_index, schema, pred_name, lkey, "train")
        val_ds = prepare_training_data_from_files(str(data_path), evidence_index, schema, pred_name, lkey, "val")

        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=64)

        encoder = VAEEncoderNetwork(p["embedding_dim"], p["latent_dim"], [256, 128])
        decoder = DecoderNetwork(p["latent_dim"], schema, [128, 64])
        model = LPFModel(encoder, decoder)
        trainer = LPFTrainer(model, p["learning_rate"], p["kl_weight"])
        trainer.train(train_loader, val_loader, p["epochs"], 15, chk_dir)
    else:
        train_with_seed_search(
            domain=p["domain"],
            data_dir=str(data_path),
            checkpoint_dir=chk_dir,
            results_dir=results_dir,
            schema=schema,
            predicate=pred_name,
            label_key=lkey,
            n_seeds=p.get("n_seeds", n_seeds),
            embedding_dim=p["embedding_dim"],
            latent_dim=p["latent_dim"],
        )

    # Save schema.json
    import json as _json
    schema_out = Path(chk_dir) / "schema.json"
    schema_data = {
        "model_id": f"{p['domain']}-v1",
        "version": "1.0.0",
        "domain": p["domain"],
        "predicate": pred_name,
        "domain_values": dom_values,
        "variable_name": variable_name,
        "embedding_dim": p["embedding_dim"],
        "latent_dim": p["latent_dim"],
        "has_aggregator": False,
        "has_evidence_index": False,
        "description": f"{p['domain']} classification",
        "released": "",
    }
    with open(schema_out, "w") as f:
        _json.dump(schema_data, f, indent=2)

    typer.echo(f"\n✓ Saved: {chk_dir}/best_model.pt")
    typer.echo(f"✓ Saved: {schema_out}")
    typer.echo(f"\nNext step:")
    typer.echo(f"  epalea index \\")
    typer.echo(f"    --checkpoint {chk_dir}/best_model.pt \\")
    typer.echo(f"    --evidence   {data_path}/train_evidence.json \\")
    typer.echo(f"    --predicate  {pred_name} \\")
    typer.echo(f"    --index-dir  ./user_workspace/index/{p['domain']}")


# ─────────────────────────────────────────────────────────────────────────────
# epalea index
# ─────────────────────────────────────────────────────────────────────────────

@app.command("index")
def index_evidence(
    config: Optional[str] = typer.Option(None, help="YAML config file"),
    checkpoint: Optional[str] = typer.Option(None, help="Path to best_model.pt"),
    schema_path: Optional[str] = typer.Option(None, "--schema", help="Path to schema.json"),
    evidence: Optional[List[str]] = typer.Option(None, help="Evidence JSON file(s)"),
    predicate: Optional[str] = typer.Option(None, help="Predicate name"),
    index_dir: Optional[str] = typer.Option(None, help="Output index directory"),
    embedding_dim: int = typer.Option(384, help="Embedding dimension"),
):
    """
    Embed evidence and build FAISS index.

    Prerequisite: best_model.pt (from epalea train).
    Required for all inference and for epalea train-aggregator.
    """
    cfg = _load_config(config)
    p = _merge(cfg, checkpoint=checkpoint, predicate=predicate,
               index_dir=index_dir, embedding_dim=embedding_dim)
    if evidence:
        p["evidence_files"] = evidence
    if schema_path:
        p["schema"] = schema_path

    for required in ["checkpoint", "predicate", "evidence_files", "index_dir"]:
        if not p.get(required):
            typer.echo(typer.style(f"✗  --{required.replace('_','-').replace('files','').rstrip('-')} is required", fg=typer.colors.RED))
            raise typer.Exit(1)

    chk = Path(p["checkpoint"])
    _require(chk, f"Checkpoint not found at {chk}\n   Run first:\n     epalea train ...")

    _guard_output_path(p["index_dir"])

    sys.path.insert(0, str(_repo_root()))
    from epalea.core.evidence_index import EvidenceIndex

    idx_dir = Path(p["index_dir"])
    idx_dir.mkdir(parents=True, exist_ok=True)

    evidence_index = EvidenceIndex(
        embedding_dim=p["embedding_dim"],
        vector_store_path=str(idx_dir / "vector_store.faiss"),
        metadata_store_path=str(idx_dir / "metadata.jsonl"),
    )

    total = 0
    files = p["evidence_files"]
    if isinstance(files, str):
        files = [files]

    typer.echo(f"\nIndexing evidence...")
    for i, ev_file in enumerate(files, 1):
        typer.echo(f"  File {i}/{len(files)}: {Path(ev_file).name}")
        with open(ev_file) as f:
            import json as _json
            items = _json.load(f)

        count = 0
        for item in items:
            text = item.get("text_content", "")
            if not text and item.get("structured_data"):
                text = str(item["structured_data"])
            try:
                evidence_index.add_evidence(
                    evidence_id=item["evidence_id"],
                    entity_id=item.get("company_id") or item.get("entity_id"),
                    predicate=p["predicate"],
                    text_content=text or None,
                    structured_data=item.get("structured_data"),
                    credibility=item.get("credibility", 1.0),
                    timestamp=item.get("timestamp"),
                    evidence_type=item.get("evidence_type", "text"),
                    source=Path(ev_file).stem,
                )
                count += 1
            except Exception as e:
                pass
        typer.echo(f"    → {count} items")
        total += count

    typer.echo(f"\nRebuilding entity-predicate lookup...")
    evidence_index._rebuild_entity_index()

    typer.echo(f"Saving index...")
    evidence_index.save()

    # Verify
    n_entities = len(evidence_index.entity_index)
    sample_keys = list(evidence_index.entity_index.keys())[:1]
    typer.echo(f"\n✓ Entities indexed: {n_entities}")
    typer.echo(f"✓ Evidence items:   {total}")
    if sample_keys:
        eid, pred = sample_keys[0]
        n_items = len(evidence_index.entity_index[sample_keys[0]])
        typer.echo(f"✓ Sample lookup: {eid} → {pred} → {n_items} items ✓")
    typer.echo(f"✓ Saved: {idx_dir}/vector_store.faiss")
    typer.echo(f"✓ Saved: {idx_dir}/metadata.jsonl")

    # Update schema
    if p.get("schema") and Path(p["schema"]).exists():
        with open(p["schema"]) as f:
            sch = _json.load(f)     # type: ignore
        sch["has_evidence_index"] = True
        with open(p["schema"], "w") as f:
            _json.dump(sch, f, indent=2) # type: ignore
        typer.echo(f"✓ Updated schema.json (has_evidence_index=true)")

    typer.echo(f"\nNext step: epalea infer  OR  epalea train-aggregator ...")


# ─────────────────────────────────────────────────────────────────────────────
# epalea train-aggregator
# ─────────────────────────────────────────────────────────────────────────────

@app.command("train-aggregator")
def train_aggregator(
    config: Optional[str] = typer.Option(None, help="YAML config file"),
    checkpoint: Optional[str] = typer.Option(None, help="Path to best_model.pt"),
    schema_path: Optional[str] = typer.Option(None, "--schema", help="Path to schema.json"),
    index_dir: Optional[str] = typer.Option(None, help="Evidence index directory (defaults to --data-dir if not set)"),
    data_dir: Optional[str] = typer.Option(None, help="Training data directory"),
    aggregator_checkpoint_dir: Optional[str] = typer.Option(None, help="Output directory"),
    latent_dim: int = typer.Option(64, help="Latent dimension (must match training)"),
    hidden_dim: int = typer.Option(128, help="Aggregator hidden dimension"),
    dropout: float = typer.Option(0.1, help="Dropout rate"),
    epochs: int = typer.Option(30, help="Training epochs"),
    learning_rate: float = typer.Option(1e-3, help="Adam learning rate"),
):
    """
    Train the EvidenceAggregator network.

    Prerequisites: best_model.pt + evidence_index/.
    Enables --mode aggregator and --mode both for inference.
    """
    cfg = _load_config(config)
    p = _merge(cfg, checkpoint=checkpoint, index_dir=index_dir, data_dir=data_dir,
               aggregator_checkpoint_dir=aggregator_checkpoint_dir,
               latent_dim=latent_dim, hidden_dim=hidden_dim,
               dropout=dropout, epochs=epochs, learning_rate=learning_rate)
    
    # default index_dir to data_dir if not provided
    if not p.get("index_dir") and p.get("data_dir"):
        p["index_dir"] = p["data_dir"]

    if schema_path:
        p["schema"] = schema_path

    for req in ["checkpoint", "index_dir", "data_dir"]:
        if not p.get(req):
            typer.echo(typer.style(f"✗  --{req.replace('_','-')} is required", fg=typer.colors.RED))
            raise typer.Exit(1)

    chk = Path(p["checkpoint"])
    idx = Path(p["index_dir"])
    _require(chk, f"Checkpoint not found at {chk}\n   Run first:\n     epalea train ...")
    _require(idx / "vector_store.faiss",
             f"Evidence index not found at {idx}\n   Run first:\n     epalea index ...")

    out_dir = p.get("aggregator_checkpoint_dir") or str(chk.parent)
    _guard_output_path(out_dir)

    sys.path.insert(0, str(_repo_root()))
    import torch
    import json as _json
    from epalea.models.schema import Schema
    from epalea.models.vae_encoder import VAEEncoderNetwork, VAEEncoder
    from epalea.models.decoder_network import DecoderNetwork
    from epalea.core.evidence_index import EvidenceIndex
    from epalea.core.orchestrator import Orchestrator
    from epalea.core.canonical_db import CanonicalDB
    from epalea.core.provenance_ledger import ProvenanceLedger
    from epalea.models.learned_aggregator import add_learned_aggregation_to_orchestrator, prepare_aggregator_training_data, AggregatorTrainer

    # Load schema
    schema_file = p.get("schema") or str(chk.parent / "schema.json")
    with open(schema_file) as f:
        schema_data = _json.load(f)

    schema = Schema()
    pred_name = schema_data["predicate"]
    dom_values = schema_data["domain_values"]
    variable_name = schema_data.get("variable_name", schema_data["domain"])
    schema.add_variable(variable_name, dom_values)
    schema.add_predicate(pred_name, [variable_name], dom_values)

    lat = p["latent_dim"]
    chk_data = torch.load(str(chk), map_location="cpu")

    encoder_network = VAEEncoderNetwork(schema_data.get("embedding_dim", 384), lat, [256, 128])
    encoder_network.load_state_dict(chk_data["encoder_state_dict"])
    decoder_network = DecoderNetwork(lat, schema, [128, 64])
    decoder_network.load_state_dict(chk_data["decoder_state_dict"])
    vae_encoder = VAEEncoder(encoder_network, device="cpu")

    evidence_index = EvidenceIndex(
        embedding_dim=schema_data.get("embedding_dim", 384),
        vector_store_path=str(idx / "vector_store.faiss"),
        metadata_store_path=str(idx / "metadata.jsonl"),
    )
    evidence_index._rebuild_entity_index()

    orchestrator = Orchestrator(
        schema=schema, canonical_db=CanonicalDB(),
        evidence_index=evidence_index, vae_encoder=vae_encoder,
        decoder_network=decoder_network, provenance_ledger=ProvenanceLedger(),
        device="cpu",
    )
    add_learned_aggregation_to_orchestrator(orchestrator, latent_dim=lat)

    # Load entity labels from data
    data_path = Path(p["data_dir"])
    with open(data_path / "train_companies.json") as f:
        train_companies = _json.load(f)
    with open(data_path / "train_evidence.json") as f:
        train_evidence = _json.load(f)

    entity_labels = {e["company_id"]: e[pred_name] for e in train_companies}

    typer.echo(f"\nTraining aggregator...")
    typer.echo(f"  Preparing training data: {len(entity_labels)} entities")

    training_data = prepare_aggregator_training_data(
        orchestrator,
        train_companies,
        train_evidence,
        pred_name,
        pred_name,
    )

    trainer: AggregatorTrainer = orchestrator.aggregator_trainer  # type: ignore

    # Training loop
    for epoch in range(1, p["epochs"] + 1):
        total_loss = 0.0
        for item in training_data:
            loss = trainer.train_step(
                item["posteriors"],
                pred_name,
                item["true_label"],
                dom_values,
            )
            total_loss += loss
        avg_loss = total_loss / len(training_data) if training_data else 0.0
        if epoch % 10 == 0 or epoch == p["epochs"]:
            typer.echo(f"  Epoch {epoch}/{p['epochs']}, Loss: {avg_loss:.3f}")

    agg_path = Path(out_dir) / "aggregator.pt"
    trainer.save_checkpoint(str(agg_path))
    typer.echo(f"\n✓ Saved: {agg_path}")
    typer.echo("--mode aggregator and --mode both are now available.")

    # Update schema
    if Path(schema_file).exists():
        with open(schema_file) as f:
            sch = _json.load(f)
        sch["has_aggregator"] = True
        with open(schema_file, "w") as f:
            _json.dump(sch, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# epalea train-full
# ─────────────────────────────────────────────────────────────────────────────

@app.command("train-full")
def train_full(
    config: str = typer.Option(..., help="Combined config YAML (required)"),
):
    """
    Run full training pipeline: LPF base model + aggregator.

    Stage 1: Train LPF encoder + decoder (also indexes train/val evidence).
    Stage 2: Train aggregator.

    Note: test evidence is NOT indexed here — run epalea index separately
    before running infer or evaluate.
    """
    cfg = _load_config(config)

    typer.echo("\n══ Stage 1/2: LPF Base Model Training ═══════════════")
    try:
        _invoke_train(cfg)
    except SystemExit:
        typer.echo(typer.style("\n✗ Stage 1 failed. Pipeline stopped.", fg=typer.colors.RED))
        raise typer.Exit(1)

    typer.echo("\n══ Stage 2/2: Aggregator Training ═══════════════════")
    try:
        _invoke_train_aggregator(cfg)
    except SystemExit:
        typer.echo(typer.style("\n✗ Stage 2 failed. Pipeline stopped.", fg=typer.colors.RED))
        raise typer.Exit(1)

    typer.echo("\n══ train-full complete ═══════════════════════════════")
    typer.echo("✓ Base model and aggregator trained.")
    typer.echo("\nNext step: index test evidence before running inference:")
    typer.echo("  epalea index \\")
    typer.echo(f"    --checkpoint {cfg.get('checkpoint_dir', './user_workspace/checkpoints/compliance')}/best_model.pt \\")
    typer.echo(f"    --schema {cfg.get('checkpoint_dir', './user_workspace/checkpoints/compliance')}/schema.json \\")
    typer.echo(f"    --evidence {cfg.get('data_dir', './user_workspace/data/compliance')}/test_evidence.json \\")
    typer.echo(f"    --predicate {cfg.get('predicate', 'compliance_level')} \\")
    typer.echo(f"    --index-dir {cfg.get('data_dir', './user_workspace/data/compliance')}")


def _invoke_train(cfg: dict):
    """Internal: run train stage from combined config."""
    chk_dir = cfg.get("checkpoint_dir", f"./user_workspace/checkpoints/{cfg.get('domain','custom')}")
    evidence_files = cfg.get("evidence_files", [])

    sys.path.insert(0, str(_repo_root()))
    from epalea.training.train_generic_lpf import create_schema_from_args, train_with_seed_search
    import json as _json

    variable_name = cfg.get("variable_name", cfg.get("domain", "custom"))
    schema = create_schema_from_args(variable_name, cfg["predicate"], cfg["domain_values"])

    train_with_seed_search(
        domain=cfg["domain"],
        data_dir=cfg["data_dir"],
        checkpoint_dir=chk_dir,
        results_dir=f"./user_workspace/results/{cfg['domain']}",
        schema=schema,
        predicate=cfg["predicate"],
        label_key=cfg.get("label_key", cfg["predicate"]),
        n_seeds=cfg.get("n_seeds", 7),
        embedding_dim=cfg.get("embedding_dim", 384),
        latent_dim=cfg.get("latent_dim", 64),
    )

    schema_out = Path(chk_dir) / "schema.json"
    schema_data = {
        "model_id": f"{cfg['domain']}-v1",
        "version": "1.0.0",
        "domain": cfg["domain"],
        "predicate": cfg["predicate"],
        "domain_values": cfg["domain_values"],
        "variable_name": variable_name,
        "embedding_dim": cfg.get("embedding_dim", 384),
        "latent_dim": cfg.get("latent_dim", 64),
        "has_aggregator": False,
        "has_evidence_index": False,
        "description": f"{cfg['domain']} classification",
        "released": "",
    }
    with open(schema_out, "w") as f:
        _json.dump(schema_data, f, indent=2)


def _invoke_index(cfg: dict):
    """Internal: run index stage from combined config."""
    chk_dir = cfg.get("checkpoint_dir", f"./user_workspace/checkpoints/{cfg.get('domain','custom')}")
    idx_dir = cfg.get("index_dir", f"./user_workspace/index/{cfg.get('domain','custom')}")
    files = cfg.get("evidence_files", [])

    sys.path.insert(0, str(_repo_root()))
    from epalea.core.evidence_index import EvidenceIndex
    import json as _json

    idx_path = Path(idx_dir)
    idx_path.mkdir(parents=True, exist_ok=True)

    evidence_index = EvidenceIndex(
        embedding_dim=cfg.get("embedding_dim", 384),
        vector_store_path=str(idx_path / "vector_store.faiss"),
        metadata_store_path=str(idx_path / "metadata.jsonl"),
    )

    for ev_file in files:
        with open(ev_file) as f:
            items = _json.load(f)
        for item in items:
            text = item.get("text_content", "") or str(item.get("structured_data", ""))
            try:
                evidence_index.add_evidence(
                    evidence_id=item["evidence_id"],
                    entity_id=item.get("company_id") or item.get("entity_id"),
                    predicate=cfg["predicate"],
                    text_content=text or None,
                    structured_data=item.get("structured_data"),
                    credibility=item.get("credibility", 1.0),
                    timestamp=item.get("timestamp"),
                    evidence_type=item.get("evidence_type", "text"),
                    source=Path(ev_file).stem,
                )
            except Exception:
                pass

    evidence_index._rebuild_entity_index()
    evidence_index.save()
    typer.echo(f"  ✓ Indexed {len(evidence_index)} items across {len(evidence_index.entity_index)} entities")


def _invoke_train_aggregator(cfg: dict):
    """Internal: run aggregator training from combined config."""
    chk_dir = cfg.get("checkpoint_dir", f"./user_workspace/checkpoints/{cfg.get('domain','custom')}")
    idx_dir = cfg.get("index_dir", f"./user_workspace/index/{cfg.get('domain','custom')}")

    sys.path.insert(0, str(_repo_root()))
    import torch, json as _json
    from epalea.models.schema import Schema
    from epalea.models.vae_encoder import VAEEncoderNetwork, VAEEncoder
    from epalea.models.decoder_network import DecoderNetwork
    from epalea.core.evidence_index import EvidenceIndex
    from epalea.core.orchestrator import Orchestrator
    from epalea.core.canonical_db import CanonicalDB
    from epalea.core.provenance_ledger import ProvenanceLedger
    from epalea.models.learned_aggregator import add_learned_aggregation_to_orchestrator, prepare_aggregator_training_data

    schema_file = Path(chk_dir) / "schema.json"
    with open(schema_file) as f:
        schema_data = _json.load(f)

    schema = Schema()
    pred_name = schema_data["predicate"]
    dom_values = schema_data["domain_values"]
    variable_name = schema_data.get("variable_name", schema_data["domain"])
    schema.add_variable(variable_name, dom_values)
    schema.add_predicate(pred_name, [variable_name], dom_values)

    lat = cfg.get("latent_dim", 64)
    chk_data = torch.load(str(Path(chk_dir) / "best_model.pt"), map_location="cpu")
    encoder_network = VAEEncoderNetwork(cfg.get("embedding_dim", 384), lat, [256, 128])
    encoder_network.load_state_dict(chk_data["encoder_state_dict"])
    decoder_network = DecoderNetwork(lat, schema, [128, 64])
    decoder_network.load_state_dict(chk_data["decoder_state_dict"])
    vae_encoder = VAEEncoder(encoder_network, device="cpu")

    evidence_index = EvidenceIndex(
        embedding_dim=cfg.get("embedding_dim", 384),
        vector_store_path=str(Path(idx_dir) / "vector_store.faiss"),
        metadata_store_path=str(Path(idx_dir) / "metadata.jsonl"),
    )
    evidence_index._rebuild_entity_index()

    orchestrator = Orchestrator(
        schema=schema, canonical_db=CanonicalDB(),
        evidence_index=evidence_index, vae_encoder=vae_encoder,
        decoder_network=decoder_network, provenance_ledger=ProvenanceLedger(), device="cpu",
    )
    add_learned_aggregation_to_orchestrator(orchestrator, latent_dim=lat)

    with open(Path(cfg["data_dir"]) / "train_companies.json") as f:
        train_companies = _json.load(f)
    with open(Path(cfg["data_dir"]) / "train_evidence.json") as f:
        train_evidence = _json.load(f)

    entity_labels = {e["company_id"]: e[pred_name] for e in train_companies}

    training_data = prepare_aggregator_training_data(
        orchestrator,
        train_companies,
        train_evidence,
        pred_name,
        pred_name,
    )
    trainer = orchestrator.aggregator_trainer  # type: ignore
    epochs = cfg.get("aggregator_epochs", 30)
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for item in training_data:
            loss = trainer.train_step( # type: ignore
                item["posteriors"],
                pred_name,
                item["true_label"],
                dom_values,
            )
            total_loss += loss
        avg_loss = total_loss / len(training_data) if training_data else 0.0
        if epoch % 10 == 0 or epoch == epochs:
            typer.echo(f"  Epoch {epoch}/{epochs}, Loss: {avg_loss:.3f}")

    agg_path = Path(chk_dir) / "aggregator.pt"
    trainer.save_checkpoint(str(agg_path)) # type: ignore
    typer.echo(f"  ✓ Saved: {agg_path}")

    with open(schema_file) as f:
        sch = _json.load(f)
    sch["has_aggregator"] = True
    sch["has_evidence_index"] = True
    with open(schema_file, "w") as f:
        _json.dump(sch, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# epalea infer
# ─────────────────────────────────────────────────────────────────────────────

@app.command("infer")
def infer(
    config: Optional[str] = typer.Option(None, help="YAML config file"),
    checkpoint: Optional[str] = typer.Option(None, help="Path to best_model.pt"),
    aggregator_checkpoint: Optional[str] = typer.Option(None, help="Path to aggregator.pt"),
    schema_path: Optional[str] = typer.Option(None, "--schema", help="Path to schema.json"),
    index_dir: Optional[str] = typer.Option(None, help="Evidence index directory"),
    entity_id: Optional[str] = typer.Option(None, help="Entity ID to query"),
    evidence_file: Optional[str] = typer.Option(None, help="Evidence JSON file"),
    mode: str = typer.Option("spn", help="spn | aggregator | both"),
    output_format: str = typer.Option("nested", help="nested | flat"),
    top_k: int = typer.Option(5, help="Evidence items to retrieve"),
    n_samples: int = typer.Option(4, help="Monte Carlo samples"),
    temperature: float = typer.Option(0.8, help="Temperature scaling"),
    alpha: float = typer.Option(0.1, help="Weight penalty α"),
):
    """Run inference for a single entity."""
    cfg = _load_config(config)
    p = _merge(cfg, checkpoint=checkpoint, aggregator_checkpoint=aggregator_checkpoint,
               index_dir=index_dir, entity_id=entity_id, evidence_file=evidence_file,
               mode=mode, output_format=output_format, top_k=top_k,
               n_samples=n_samples, temperature=temperature, alpha=alpha)
    if schema_path:
        p["schema"] = schema_path

    for req in ["checkpoint", "index_dir", "entity_id"]:
        if not p.get(req):
            typer.echo(typer.style(f"✗  --{req.replace('_','-')} is required", fg=typer.colors.RED))
            raise typer.Exit(1)

    chk = Path(p["checkpoint"])
    idx = Path(p["index_dir"])
    _require(chk, f"Checkpoint not found at {chk}\n   Run first:\n     epalea train ...")
    _require(idx / "vector_store.faiss",
             f"Evidence index not found at {idx}/vector_store.faiss\n   Run first:\n     epalea index ...")

    if p["mode"] in ("aggregator", "both"):
        agg_chk = p.get("aggregator_checkpoint")
        if not agg_chk or not Path(agg_chk).exists():
            typer.echo(typer.style(
                f"✗  Aggregator checkpoint not found.\n"
                f"   This is required for --mode {p['mode']}.\n"
                f"   Run first:\n     epalea train-aggregator ...",
                fg=typer.colors.RED
            ))
            raise typer.Exit(1)

    result = _run_single_infer(p)

    import json as _json
    typer.echo(_json.dumps(result, indent=2))


def _run_single_infer(p: dict) -> dict:
    """Internal: run single inference, return dict."""
    import time, json as _json
    sys.path.insert(0, str(_repo_root()))
    import torch
    from epalea.models.schema import Schema
    from epalea.models.vae_encoder import VAEEncoderNetwork, VAEEncoder
    from epalea.models.decoder_network import DecoderNetwork
    from epalea.core.evidence_index import EvidenceIndex
    from epalea.core.orchestrator import Orchestrator, QueryOptions
    from epalea.core.canonical_db import CanonicalDB
    from epalea.core.provenance_ledger import ProvenanceLedger

    schema_file = p.get("schema") or str(Path(p["checkpoint"]).parent / "schema.json")
    with open(schema_file) as f:
        schema_data = _json.load(f)

    schema = Schema()
    pred_name = schema_data["predicate"]
    dom_values = schema_data["domain_values"]
    variable_name = schema_data.get("variable_name", schema_data.get("domain", pred_name))
    schema.add_variable(variable_name, dom_values)
    schema.add_predicate(pred_name, [variable_name], dom_values)

    lat = schema_data.get("latent_dim", 64)
    emb_dim = schema_data.get("embedding_dim", 384)
    chk_data = torch.load(p["checkpoint"], map_location="cpu")

    encoder_network = VAEEncoderNetwork(emb_dim, lat, [256, 128])
    encoder_network.load_state_dict(chk_data["encoder_state_dict"])
    decoder_network = DecoderNetwork(lat, schema, [128, 64])
    decoder_network.load_state_dict(chk_data["decoder_state_dict"])
    vae_encoder = VAEEncoder(encoder_network, device="cpu")

    evidence_index = EvidenceIndex(
        embedding_dim=emb_dim,
        vector_store_path=str(Path(p["index_dir"]) / "vector_store.faiss"),
        metadata_store_path=str(Path(p["index_dir"]) / "metadata.jsonl"),
    )
    evidence_index._rebuild_entity_index()

    orchestrator = Orchestrator(
        schema=schema, canonical_db=CanonicalDB(),
        evidence_index=evidence_index, vae_encoder=vae_encoder,
        decoder_network=decoder_network, provenance_ledger=ProvenanceLedger(), device="cpu",
    )

    if p.get("aggregator_checkpoint") and Path(p["aggregator_checkpoint"]).exists():
        from epalea.models.learned_aggregator import add_learned_aggregation_to_orchestrator
        add_learned_aggregation_to_orchestrator(orchestrator, latent_dim=lat)
        trainer = getattr(orchestrator, "aggregator_trainer", None)
        if trainer:
            trainer.load_checkpoint(p["aggregator_checkpoint"])
        has_aggregator = True
    else:
        has_aggregator = False

    t0 = time.time()
    results = {}

    def _run_mode(use_spn: bool):
        opts = QueryOptions(
            top_k=p.get("top_k", 5), n_samples=p.get("n_samples", 4),
            temperature=p.get("temperature", 0.8), alpha=p.get("alpha", 0.1),
            use_canonical=True, use_spn=use_spn,
        )
        r = orchestrator.query(p["entity_id"], pred_name, opts, return_uncertainty=True)
        return {
            "prediction": r.get("top_value", "unknown"),
            "confidence": float(r.get("confidence", 0.0)),
            "distribution": r.get("distribution", {}),
        }

    mode = p.get("mode", "spn")
    if mode in ("spn", "both"):
        results["spn"] = _run_mode(True)
    if mode in ("aggregator", "both"):
        results["aggregator"] = _run_mode(False)

    # Per-mode uncertainty — both signals returned independently, neither suppressed.
    def _extract_unc(use_spn: bool, weights_source: str) -> dict:
        r = orchestrator.query(p["entity_id"], pred_name,
            QueryOptions(top_k=p.get("top_k", 5), n_samples=p.get("n_samples", 4),
                         temperature=p.get("temperature", 0.8), alpha=p.get("alpha", 0.1),
                         use_canonical=True, use_spn=use_spn),
            return_uncertainty=True)
        ep = float(r.get("epistemic_uncertainty", 0.0))
        al = float(r.get("aleatoric_uncertainty", 0.0))
        return {"epistemic": round(ep, 4), "aleatoric": round(al, 4), "total": round(ep + al, 4),
                "decomposition_error": round(float(r.get("decomposition_error", 0.0)), 4),
                "weights_source": weights_source}

    uncertainty: dict = {}
    if mode in ("spn", "both"):
        uncertainty["spn"] = _extract_unc(True, "uniform")
    if mode in ("aggregator", "both"):
        uncertainty["aggregator"] = _extract_unc(False, "aggregator")

    evidence_ids = orchestrator.evidence_index.search(p["entity_id"], pred_name, top_k=p.get("top_k",5))
    exec_ms = (time.time() - t0) * 1000

    data = {
        "entity_id": p["entity_id"],
        "mode": mode,
        "results": results,
        "uncertainty": uncertainty,
        "n_evidence_used": len(evidence_ids),
        "execution_time_ms": round(exec_ms, 1),
    }

    if p.get("output_format") == "flat":
        from epalea._model import _flatten_result
        data = _flatten_result(data)

    return data


# ─────────────────────────────────────────────────────────────────────────────
# epalea batch-infer
# ─────────────────────────────────────────────────────────────────────────────

@app.command("batch-infer")
def batch_infer(
    config: Optional[str] = typer.Option(None, help="YAML config file"),
    checkpoint: Optional[str] = typer.Option(None, help="Path to best_model.pt"),
    aggregator_checkpoint: Optional[str] = typer.Option(None, help="Path to aggregator.pt"),
    schema_path: Optional[str] = typer.Option(None, "--schema", help="Path to schema.json"),
    index_dir: Optional[str] = typer.Option(None, help="Evidence index directory"),
    companies: Optional[str] = typer.Option(None, help="Companies JSON file"),
    evidence_file: Optional[str] = typer.Option(None, help="Evidence JSON file"),
    output_dir: str = typer.Option("./user_workspace/results", help="Output directory"),
    mode: str = typer.Option("spn", help="spn | aggregator | both"),
    output_format: str = typer.Option("nested", help="nested | flat"),
    top_k: int = typer.Option(5),
    n_samples: int = typer.Option(4),
    temperature: float = typer.Option(0.8),
    alpha: float = typer.Option(0.1),
):
    """Run batch inference for all entities in a companies JSON file."""
    cfg = _load_config(config)
    p = _merge(cfg, checkpoint=checkpoint, aggregator_checkpoint=aggregator_checkpoint,
               index_dir=index_dir, companies=companies, evidence_file=evidence_file,
               output_dir=output_dir, mode=mode, output_format=output_format,
               top_k=top_k, n_samples=n_samples, temperature=temperature, alpha=alpha)
    if schema_path:
        p["schema"] = schema_path

    _guard_output_path(p["output_dir"])

    import json as _json

    companies_file = p.get("companies")
    if not companies_file:
        typer.echo(typer.style("✗  --companies is required", fg=typer.colors.RED))
        raise typer.Exit(1)

    with open(companies_file) as f:
        all_companies = _json.load(f)

    typer.echo(f"\nBatch inference: {len(all_companies)} entities, mode={p['mode']}")

    results = []
    failed = 0
    for i, company in enumerate(all_companies, 1):
        entity_id = company.get("company_id") or company.get("entity_id")
        try:
            entity_p = dict(p)
            entity_p["entity_id"] = entity_id
            result = _run_single_infer(entity_p)
            results.append(result)
        except Exception as e:
            failed += 1

        if i % 10 == 0:
            typer.echo(f"  {i}/{len(all_companies)} processed...")

    out = Path(p["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    out_file = out / "batch_predictions.json"
    with open(out_file, "w") as f:
        _json.dump(results, f, indent=2)

    typer.echo(f"\n✓ {len(results)} predictions written to {out_file}")
    if failed:
        typer.echo(f"  ⚠  {failed} entities failed")


# ─────────────────────────────────────────────────────────────────────────────
# epalea evaluate
# ─────────────────────────────────────────────────────────────────────────────

@app.command("evaluate")
def evaluate(
    config: Optional[str] = typer.Option(None, help="YAML config file"),
    checkpoint: Optional[str] = typer.Option(None, help="Path to best_model.pt"),
    aggregator_checkpoint: Optional[str] = typer.Option(None, help="Path to aggregator.pt"),
    schema_path: Optional[str] = typer.Option(None, "--schema", help="Path to schema.json"),
    index_dir: Optional[str] = typer.Option(None, help="Evidence index directory"),
    test_companies: Optional[str] = typer.Option(None, help="Test companies JSON"),
    test_evidence: Optional[str] = typer.Option(None, help="Test evidence JSON"),
    output_dir: str = typer.Option("./user_workspace/results", help="Output directory"),
    mode: str = typer.Option("spn", help="spn | aggregator | both"),
    output_format: str = typer.Option("nested", help="nested | flat"),
    top_k: int = typer.Option(5),
    n_samples: int = typer.Option(4),
):
    """Run batch inference + full metrics evaluation."""
    cfg = _load_config(config)
    p = _merge(cfg, checkpoint=checkpoint, aggregator_checkpoint=aggregator_checkpoint,
               index_dir=index_dir, test_companies=test_companies, test_evidence=test_evidence,
               output_dir=output_dir, mode=mode, output_format=output_format,
               top_k=top_k, n_samples=n_samples)
    if schema_path:
        p["schema"] = schema_path

    _guard_output_path(p["output_dir"])

    import json as _json
    sys.path.insert(0, str(_repo_root()))

    companies_file = p.get("test_companies")
    if not companies_file:
        typer.echo(typer.style("✗  --test-companies is required", fg=typer.colors.RED))
        raise typer.Exit(1)

    with open(companies_file) as f:
        all_companies = _json.load(f)

    schema_file = p.get("schema") or str(Path(p["checkpoint"]).parent / "schema.json")
    with open(schema_file) as f:
        schema_data = _json.load(f)

    pred_name = schema_data["predicate"]
    dom_values = schema_data["domain_values"]
    entity_labels = {c["company_id"]: c[pred_name] for c in all_companies}

    typer.echo(f"\nEvaluating: {len(all_companies)} entities, mode={p['mode']}")

    from epalea.evaluation.metrics import MetricsCalculator

    calc = MetricsCalculator(dom_values)

    predictions_by_mode = {"spn": [], "aggregator": []}
    true_labels = []
    per_entity_rows = []  # stores per-entity detail including uncertainty

    for i, company in enumerate(all_companies, 1):
        entity_id = company.get("company_id")
        true_label = entity_labels.get(entity_id, dom_values[0])
        true_labels.append(true_label)
        entity_p = dict(p)
        entity_p["entity_id"] = entity_id
        try:
            result = _run_single_infer(entity_p)
            res = result.get("results", {})
            unc = result.get("uncertainty", {})  # now {"spn": {...}, "aggregator": {...}}
            row = {"entity_id": entity_id, "true_label": true_label, "uncertainty": unc}
            if "spn" in res:
                predictions_by_mode["spn"].append(res["spn"]["distribution"])
                row["spn_prediction"] = res["spn"]["prediction"]
                row["spn_confidence"] = res["spn"]["confidence"]
            if "aggregator" in res:
                predictions_by_mode["aggregator"].append(res["aggregator"]["distribution"])
                row["aggregator_prediction"] = res["aggregator"]["prediction"]
                row["aggregator_confidence"] = res["aggregator"]["confidence"]
            per_entity_rows.append(row)
        except Exception:
            for k in predictions_by_mode:
                predictions_by_mode[k].append({v: 1.0/len(dom_values) for v in dom_values})
            per_entity_rows.append({"entity_id": entity_id, "true_label": true_label,
                                    "error": "inference_failed", "uncertainty": {}})
        
        if i % 10 == 0:
            typer.echo(f"  {i}/{len(all_companies)} processed...")

    out = Path(p["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    metrics_out = {}

    header_mode = p["mode"]
    typer.echo(f"\n── mode: {header_mode} {'─'*(50-len(header_mode))}")

    modes_to_eval = []
    if header_mode == "both":
        modes_to_eval = ["spn", "aggregator"]
    elif header_mode == "spn":
        modes_to_eval = ["spn"]
    else:
        modes_to_eval = ["aggregator"]

    header = f"{'System':<16} {'Acc':>8} {'Macro F1':>10} {'NLL':>8} {'Brier':>8} {'ECE':>8} {'Ep.Unc':>8} {'Al.Unc':>8}"
    typer.echo(header)
    typer.echo("─" * len(header))

    for m in modes_to_eval:
        preds = predictions_by_mode.get(m, [])
        if not preds:
            continue
        report = calc.compute_all_metrics(preds, true_labels)
        label = "LPF-SPN" if m == "spn" else "LPF-Agg"
        # Compute mean uncertainty for this mode across all entities
        ep_vals = [r["uncertainty"].get(m, {}).get("epistemic", 0.0)
                   for r in per_entity_rows if r.get("uncertainty", {}).get(m)]
        al_vals = [r["uncertainty"].get(m, {}).get("aleatoric", 0.0)
                   for r in per_entity_rows if r.get("uncertainty", {}).get(m)]
        mean_ep = sum(ep_vals) / len(ep_vals) if ep_vals else 0.0
        mean_al = sum(al_vals) / len(al_vals) if al_vals else 0.0
        typer.echo(
            f"{label:<16} {report.accuracy:>8.3f} {report.macro_f1:>10.3f} "
            f"{report.nll:>8.3f} {report.brier:>8.3f} {report.ece:>8.3f} "
            f"{mean_ep:>8.3f} {mean_al:>8.3f}"
        )
        metrics_out[m] = report.to_dict()

    out_file = out / "evaluation.json"
    with open(out_file, "w") as f:
        _json.dump({
            "mode": p["mode"],
            "metrics": metrics_out,
            "n_evaluated": len(all_companies),
            "per_entity": per_entity_rows,  # full per-entity detail with uncertainty
        }, f, indent=2)

    typer.echo(f"\nSaved: {out_file}")


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def main():
    app()


if __name__ == "__main__":
    main()
