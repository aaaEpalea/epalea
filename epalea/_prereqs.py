"""
Prerequisite checking for CLI commands.
All checks are hard exits with exact fix commands printed.
"""
from __future__ import annotations

from pathlib import Path
import typer
from rich.console import Console
from rich.text import Text

console = Console()


def _fail(msg: str) -> None:
    console.print(f"\n[bold red]✗[/bold red]  {msg}\n")
    raise typer.Exit(code=1)


def require_checkpoint(path: str, fix_cmd: str = "") -> None:
    """Fail if best_model.pt not found."""
    if not Path(path).exists():
        msg = f"Model checkpoint not found:\n   {path}"
        if fix_cmd:
            msg += f"\n\n   Run first:\n     {fix_cmd}"
        _fail(msg)


def require_schema(path: str) -> None:
    if not Path(path).exists():
        _fail(
            f"Schema file not found:\n   {path}\n\n"
            "   The schema is saved alongside best_model.pt during training."
        )


def require_index(index_dir: str, checkpoint: str = "", predicate: str = "") -> None:
    """Fail with actionable message if evidence index missing."""
    p = Path(index_dir)
    faiss_ok = (p / "vector_store.faiss").exists()
    meta_ok = (p / "metadata.jsonl").exists()
    if not faiss_ok or not meta_ok:
        fix = ""
        if checkpoint:
            fix = (
                f"\n\n   Run first:\n"
                f"     epalea index \\\n"
                f"       --checkpoint {checkpoint} \\\n"
                f"       --evidence   <path/to/evidence.json> \\\n"
                f"       --predicate  {predicate or '<predicate>'} \\\n"
                f"       --index-dir  {index_dir}"
            )
        _fail(f"Evidence index not found at:\n   {index_dir}{fix}")


def require_aggregator(path: str, checkpoint: str = "", index_dir: str = "") -> None:
    """Fail with actionable message if aggregator checkpoint missing."""
    if not Path(path).exists():
        fix = ""
        if checkpoint:
            fix = (
                f"\n\n   This is required for --mode aggregator and --mode both.\n"
                f"   Run first:\n"
                f"     epalea train-aggregator \\\n"
                f"       --checkpoint {checkpoint} \\\n"
                f"       --index-dir  {index_dir or '<index-dir>'} \\\n"
                f"       --data-dir   <path/to/data>"
            )
        _fail(f"Aggregator checkpoint not found:\n   {path}{fix}")


def require_data_dir(data_dir: str) -> None:
    p = Path(data_dir)
    if not p.exists():
        _fail(
            f"Data directory not found:\n   {data_dir}\n\n"
            "   Run first:\n"
            "     epalea generate-data --domain compliance --output-dir " + data_dir
        )
    train_c = p / "train_companies.json"
    if not train_c.exists():
        _fail(
            f"Training data not found in {data_dir}.\n"
            f"   Expected: {train_c}\n\n"
            "   Run first:\n"
            "     epalea generate-data --domain compliance --output-dir " + data_dir
        )


def guard_output_path(path: str) -> str:
    """Prevent writing into pretrained/."""
    from pathlib import Path as _P
    import epalea as _pkg
    pretrained = (_P(_pkg.__file__).parent.parent / "pretrained").resolve()
    resolved = _P(path).resolve()
    if resolved == pretrained or pretrained in resolved.parents:
        _fail(
            "Output path cannot be inside pretrained/.\n"
            "   Use ./user_workspace/ instead.\n"
            f"   Blocked path: {path}"
        )
    return path
