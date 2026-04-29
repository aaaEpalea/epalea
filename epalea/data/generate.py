"""
Wrapper around data.synthetic_data.SyntheticDataGenerator.
Provides the generate_data() function used by notebooks and the CLI.
"""

import sys
from pathlib import Path
from typing import List, Optional


def generate_data(
    domain: str = "compliance",
    config: Optional[str] = None,
    predicate: Optional[str] = None,
    domain_values: Optional[List[str]] = None,
    n_entities: int = 300,
    years: Optional[List[int]] = None,
    evidence_per_entity: int = 5,
    noise_level: float = 0.1,
    contradictory_rate: float = 0.05,
    output_dir: str = "./user_workspace/data",
    seed: int = 42,
):
    """
    Generate synthetic training data for any domain.

    Args:
        domain: Domain name ('compliance' or 'custom').
        config: Path to custom domain YAML config.
        predicate: Predicate name (custom mode).
        domain_values: Class values (custom mode).
        n_entities: Number of entities.
        years: Fiscal years.
        evidence_per_entity: Evidence items per entity.
        noise_level: Noise level [0, 1].
        contradictory_rate: Fraction of contradictory evidence.
        output_dir: Output directory.
        seed: Random seed.
    """
    # Resolve package root to find data.synthetic_data
    pkg_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(pkg_root))

    if years is None:
        years = [2022]

    from data.synthetic_data import SyntheticDataGenerator

    gen = SyntheticDataGenerator(seed=seed)
    companies, evidence = gen.generate_dataset(
        n_companies=n_entities,
        years=years,
        evidence_per_company=evidence_per_entity,
        noise_level=noise_level,
        contradictory_rate=contradictory_rate,
    )

    splits = gen.create_splits(companies, evidence)
    out = Path(output_dir) / domain
    out.mkdir(parents=True, exist_ok=True)

    for split_name, (split_companies, split_evidence) in splits.items():
        gen.save_dataset(split_companies, split_evidence, str(out), prefix=f"{split_name}_")

    print(f"✓ Generated {len(companies)} entities, {len(evidence)} evidence items → {out}")
    return out
