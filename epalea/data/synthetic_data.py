"""
Synthetic Data Generator
Generates controlled synthetic datasets for tax compliance experiments.
Implements Section 10.2.2 from the paper.
"""

import json
import random
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timedelta


# Fixed seeds for reproducibility
RANDOM_SEED = 42


@dataclass
class Company:
    """Synthetic company record."""
    company_id: str
    company_name: str
    year: int
    industry: str
    country: str
    revenue: float
    profit: float
    tax_paid: float
    num_employees: int
    subsidiaries: int
    on_time_filing: bool
    accurate_reporting: bool
    past_violations: int
    audit_score: float
    compliance_level: str  # "low", "medium", "high"
    compliance_score: float  # continuous [0, 1]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Evidence:
    """Synthetic evidence record."""
    evidence_id: str
    company_id: str
    evidence_type: str  # "structured", "filing", "audit_report", "news"
    text_content: str
    structured_data: Dict[str, Any]
    credibility: float
    timestamp: str
    supports_compliance: str  # "low", "medium", "high"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SyntheticDataGenerator:
    """
    Generates synthetic tax compliance datasets with controllable parameters.
    """
    
    # Class-level constants from Section 10.2.2
    INDUSTRIES = ["Technology", "Finance", "Manufacturing", "Retail", "Healthcare"]
    COUNTRIES = ["US", "UK", "Germany", "Japan", "Singapore"]
    COMPLIANCE_LEVELS = ["low", "medium", "high"]
    
    # Compliance level distributions
    COMPLIANCE_WEIGHTS = [0.2, 0.5, 0.3]  # low, medium, high
    
    # Feature-label correlations from Table in Section 10.2.2
    COMPLIANCE_PARAMS = {
        "high": {
            "tax_rate_range": (0.18, 0.25),
            "on_time_rate": 0.95,
            "violations_lambda": 0.3,
            "audit_score_range": (80, 100),
            "score_range": (0.75, 1.0)
        },
        "medium": {
            "tax_rate_range": (0.12, 0.22),
            "on_time_rate": 0.75,
            "violations_lambda": 1.0,
            "audit_score_range": (50, 80),
            "score_range": (0.40, 0.75)
        },
        "low": {
            "tax_rate_range": (0.02, 0.15),
            "on_time_rate": 0.50,
            "violations_lambda": 2.5,
            "audit_score_range": (10, 50),
            "score_range": (0.0, 0.40)
        }
    }
    
    # Text templates for evidence generation
    TEXT_TEMPLATES = {
        "high": [
            "{company} demonstrates strong tax compliance with timely filings.",
            "Audit report shows {company} maintains excellent record-keeping.",
            "{company} has consistently met all regulatory requirements.",
            "Financial analysis indicates {company} follows best practices in tax reporting."
        ],
        "medium": [
            "Audit found some discrepancies in {company}'s filings, now resolved.",
            "{company} shows adequate compliance with minor reporting delays.",
            "Tax review of {company} revealed procedural improvements needed.",
            "{company} maintains acceptable compliance standards with occasional issues."
        ],
        "low": [
            "Investigation reveals significant compliance issues at {company}.",
            "{company} faces penalties for late and inaccurate tax filings.",
            "Audit uncovered serious deficiencies in {company}'s tax practices.",
            "Regulatory action taken against {company} for compliance violations."
        ]
    }
    
    def __init__(self, seed: int = RANDOM_SEED):
        """Initialize generator with seed."""
        self.seed = seed
        # Create instance-specific RNG objects instead of using global state
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.company_counter = 0
        self.evidence_counter = 0
    
    def _generate_company_name(self) -> str:
        """Generate realistic company name."""
        prefixes = ["Global", "Tech", "United", "Advanced", "Premier", "Innovative"]
        middles = ["Solutions", "Corp", "Industries", "Systems", "Group", "Holdings"]
        suffixes = ["Inc", "LLC", "Ltd", "Co"]
        
        return f"{self.rng.choice(prefixes)} {self.rng.choice(middles)} {self.rng.choice(suffixes)}"
    
    def _sample_compliance_level(self) -> str:
        """Sample compliance level according to distribution."""
        return self.np_rng.choice(self.COMPLIANCE_LEVELS, p=self.COMPLIANCE_WEIGHTS)
    
    def generate_company(self, year: int) -> Company:
        """
        Generate a single synthetic company record.
        
        Args:
            year: Fiscal year
            
        Returns:
            Company object
        """
        self.company_counter += 1
        company_id = f"C{self.company_counter:04d}"
        company_name = self._generate_company_name()
        industry = self.rng.choice(self.INDUSTRIES)
        country = self.rng.choice(self.COUNTRIES)
        
        # Sample compliance level
        compliance_level = self._sample_compliance_level()
        params = self.COMPLIANCE_PARAMS[compliance_level]
        
        # Generate financial features (log-normal distributions)
        revenue = self.np_rng.lognormal(mean=15, sigma=1.5) * 1000  # $3M-$300M range
        profit_margin = self.np_rng.uniform(0.05, 0.30)
        profit = revenue * profit_margin
        
        # Tax rate correlated with compliance
        tax_rate = self.np_rng.uniform(*params["tax_rate_range"])
        tax_paid = profit * tax_rate
        
        # Employee count
        num_employees = int(self.np_rng.lognormal(mean=5, sigma=1.5))
        subsidiaries = self.np_rng.poisson(lam=2)
        
        # Compliance indicators
        on_time_filing = self.np_rng.random() < params["on_time_rate"]
        accurate_reporting = self.np_rng.random() < params["on_time_rate"]  # Similar rate
        past_violations = self.np_rng.poisson(lam=params["violations_lambda"])
        audit_score = self.np_rng.uniform(*params["audit_score_range"])
        
        # Continuous compliance score (Section 10.2.2.6)
        base_score = self.np_rng.uniform(*params["score_range"])
        adjustments = 0.0
        if on_time_filing:
            adjustments += 0.05
        if accurate_reporting:
            adjustments += 0.05
        adjustments -= 0.03 * past_violations
        adjustments += 0.10 * (audit_score / 100 - 0.5)
        
        # Add noise and clip
        compliance_score = base_score + adjustments + self.np_rng.normal(0, 0.10)
        compliance_score = np.clip(compliance_score, 0, 1)
        
        return Company(
            company_id=company_id,
            company_name=company_name,
            year=year,
            industry=industry,
            country=country,
            revenue=revenue,
            profit=profit,
            tax_paid=tax_paid,
            num_employees=num_employees,
            subsidiaries=subsidiaries,
            on_time_filing=on_time_filing,
            accurate_reporting=accurate_reporting,
            past_violations=past_violations,
            audit_score=audit_score,
            compliance_level=compliance_level,
            compliance_score=compliance_score
        )
    
    def generate_evidence(
        self,
        company: Company,
        evidence_type: str,
        noise_level: float = 0.0,
        contradictory: bool = False
    ) -> Evidence:
        """
        Generate evidence for a company.
        
        Args:
            company: Company to generate evidence for
            evidence_type: Type of evidence
            noise_level: Noise level [0, 1]
            contradictory: Whether evidence contradicts ground truth
            
        Returns:
            Evidence object
        """
        self.evidence_counter += 1
        evidence_id = f"{company.company_id}_E{self.evidence_counter:03d}"
        
        # Determine which compliance level this evidence supports
        if contradictory:
            # Pick a different compliance level
            other_levels = [l for l in self.COMPLIANCE_LEVELS if l != company.compliance_level]
            supports_compliance = self.rng.choice(other_levels)
        else:
            supports_compliance = company.compliance_level
        
        # Generate timestamp within fiscal year
        base_date = datetime(company.year, 1, 1)
        days_offset = self.rng.randint(0, 364)
        timestamp = (base_date + timedelta(days=days_offset)).isoformat()
        
        # Credibility (reduced by noise)
        base_credibility = 1.0
        credibility = base_credibility * (1.0 - noise_level * 0.5)
        credibility = np.clip(credibility, 0.1, 1.0)
        
        # Generate content based on type
        if evidence_type == "structured":
            text_content = ""
            structured_data = self._generate_structured_data(company, noise_level)
        else:
            text_content = self._generate_text_evidence(
                company, supports_compliance, noise_level
            )
            structured_data = {}
        
        return Evidence(
            evidence_id=evidence_id,
            company_id=company.company_id,
            evidence_type=evidence_type,
            text_content=text_content,
            structured_data=structured_data,
            credibility=credibility,
            timestamp=timestamp,
            supports_compliance=supports_compliance
        )
    
    def _generate_structured_data(
        self,
        company: Company,
        noise_level: float
    ) -> Dict[str, Any]:
        """Generate structured data evidence."""
        # Add Gaussian noise to numeric features
        noise = lambda x, scale: x + self.np_rng.normal(0, x * scale * noise_level)
        
        return {
            "tax_rate": noise(company.tax_paid / company.profit, 0.1),
            "audit_score": noise(company.audit_score, 0.1),
            "violations": company.past_violations,
            "on_time": company.on_time_filing
        }
    
    def _generate_text_evidence(
        self,
        company: Company,
        supports_compliance: str,
        noise_level: float
    ) -> str:
        """Generate text evidence with optional hedging."""
        templates = self.TEXT_TEMPLATES[supports_compliance]
        text = self.rng.choice(templates).format(company=company.company_name)
        
        # Add hedging language for noisy evidence
        if noise_level > 0.3:
            hedges = ["allegedly", "reportedly", "according to sources"]
            hedge = self.rng.choice(hedges)
            text = f"{hedge.capitalize()}, {text.lower()}"
        
        return text
    
    def generate_dataset(
        self,
        n_companies: int,
        years: List[int],
        evidence_per_company: int = 5,
        noise_level: float = 0.0,
        contradictory_rate: float = 0.05
    ) -> Tuple[List[Company], List[Evidence]]:
        """
        Generate complete dataset.
        
        Args:
            n_companies: Number of unique companies
            years: List of fiscal years
            evidence_per_company: Evidence items per company-year
            noise_level: Noise level [0, 1]
            contradictory_rate: Fraction of contradictory evidence
            
        Returns:
            Tuple of (companies, evidence_items)
        """
        companies = []
        evidence_items = []
        
        # Generate companies across years
        for year in years:
            for _ in range(n_companies):
                company = self.generate_company(year)
                companies.append(company)
                
                # Generate evidence for this company
                evidence_types = ["structured", "filing", "audit_report", "news"]
                
                for _ in range(evidence_per_company):
                    evidence_type = self.rng.choice(evidence_types)
                    contradictory = self.np_rng.random() < contradictory_rate
                    
                    evidence = self.generate_evidence(
                        company,
                        evidence_type,
                        noise_level,
                        contradictory
                    )
                    evidence_items.append(evidence)
        
        return companies, evidence_items
    
    def create_splits(
        self,
        companies: List[Company],
        evidence_items: List[Evidence],
        ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)
    ) -> Dict[str, Tuple[List[Company], List[Evidence]]]:
        """
        Create train/val/test splits.
        Entity-based splitting to prevent leakage.
        
        Args:
            companies: List of all companies
            evidence_items: List of all evidence
            ratios: (train, val, test) split ratios
            
        Returns:
            Dictionary with 'train', 'val', 'test' keys
        """
        # Get unique company IDs (base ID without year)
        company_base_ids = list(set([c.company_id for c in companies]))
        self.rng.shuffle(company_base_ids)
        
        # Split company IDs
        n_train = int(len(company_base_ids) * ratios[0])
        n_val = int(len(company_base_ids) * ratios[1])
        
        train_ids = set(company_base_ids[:n_train])
        val_ids = set(company_base_ids[n_train:n_train + n_val])
        test_ids = set(company_base_ids[n_train + n_val:])
        
        # Split companies
        train_companies = [c for c in companies if c.company_id in train_ids]
        val_companies = [c for c in companies if c.company_id in val_ids]
        test_companies = [c for c in companies if c.company_id in test_ids]
        
        # Split evidence
        train_evidence = [e for e in evidence_items if e.company_id in train_ids]
        val_evidence = [e for e in evidence_items if e.company_id in val_ids]
        test_evidence = [e for e in evidence_items if e.company_id in test_ids]
        
        return {
            "train": (train_companies, train_evidence),
            "val": (val_companies, val_evidence),
            "test": (test_companies, test_evidence)
        }
    
    def save_dataset(
        self,
        companies: List[Company],
        evidence_items: List[Evidence],
        output_dir: str,
        prefix: str = ""
    ):
        """
        Save dataset to JSON files.
        
        Args:
            companies: List of companies
            evidence_items: List of evidence
            output_dir: Output directory
            prefix: Filename prefix (e.g., "train_", "test_")
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save companies
        companies_file = output_path / f"{prefix}companies.json"
        with open(companies_file, 'w') as f:
            json.dump([c.to_dict() for c in companies], f, indent=2)
        
        # Save evidence
        evidence_file = output_path / f"{prefix}evidence.json"
        with open(evidence_file, 'w') as f:
            json.dump([e.to_dict() for e in evidence_items], f, indent=2)
        
        print(f"Saved {len(companies)} companies to {companies_file}")
        print(f"Saved {len(evidence_items)} evidence items to {evidence_file}")
    
    def generate_embeddings(
        self,
        companies: List[Company],
        embedding_dim: int = 64
    ) -> np.ndarray:
        """
        Generate embeddings for clustering experiments (Task 5).
        Following Section 10.2.2.5.
        
        Args:
            companies: List of companies
            embedding_dim: Embedding dimension
            
        Returns:
            Array of shape (n_companies, embedding_dim)
        """
        embeddings = []
        
        for company in companies:
            # Base features [0:9]
            base_features = np.array([
                np.log(company.revenue) / 20,
                np.log(company.profit) / 18,
                company.tax_paid / company.profit if company.profit > 0 else 0,
                float(company.on_time_filing),
                float(company.accurate_reporting),
                company.past_violations / 5,
                company.audit_score / 100,
                company.num_employees / 1000,
                company.subsidiaries / 10
            ])
            
            # Categorical encoding [9:11] (hash-based)
            industry_hash = hash(company.industry) % 256 / 256
            country_hash = hash(company.country) % 256 / 256
            categorical = np.array([industry_hash, country_hash])
            
            # Compliance structure [11:41] - Gaussian clusters
            compliance_cluster = np.zeros(30)
            if company.compliance_level == "high":
                compliance_cluster[:10] = self.np_rng.randn(10) * 0.3 + 1.0
            elif company.compliance_level == "medium":
                compliance_cluster[10:20] = self.np_rng.randn(10) * 0.3 + 1.0
            else:  # low
                compliance_cluster[20:30] = self.np_rng.randn(10) * 0.3 + 1.0
            
            # Combine and pad to embedding_dim
            embedding = np.concatenate([base_features, categorical, compliance_cluster])
            
            # Pad or truncate to embedding_dim
            if len(embedding) < embedding_dim:
                padding = np.zeros(embedding_dim - len(embedding))
                embedding = np.concatenate([embedding, padding])
            else:
                embedding = embedding[:embedding_dim]
            
            # Add evidence-based noise (inversely proportional to credibility)
            # For simplicity, use uniform high credibility here
            noise = self.np_rng.randn(embedding_dim) * 0.1
            embedding += noise
            
            embeddings.append(embedding)
        
        return np.array(embeddings)


def create_standard_dataset(output_dir: str = "./data/synthetic"):
    """
    Create the standard synthetic dataset for experiments.
    300 companies × 3 years = 900 company-year records.
    """
    generator = SyntheticDataGenerator(seed=42)
    
    # Generate dataset
    companies, evidence = generator.generate_dataset(
        n_companies=300,
        years=[2020, 2021, 2022],
        evidence_per_company=5,
        noise_level=0.1,
        contradictory_rate=0.05
    )
    
    # Create splits
    splits = generator.create_splits(companies, evidence)
    
    # Save splits
    for split_name, (split_companies, split_evidence) in splits.items():
        generator.save_dataset(
            split_companies,
            split_evidence,
            output_dir,
            prefix=f"{split_name}_"
        )
    
    # Generate and save embeddings for clustering
    all_embeddings = generator.generate_embeddings(companies, embedding_dim=64)
    embeddings_file = Path(output_dir) / "embeddings.npy"
    np.save(embeddings_file, all_embeddings)
    print(f"Saved embeddings to {embeddings_file}")
    
    print(f"\nDataset statistics:")
    print(f"Total companies: {len(companies)}")
    print(f"Total evidence: {len(evidence)}")
    print(f"Train: {len(splits['train'][0])} companies")
    print(f"Val: {len(splits['val'][0])} companies")
    print(f"Test: {len(splits['test'][0])} companies")


if __name__ == "__main__":
    create_standard_dataset()
    