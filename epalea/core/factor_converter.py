"""
Factor Converter Module
Converts VAE latent posteriors into soft likelihood factors for SPN reasoning.
Implements Algorithm 1: ConvertLatentToFactors from the paper.
"""

import numpy as np
import torch
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from epalea.models.vae_encoder import LatentPosterior, sample_from_posterior
from epalea.models.decoder_network import DecoderNetwork
from epalea.models.schema import Schema


@dataclass
class SoftFactor:
    """
    Soft likelihood factor for SPN.
    Represents Φ_e(y) from Section 4.1.
    """
    evidence_id: str
    variables: List[str]  # Variable names this factor depends on
    factor_type: str  # "likelihood"
    potential: Dict[str, float]  # Distribution over predicate values
    weight: float  # Credibility weight w(e)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'evidence_id': self.evidence_id,
            'variables': self.variables,
            'factor_type': self.factor_type,
            'potential': self.potential,
            'weight': self.weight,
            'metadata': self.metadata
        }


class FactorConverter:
    """
    Converts latent posteriors to soft factors.
    Implements Algorithm 1: ConvertLatentToFactors
    
    The key transformation:
        Φ_e(y) = ∫ p_θ(y|z) q_φ(z|e) dz
    
    Approximated with Monte Carlo:
        Φ̂_e(y) = (1/M) Σ_m p_θ(y|z^(m)) where z^(m) ~ q_φ(z|e)
    """
    
    def __init__(
        self,
        decoder_network: DecoderNetwork,
        schema: Schema,
        n_samples: int = 16,
        temperature: float = 1.0,
        alpha: float = 2.0,
        device: str = "cpu"
    ):
        """
        Initialize factor converter.
        
        Args:
            decoder_network: Trained decoder network
            schema: Schema defining predicates and variables
            n_samples: Number of MC samples (M in the paper)
            temperature: Temperature for softening distributions (T)
            alpha: Weight penalty parameter (α)
            device: Device to run on
        """
        self.decoder_network = decoder_network.to(device)
        self.schema = schema
        self.n_samples = n_samples
        self.temperature = temperature
        self.alpha = alpha
        self.device = device
        self.decoder_network.eval()
    
    def convert_latent_to_factors(
        self,
        latent_posteriors: List[LatentPosterior],
        predicate: str
    ) -> List[SoftFactor]:
        """
        Convert latent posteriors to soft factors.
        Implements Algorithm 1: ConvertLatentToFactors
        
        Args:
            latent_posteriors: List of LatentPosterior objects
            predicate: Predicate name to decode
            
        Returns:
            List of SoftFactor objects
        """
        soft_factors = []
        
        for posterior in latent_posteriors:
            # Lines 3-6: Extract parameters
            eid = posterior.evidence_id
            mu = posterior.mu
            sigma = posterior.sigma
            base_conf = posterior.confidence
            
            # Lines 8-14: Draw reparameterized samples
            # z^(k) = μ + σ ⊙ ε where ε ~ N(0, I)
            z_samples = sample_from_posterior(mu, sigma, n_samples=self.n_samples)
            
            # Lines 16-21: Decode each sample
            z_samples_tensor = torch.FloatTensor(z_samples).to(self.device)
            
            with torch.no_grad():
                # Get p_θ(y|z) for each sample
                pred_dists = self.decoder_network.decode_batch(
                    z_samples_tensor,
                    predicate
                )
            
            # Lines 23-27: Aggregate distributions (Monte Carlo averaging)
            # Φ̂_e(y) = (1/M) Σ_m p_θ(y|z^(m))
            aggregated = self._aggregate_distributions(pred_dists)
            
            # Lines 29-36: Temperature scaling
            if self.temperature != 1.0:
                aggregated = self._apply_temperature(aggregated, self.temperature)
            
            # Normalize after temperature
            aggregated = self._normalize_distribution(aggregated)
            
            # Lines 38-39: Compute factor weight
            weight = self._compute_weight(sigma, base_conf)
            
            # Lines 42-50: Build factor
            variables = self.schema.get_variables_for_predicate(predicate)
            
            factor = SoftFactor(
                evidence_id=eid,
                variables=variables,
                factor_type="likelihood",
                potential=aggregated,
                weight=weight,
                metadata={
                    'n_samples': self.n_samples,
                    'temperature': self.temperature,
                    'base_confidence': base_conf,
                    'mean_sigma': float(np.mean(sigma)),
                    'predicate': predicate
                }
            )
            
            soft_factors.append(factor)
        
        return soft_factors
    
    def _aggregate_distributions(
        self,
        distributions: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Aggregate multiple distributions by averaging.
        Lines 23-27 of Algorithm 1.
        
        Args:
            distributions: List of probability distributions
            
        Returns:
            Averaged distribution
        """
        if not distributions:
            return {}
        
        # Get all keys
        keys = distributions[0].keys()
        
        # Average each key
        aggregated = {}
        for key in keys:
            values = [d[key] for d in distributions]
            aggregated[key] = np.mean(values)
        
        return aggregated
    
    def _apply_temperature(
        self,
        distribution: Dict[str, float],
        temperature: float
    ) -> Dict[str, float]:
        """
        Apply temperature scaling to soften/sharpen distribution.
        Lines 29-36 of Algorithm 1.
        
        Transform: p^(1/T) / Σ p^(1/T)
        - T > 1: softens (reduces overconfidence)
        - T < 1: sharpens (increases confidence)
        - T = 1: no change
        
        Args:
            distribution: Original distribution
            temperature: Temperature parameter T
            
        Returns:
            Temperature-scaled distribution
        """
        if temperature == 1.0:
            return distribution
        
        # Apply temperature: raise to power 1/T
        scaled = {}
        for key, prob in distribution.items():
            scaled[key] = prob ** (1.0 / temperature)
        
        return scaled
    
    def _normalize_distribution(
        self,
        distribution: Dict[str, float],
        eps: float = 1e-12
    ) -> Dict[str, float]:
        """
        Normalize distribution to sum to 1.
        Lines 35-36 of Algorithm 1.
        
        Args:
            distribution: Unnormalized distribution
            eps: Small constant for numerical stability
            
        Returns:
            Normalized distribution
        """
        total = sum(distribution.values()) + eps
        normalized = {key: val / total for key, val in distribution.items()}
        return normalized
    
    def _compute_weight(
        self,
        sigma: np.ndarray,
        base_conf: float
    ) -> float:
        """
        Compute credibility weight for the factor.
        Line 39 of Algorithm 1 and Section 4.2.
        
        weight = base_conf × sigmoid(-α × mean(σ))
        
        This downweights evidence with high uncertainty.
        
        Args:
            sigma: Standard deviation vector
            base_conf: Base confidence from encoder
            
        Returns:
            Weight in [0, 1]
        """
        mean_sigma = float(np.mean(sigma))
        
        # Sigmoid for smooth weighting
        calibration_weight = 1.0 / (1.0 + np.exp(self.alpha * mean_sigma))
        
        weight = base_conf * calibration_weight
        
        # Ensure in valid range
        weight = np.clip(weight, 0.0, 1.0)
        
        return float(weight)
    
    def apply_weighted_factor(
        self,
        factor: SoftFactor
    ) -> Dict[str, float]:
        """
        Apply weight to factor potential.
        Section 5.6: Φ̃_e(y) = (p(y|e))^w(e) / Σ (p(y'|e))^w(e)
        
        Args:
            factor: Soft factor
            
        Returns:
            Weighted and normalized potential
        """
        weighted = {}
        
        # Raise each probability to power w(e)
        for key, prob in factor.potential.items():
            weighted[key] = prob ** factor.weight
        
        # Normalize
        weighted = self._normalize_distribution(weighted)
        
        return weighted
    
    def set_temperature(self, temperature: float):
        """Update temperature parameter."""
        self.temperature = temperature
    
    def set_n_samples(self, n_samples: int):
        """Update number of MC samples."""
        self.n_samples = n_samples
    
    def set_alpha(self, alpha: float):
        """Update weight penalty parameter."""
        self.alpha = alpha


def compute_monte_carlo_error(
    n_samples: int,
    confidence_level: float = 0.95,
    return_ci: bool = False
) -> float:
    """
    Compute Monte Carlo error estimates.

    Standard Error (SE) ≈ sqrt(0.25 / M)
    If return_ci=True, return the CI half-width (z * SE).
    """
    variance_bound = 0.25 / n_samples
    se = np.sqrt(variance_bound)

    if not return_ci:
        # Tests expect THIS behavior
        return se

    # Compute CI half-width if requested
    if confidence_level == 0.95:
        z = 1.96
    elif confidence_level == 0.99:
        z = 2.576
    else:
        from scipy.stats import norm
        z = norm.ppf((1 + confidence_level) / 2)

    return z * se

def get_recommended_n_samples(
    target_error: float = 0.05
) -> int:
    """
    Get recommended number of samples for target error.
    
    From Section 5.9 table:
    - M=16: SE ≈ 0.125
    - M=32: SE ≈ 0.088
    - M=64: SE ≈ 0.063
    
    Args:
        target_error: Target standard error
        
    Returns:
        Recommended number of samples
    """
    # SE = sqrt(0.25 / M)
    # M = 0.25 / SE²
    n_samples = int(np.ceil(0.25 / (target_error ** 2)))
    return n_samples


def batch_convert_latent_to_factors(
    converter: FactorConverter,
    posteriors_by_predicate: Dict[str, List[LatentPosterior]]
) -> Dict[str, List[SoftFactor]]:
    """
    Convert posteriors for multiple predicates efficiently.
    
    Args:
        converter: FactorConverter instance
        posteriors_by_predicate: Dict mapping predicate → list of posteriors
        
    Returns:
        Dict mapping predicate → list of soft factors
    """
    factors_by_predicate = {}
    
    for predicate, posteriors in posteriors_by_predicate.items():
        factors = converter.convert_latent_to_factors(posteriors, predicate)
        factors_by_predicate[predicate] = factors
    
    return factors_by_predicate
