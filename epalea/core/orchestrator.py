"""
Orchestrator Module
Main query engine that coordinates all components.
Implements Algorithm 4: Orchestrator.HandleQuery from the paper.
"""

import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from epalea.models.schema import Schema
from epalea.core.canonical_db import CanonicalDB
from epalea.core.evidence_index import EvidenceIndex
from epalea.models.vae_encoder import VAEEncoder
from epalea.models.decoder_network import DecoderNetwork
from epalea.core.factor_converter import FactorConverter
from epalea.models.spn_module import SPN, build_spn_for_schema
from epalea.core.provenance_ledger import ProvenanceLedger, create_factor_metadata_summary


@dataclass
class QueryOptions:
    """Options for query execution."""
    top_k: int = 10  # Number of evidence items to retrieve
    n_samples: int = 16  # MC samples for factor conversion
    temperature: float = 1.0  # Temperature scaling
    alpha: float = 2.0  # Weight penalty
    max_age_days: float = 30.0  # Staleness threshold for canonical
    use_canonical: bool = True  # Check canonical DB first
    use_spn: bool = True  # Use SPN reasoning (vs pure VAE)


class Orchestrator:
    """
    Main orchestrator for epistemic queries.
    
    Implements the complete pipeline:
    1. Check canonical DB (fast path)
    2. Retrieve evidence
    3. Encode to latent posteriors
    4. Convert to soft factors
    5. SPN reasoning
    6. Provenance logging
    
    This is Algorithm 4: Orchestrator.HandleQuery
    """
    
    def __init__(
        self,
        schema: Schema,
        canonical_db: CanonicalDB,
        evidence_index: EvidenceIndex,
        vae_encoder: VAEEncoder,
        decoder_network: DecoderNetwork,
        provenance_ledger: Optional[ProvenanceLedger] = None,
        device: str = "cpu"
    ):
        """
        Initialize orchestrator.
        
        Args:
            schema: Schema defining predicates
            canonical_db: Canonical database for fast path
            evidence_index: Evidence retrieval system
            vae_encoder: VAE encoder for evidence
            decoder_network: Decoder for latent→distribution
            provenance_ledger: Optional provenance tracking
            device: Device for computation
        """
        self.schema = schema
        self.canonical_db = canonical_db
        self.evidence_index = evidence_index
        self.vae_encoder = vae_encoder
        self.decoder_network = decoder_network
        self.provenance_ledger = provenance_ledger
        self.device = device
        
        # Create factor converter
        self.factor_converter = FactorConverter(
            decoder_network=decoder_network,
            schema=schema,
            n_samples=16,
            temperature=1.0,
            alpha=2.0,
            device=device
        )
        
        # Cache SPNs by predicate
        self.spn_cache: Dict[str, SPN] = {}

        # Aggregator components (initialized as None, set by add_learned_aggregation_to_orchestrator)
        self.aggregator: Optional[Any] = None  # Type: Optional[EvidenceAggregator]
        self.aggregator_trainer: Optional[Any] = None  # Type: Optional[AggregatorTrainer]
    
    def query(
        self,
        entity_id: str,
        predicate: str,
        options: Optional[QueryOptions] = None,
        return_uncertainty: bool = False
    ) -> Dict[str, Any]:
        """
        Handle a query for an entity and predicate.
        Implements Algorithm 4: Orchestrator.HandleQuery
        
        Args:
            entity_id: Entity identifier
            predicate: Predicate to query
            options: Query options
            
        Returns:
            Dictionary with:
                - distribution: Probability distribution
                - top_value: Most likely value
                - confidence: Confidence in top value
                - source: "canonical" or "inference"
                - evidence_chain: Evidence IDs used (if inference)
                - audit_ptr: Provenance record ID (if ledger enabled)
                - execution_time_ms: Query execution time
        """
        start_time = time.time()
        
        if options is None:
            options = QueryOptions()
        
        # Lines 2-5: Fast canonical check
        if options.use_canonical:
            canonical_result = self._check_canonical(
                entity_id, 
                predicate, 
                options.max_age_days
            )
            if canonical_result is not None:
                execution_time = (time.time() - start_time) * 1000
                canonical_result['execution_time_ms'] = execution_time
                return canonical_result
        
        # Lines 7-11: Retrieve evidence
        evidence_ids = self._retrieve_evidence(entity_id, predicate, options.top_k)
        
        if not evidence_ids:
            execution_time = (time.time() - start_time) * 1000
            return self._no_evidence_result(execution_time)
        
        # Lines 13-14: Encode evidence
        posteriors = self.vae_encoder.encode(evidence_ids, self.evidence_index)
        
        if not posteriors:
            execution_time = (time.time() - start_time) * 1000
            return self._no_evidence_result(execution_time)
        
        # Lines 16-18: Convert to factors
        self.factor_converter.set_n_samples(options.n_samples)
        self.factor_converter.set_temperature(options.temperature)
        self.factor_converter.set_alpha(options.alpha)
        
        soft_factors = self.factor_converter.convert_latent_to_factors(
            posteriors,
            predicate
        )
        
        # Lines 20-21: Gather conditionals (hard evidence)
        conditionals = self._get_related_facts(entity_id)
        
        # Lines 23-28: Run inference
        if options.use_spn and self.schema.covers_predicate(predicate):
            posterior = self._spn_query(predicate, soft_factors, conditionals)
        else:
            # Fallback: aggregate VAE predictions without SPN
            posterior = self._aggregate_vae_predictions(posteriors, predicate)
        
        # Lines 30-33: Compute confidence
        if posterior:
            top_value = max(posterior.keys(), key=lambda k: posterior[k])
            confidence = posterior[top_value]
        else:
            top_value = "unknown"
            confidence = 0.0
        
        execution_time = (time.time() - start_time) * 1000
        
        # Lines 34: Append to provenance ledger
        audit_ptr = None
        if self.provenance_ledger is not None:
            try:
                factor_metadata = create_factor_metadata_summary(soft_factors)
            except Exception as e:
                # Fallback if create_factor_metadata_summary fails
                factor_metadata = []
            
            try:
                record = self.provenance_ledger.append_inference_record(
                    entity_id=entity_id,
                    predicate=predicate,
                    distribution=posterior,
                    evidence_chain=evidence_ids,
                    factor_metadata=factor_metadata,
                    model_versions={
                        'encoder': 'vae_v1.0',
                        'decoder': 'decoder_v1.0'
                    },
                    hyperparameters={
                        'n_samples': options.n_samples,
                        'temperature': options.temperature,
                        'alpha': options.alpha,
                        'top_k': options.top_k
                    },
                    execution_time_ms=execution_time
                )
                audit_ptr = record.record_id
            except Exception as e:
                print(f"Warning: Failed to create provenance record: {e}")
                import traceback
                traceback.print_exc()
        
        uncertainty_info = {}
        if return_uncertainty and posteriors:
            uncertainty_info = self.compute_uncertainty_decomposition(
                posteriors, predicate, n_samples=options.n_samples
            )
        
        # Lines 35-40: Return result
        return {
            'distribution': posterior,
            'top_value': top_value,
            'confidence': confidence,
            'source': 'inference',
            'evidence_chain': evidence_ids,
            'num_factors': len(soft_factors),
            'audit_ptr': audit_ptr,
            'execution_time_ms': execution_time,
            **uncertainty_info
        }
    
    def _check_canonical(
        self,
        entity_id: str,
        predicate: str,
        max_age_days: float
    ) -> Optional[Dict[str, Any]]:
        """
        Check canonical DB for authoritative value.
        Lines 2-5 of Algorithm 4.
        
        Args:
            entity_id: Entity identifier
            predicate: Predicate name
            max_age_days: Maximum age in days
            
        Returns:
            Result dict if found and fresh, None otherwise
        """
        canonical = self.canonical_db.get(entity_id, predicate)
        
        if canonical is None:
            return None
        
        # Check staleness
        if self.canonical_db.is_stale(entity_id, predicate, max_age_days):
            return None
        
        # Return canonical result
        domain = self.schema.get_predicate_domain(predicate)
        distribution = {val: 0.0 for val in domain}
        distribution[canonical.value] = 1.0
        
        return {
            'distribution': distribution,
            'top_value': canonical.value,
            'confidence': canonical.confidence,
            'source': 'canonical',
            'canonical_timestamp': canonical.timestamp,
            'audit_ptr': None
        }
    
    def _retrieve_evidence(
        self,
        entity_id: str,
        predicate: str,
        top_k: int
    ) -> List[str]:
        """
        Retrieve evidence from index.
        Lines 7-11 of Algorithm 4.
        
        Args:
            entity_id: Entity identifier
            predicate: Predicate name
            top_k: Number of evidence items to retrieve
            
        Returns:
            List of evidence IDs
        """
        return self.evidence_index.search(entity_id, predicate, top_k)
    
    def _get_related_facts(self, entity_id: str) -> Dict[str, Any]:
        """
        Get related canonical facts for entity.
        Line 21 of Algorithm 4.
        
        Args:
            entity_id: Entity identifier
            
        Returns:
            Dictionary of variable→value mappings
        """
        # Get all canonical facts for entity
        records = self.canonical_db.get_all_for_entity(entity_id)
        
        conditionals = {}
        for record in records:
            # Map predicate to variable values
            # Simplified: use predicate value directly
            conditionals[record.predicate] = record.value
        
        return conditionals
    
    def _spn_query(
        self,
        predicate: str,
        soft_factors: List[Any],
        conditionals: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Query using SPN reasoning.
        Lines 24-26 of Algorithm 4.
        
        Args:
            predicate: Predicate to query
            soft_factors: Soft likelihood factors
            conditionals: Hard evidence
            
        Returns:
            Posterior distribution
        """
        # Get or build SPN
        if predicate not in self.spn_cache:
            self.spn_cache[predicate] = build_spn_for_schema(self.schema, predicate)
        
        spn = self.spn_cache[predicate]
        
        # Reset and attach factors
        spn.reset_likelihoods()
        spn.attach_soft_factors(soft_factors)
        
        # Query - for simplicity, query the first variable
        # In production, query the actual predicate variable
        variables = self.schema.get_variables_for_predicate(predicate)
        if not variables:
            return {}
        
        query_variable = variables[0]
        
        # Run marginal inference
        posterior = spn.query(query_variable, evidence=conditionals)
        
        # Add numerical stability check
        total = sum(posterior.values())
        if total > 1e-10:
            # Normalize to ensure it sums to 1.0
            posterior = {k: v / total for k, v in posterior.items()}
        else:
            # Fallback to uniform distribution
            domain = self.schema.get_predicate_domain(predicate)
            posterior = {val: 1.0 / len(domain) for val in domain}
        
        return posterior
    
    def _aggregate_vae_predictions(
        self,
        posteriors: List[Any],
        predicate: str
    ) -> Dict[str, float]:
        """
        Aggregate VAE predictions without SPN (fallback).
        Line 27 of Algorithm 4.
        
        Args:
            posteriors: List of latent posteriors
            predicate: Predicate name
            
        Returns:
            Aggregated distribution
        """
        import torch
        
        # Get domain
        domain = self.schema.get_predicate_domain(predicate)
        
        # Decode each posterior and average
        distributions = []
        
        for posterior in posteriors:
            z = torch.FloatTensor(posterior.mu).to(self.device)
            dist = self.decoder_network.decode(z, predicate)
            distributions.append(dist)
        
        # Average distributions
        if not distributions:
            return {val: 1.0 / len(domain) for val in domain}
        
        aggregated = {val: 0.0 for val in domain}
        for dist in distributions:
            for val in domain:
                aggregated[val] += dist.get(val, 0.0)
        
        # Normalize
        total = sum(aggregated.values())
        if total > 1e-10:  # Add numerical stability check
            aggregated = {val: prob / total for val, prob in aggregated.items()}
        else:
            # Fallback to uniform distribution if all values are near zero
            aggregated = {val: 1.0 / len(domain) for val in domain}
        
        return aggregated
    
    def compute_uncertainty_decomposition(
        self,
        posteriors: List[Any],
        predicate: str,
        n_samples: int = 100
    ) -> Dict[str, float]:
        """
        Compute epistemic/aleatoric uncertainty decomposition (Theorem 7).
        
        Args:
            posteriors: List of latent posteriors from VAE encoder
            predicate: Predicate being queried
            n_samples: Number of Monte Carlo samples
            
        Returns:
            Dict with epistemic_uncertainty, aleatoric_uncertainty, 
            total_uncertainty, decomposition_error
        """
        import torch
        import numpy as np

        # Fix seed for deterministic MC sampling
        torch.manual_seed(16)
        np.random.seed(16)
        
        if not posteriors:
            return {
                'epistemic_uncertainty': 0.0,
                'aleatoric_uncertainty': 0.0,
                'total_uncertainty': 0.0,
                'decomposition_error': 1.0
            }
        
        # Extract mu and sigma from posteriors
        mu_list = []
        sigma_list = []
        for p in posteriors:
            mu_list.append(torch.tensor(p.mu, dtype=torch.float32))
            # Handle diagonal covariance
            if isinstance(p.sigma, np.ndarray):
                if p.sigma.ndim == 2:
                    sigma_diag = torch.tensor(np.diag(p.sigma), dtype=torch.float32)
                else:
                    sigma_diag = torch.tensor(p.sigma, dtype=torch.float32)
            else:
                sigma_diag = torch.tensor(p.sigma, dtype=torch.float32)
            sigma_list.append(sigma_diag)
        
        mu_stacked = torch.stack(mu_list)  # [K, latent_dim]
        sigma_stacked = torch.stack(sigma_list)  # [K, latent_dim]
        
        # Get aggregation weights
        if hasattr(self, 'aggregator_trainer') and self.aggregator_trainer:
            try:
                weights, _ = self.aggregator_trainer.aggregator.forward(posteriors)
                weights = weights.float()
            except:
                weights = torch.ones(len(posteriors), dtype=torch.float32) / len(posteriors)
        else:
            weights = torch.ones(len(posteriors), dtype=torch.float32) / len(posteriors)
        
        weights = weights / weights.sum()
        
        # Aggregate posterior: q(Z|E) = sum_i w_i * N(mu_i, sigma_i)
        z_mu = torch.sum(weights.unsqueeze(1) * mu_stacked, dim=0)
        z_second_moment = torch.sum(
            weights.unsqueeze(1) * (sigma_stacked + mu_stacked**2), 
            dim=0
        )
        z_var = torch.clamp(z_second_moment - z_mu**2, min=1e-8)
        
        # Sample from q(Z|E)
        z_samples = []
        for _ in range(n_samples):
            epsilon = torch.randn_like(z_mu)
            z_sample = z_mu + torch.sqrt(z_var) * epsilon
            z_samples.append(z_sample)
        
        z_samples = torch.stack(z_samples)  # [n_samples, latent_dim]
        
        # Decode samples to get p(y|z)
        try:
            with torch.no_grad():
                probs_batch = self.decoder_network(z_samples, predicate)
                probs = probs_batch.cpu().numpy()  # [n_samples, n_classes]
        except Exception:
            probs_list = []
            for z in z_samples:
                try:
                    with torch.no_grad():
                        prob = self.decoder_network(z.unsqueeze(0), predicate)
                        probs_list.append(prob.squeeze(0).cpu().numpy())
                except Exception:
                    continue
            
            if not probs_list:
                return {
                    'epistemic_uncertainty': 0.0,
                    'aleatoric_uncertainty': 0.0,
                    'total_uncertainty': 0.0,
                    'decomposition_error': 1.0
                }
            
            probs = np.array(probs_list)
        
        # Uncertainty decomposition (Theorem 7)
        mean_probs = np.mean(probs, axis=0)
        total_var = np.sum(mean_probs * (1 - mean_probs))
        aleatoric_var = np.mean([np.sum(p * (1 - p)) for p in probs])
        epistemic_var = np.sum(np.var(probs, axis=0))
        
        decomp_sum = epistemic_var + aleatoric_var
        decomp_error = abs(total_var - decomp_sum) / (total_var + 1e-8)
        
        return {
            'epistemic_uncertainty': float(epistemic_var),
            'aleatoric_uncertainty': float(aleatoric_var),
            'total_uncertainty': float(total_var),
            'decomposition_error': float(decomp_error)
        }

    def _no_evidence_result(self, execution_time_ms: float) -> Dict[str, Any]:
        """Return result when no evidence is found."""
        return {
            'distribution': {},
            'top_value': 'unknown',
            'confidence': 0.0,
            'source': 'no_evidence',
            'evidence_chain': [],
            'num_factors': 0,
            'audit_ptr': None,
            'execution_time_ms': execution_time_ms
        }
    
    def batch_query(
        self,
        queries: List[tuple],
        options: Optional[QueryOptions] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute batch queries.
        
        Args:
            queries: List of (entity_id, predicate) tuples
            options: Query options
            
        Returns:
            List of result dictionaries
        """
        results = []
        for entity_id, predicate in queries:
            result = self.query(entity_id, predicate, options)
            results.append(result)
        return results
    