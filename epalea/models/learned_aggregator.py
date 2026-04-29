"""
Learned Evidence Aggregator
This is the missing piece between VAE encoding and SPN reasoning.

Architecture:
  VAE Encoder → Multiple Posteriors → Learned Aggregator → Single Factor → SPN

The aggregator learns to:
1. Assess evidence quality (uncertainty-aware)
2. Detect contradictions
3. Weight evidence based on informativeness
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class Posterior:
    """Latent posterior from VAE encoder."""
    mu: np.ndarray
    logvar: np.ndarray
    evidence_id: str


class EvidenceAggregator(nn.Module):
    """
    Learned aggregator for combining multiple evidence posteriors.
    
    Key idea: Instead of simple averaging, LEARN how to weight
    evidence based on quality and consistency.
    """
    
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Evidence quality network
        # Input: [mu, logvar] → Output: quality score ∈ [0,1]
        self.quality_net = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Pairwise consistency network
        # Input: [mu_i - mu_j, |logvar_i - logvar_j|] → Output: consistency ∈ [0,1]
        self.consistency_net = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Final weight computation
        # Input: [quality, avg_consistency] → Output: evidence weight
        self.weight_net = nn.Sequential(
            nn.Linear(2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Ensure positive
        )
    
    def compute_quality_scores(
        self,
        posteriors: List[Posterior]
    ) -> torch.Tensor:
        """
        Compute quality score for each evidence item.
        High quality = low variance + reasonable magnitude
        
        Returns:
            Tensor of shape [n_evidence]
        """
        n = len(posteriors)
        scores = torch.zeros(n)
        
        for i, post in enumerate(posteriors):
            mu = torch.FloatTensor(post.mu)
            logvar = torch.FloatTensor(post.logvar)
            
            # Concatenate mu and logvar
            features = torch.cat([mu, logvar], dim=0)
            
            # Compute quality
            scores[i] = self.quality_net(features).squeeze()
        
        return scores
    
    def compute_consistency_matrix(
        self,
        posteriors: List[Posterior]
    ) -> torch.Tensor:
        """
        Compute pairwise consistency between evidence items.
        
        Returns:
            Tensor of shape [n_evidence, n_evidence]
        """
        n = len(posteriors)
        consistency = torch.zeros(n, n)
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    consistency[i, j] = 1.0
                    continue
                
                mu_i = torch.FloatTensor(posteriors[i].mu)
                mu_j = torch.FloatTensor(posteriors[j].mu)
                logvar_i = torch.FloatTensor(posteriors[i].logvar)
                logvar_j = torch.FloatTensor(posteriors[j].logvar)
                
                # Features: difference in means and variances
                diff_mu = mu_i - mu_j
                diff_logvar = torch.abs(logvar_i - logvar_j)
                
                features = torch.cat([diff_mu, diff_logvar], dim=0)
                
                # Compute consistency
                consistency[i, j] = self.consistency_net(features).squeeze()
        
        return consistency
    
    def forward(
        self,
        posteriors: List[Posterior]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute aggregation weights for evidence items.
        
        Args:
            posteriors: List of evidence posteriors
            
        Returns:
            weights: Tensor of shape [n_evidence] (normalized to sum to 1)
            diagnostics: Dict with quality scores, consistency, etc.
        """
        if not posteriors:
            return torch.tensor([]), {}
        
        n = len(posteriors)
        
        # 1. Compute quality scores
        quality_scores = self.compute_quality_scores(posteriors)
        
        # 2. Compute consistency
        consistency_matrix = self.compute_consistency_matrix(posteriors)
        
        # Average consistency for each evidence (exclude self)
        avg_consistency = torch.zeros(n)
        for i in range(n):
            if n > 1:
                # Average over all other items
                mask = torch.ones(n, dtype=torch.bool)
                mask[i] = False
                avg_consistency[i] = consistency_matrix[i, mask].mean()
            else:
                avg_consistency[i] = 1.0
        
        # 3. Combine quality and consistency
        weights = torch.zeros(n)
        for i in range(n):
            features = torch.stack([quality_scores[i], avg_consistency[i]])
            weights[i] = self.weight_net(features).squeeze()
        
        # 4. Normalize to sum to 1
        weights = weights / (weights.sum() + 1e-8)
        
        diagnostics = {
            'quality_scores': quality_scores,
            'avg_consistency': avg_consistency,
            'consistency_matrix': consistency_matrix
        }
        
        return weights, diagnostics
    
    def aggregate_posteriors(
        self,
        posteriors: List[Posterior]
    ) -> Posterior:
        """
        Aggregate multiple posteriors into a single weighted posterior.
        
        Args:
            posteriors: List of evidence posteriors
            
        Returns:
            Aggregated posterior
        """
        if not posteriors:
            # Return dummy posterior
            return Posterior(
                mu=np.zeros(self.latent_dim),
                logvar=np.zeros(self.latent_dim),
                evidence_id="none"
            )
        
        if len(posteriors) == 1:
            return posteriors[0]
        
        # Compute weights
        weights, _ = self.forward(posteriors)
        weights_np = weights.detach().cpu().numpy()
        
        # Weighted average of mu
        mu_agg = np.zeros(self.latent_dim)
        logvar_agg = np.zeros(self.latent_dim)
        
        for i, post in enumerate(posteriors):
            w = weights_np[i]
            mu_agg += w * post.mu
            # For variance: combine as weighted sum of variances
            # (this is approximate but reasonable)
            logvar_agg += w * post.logvar
        
        return Posterior(
            mu=mu_agg,
            logvar=logvar_agg,
            evidence_id=f"aggregated_{len(posteriors)}_items"
        )


class AggregatorTrainer:
    """
    Trainer for the evidence aggregator.
    
    Training objective: Learn to weight evidence such that
    the aggregated posterior leads to better calibrated predictions.
    """
    
    def __init__(
        self,
        aggregator: EvidenceAggregator,
        decoder_network,
        learning_rate: float = 1e-3
    ):
        self.aggregator = aggregator
        self.decoder = decoder_network
        self.optimizer = torch.optim.Adam(
            aggregator.parameters(),
            lr=learning_rate
        )
    
    def save_checkpoint(self, path: str):
        """Save aggregator checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'aggregator_state_dict': self.aggregator.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'latent_dim': self.aggregator.latent_dim
        }, path)
        print(f"  ✓ Saved aggregator checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load aggregator checkpoint."""
        if not os.path.exists(path):
            return False
        
        checkpoint = torch.load(path, map_location='cpu')
        self.aggregator.load_state_dict(checkpoint['aggregator_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"  ✓ Loaded aggregator checkpoint from {path}")
        return True
    
    def train_step(
        self,
        posteriors: List[Posterior],
        predicate: str,
        true_label: str,
        domain: List[str]
    ) -> float:
        """
        Single training step - FIXED VERSION.

        Loss: NLL of true label under aggregated distribution
        """
        self.aggregator.train()
        self.optimizer.zero_grad()

        # Get aggregation weights (this involves forward pass through aggregator)
        weights, diagnostics = self.aggregator.forward(posteriors)

        # Manually aggregate posteriors using learned weights
        # This keeps gradient flow intact
        mu_tensors = []
        for post in posteriors:
            mu_tensors.append(torch.FloatTensor(post.mu))

        mu_stacked = torch.stack(mu_tensors)  # [n_evidence, latent_dim]

        # Weighted average using learned weights
        # weights: [n_evidence], mu_stacked: [n_evidence, latent_dim]
        z_agg = torch.sum(weights.unsqueeze(1) * mu_stacked, dim=0)  # [latent_dim]

        # Now decode through the decoder network
        # CRITICAL: We need to use the decoder's forward pass properly

        # Add batch dimension for decoder (expects [batch_size, latent_dim])
        z_agg = z_agg.unsqueeze(0)  # [1, latent_dim]

        # Get predicate embedding
        pred_emb = self.decoder.predicate_embedding(predicate)
        pred_emb_batch = pred_emb.unsqueeze(0)  # [1, embedding_dim]

        # Concatenate z and predicate embedding
        x = torch.cat([z_agg, pred_emb_batch], dim=1)

        # Pass through MLP
        h = self.decoder.mlp(x)

        # Get output head for this predicate (FIXED: use output_heads, not predicate_heads)
        if predicate not in self.decoder.output_heads:
            raise ValueError(f"Unknown predicate: {predicate}")

        logits = self.decoder.output_heads[predicate](h)  # [1, num_classes]

        # Convert to probabilities
        probs = torch.softmax(logits, dim=1).squeeze(0)  # [num_classes]

        # Compute NLL loss
        true_idx = domain.index(true_label)
        loss = -torch.log(probs[true_idx] + 1e-8)

        # Backward and step
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(
        self,
        train_data: List[Dict],
        epochs: int = 30,
        verbose: bool = True
    ):
        """
        Train the aggregator.
        
        Args:
            train_data: List of dicts with:
                - posteriors: List[Posterior]
                - predicate: str
                - true_label: str
                - domain: List[str]
            epochs: Number of epochs
            verbose: Print progress
        """
        if not train_data:
            print("Warning: No training data provided!")
            return
        
        for epoch in range(epochs):
            total_loss = 0.0
            
            for example in train_data:
                loss = self.train_step(
                    example['posteriors'],
                    example['predicate'],
                    example['true_label'],
                    example['domain']
                )
                total_loss += loss
            
            if verbose and (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(train_data)
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")


# Integration with existing system
def add_learned_aggregation_to_orchestrator(orchestrator, latent_dim=64):
    """
    Add learned aggregation to orchestrator.
    
    This modifies the orchestrator to use learned aggregation
    instead of simple averaging.
    """
    # Create aggregator
    orchestrator.aggregator = EvidenceAggregator(
        latent_dim=latent_dim, # Must match VAE
        hidden_dim=latent_dim*2  # 128 (old version)
    )
    
    # Create trainer
    orchestrator.aggregator_trainer = AggregatorTrainer(
        orchestrator.aggregator,
        orchestrator.decoder_network
    )
    
    # Modify _aggregate_vae_predictions to use learned aggregation
    original_aggregate = orchestrator._aggregate_vae_predictions
    
    def learned_aggregate(posteriors_list, predicate):
        """Use learned aggregation instead of simple averaging."""
        if not posteriors_list:
            domain = orchestrator.schema.get_predicate_domain(predicate)
            return {val: 1.0/len(domain) for val in domain}
        
        # Convert to Posterior objects
        from epalea.models.vae_encoder import LatentPosterior
        posteriors = []
        for p in posteriors_list:
            if isinstance(p, LatentPosterior):
                posteriors.append(Posterior(
                    mu=p.mu,
                    logvar=p.logvar,
                    evidence_id=p.evidence_id
                ))
        
        # Aggregate using learned weights
        agg_posterior = orchestrator.aggregator.aggregate_posteriors(posteriors)
        
        # Decode
        z = torch.FloatTensor(agg_posterior.mu)
        distribution = orchestrator.decoder_network.decode(z, predicate)
        
        return distribution
    
    # Replace method
    orchestrator._aggregate_vae_predictions = learned_aggregate
    
    print("✓ Learned aggregation integrated into orchestrator")


def prepare_aggregator_training_data(
    orchestrator,
    train_companies: List[Dict],
    train_evidence: List[Dict],
    predicate_name: str,           # NEW: pass predicate
    label_field: str          # NEW: pass label field
) -> List[Dict]:
    """
    Prepare training data for the aggregator.
    
    For each company:
    1. Retrieve evidence
    2. Encode to posteriors
    3. Create training example with ground truth
    """
    training_data = []
    
    from epalea.models.vae_encoder import LatentPosterior
    
    for company in train_companies:
        entity_id = company['company_id']
        true_label = company[label_field]
        predicate = predicate_name

        """entity_id = company['company_id']
        true_label = company['compliance_level']
        predicate = 'compliance_level'"""
        
        # Retrieve evidence
        evidence_ids = orchestrator.evidence_index.search(
            entity_id, predicate, top_k=10
        )
        
        if not evidence_ids:
            continue
        
        # Encode
        posteriors_raw = orchestrator.vae_encoder.encode(
            evidence_ids,
            orchestrator.evidence_index
        )
        
        if not posteriors_raw:
            continue
        
        # Convert to Posterior objects
        posteriors = []
        for p in posteriors_raw:
            if isinstance(p, LatentPosterior):
                posteriors.append(Posterior(
                    mu=p.mu,
                    logvar=p.logvar,
                    evidence_id=p.evidence_id
                ))
        
        if not posteriors:
            continue
        
        # Get domain
        domain = orchestrator.schema.get_predicate_domain(predicate)
        
        # Create training example
        training_data.append({
            'posteriors': posteriors,
            'predicate': predicate,
            'true_label': true_label,
            'domain': list(domain)
        })
    
    return training_data


if __name__ == "__main__":
    print("Learned Evidence Aggregator for LPF")
    print("=" * 70)
    print("\nThis module adds LEARNED aggregation to the LPF system:")
    print()
    print("Before: VAE → Simple Average → SPN → Prediction")
    print("After:  VAE → Learned Aggregator → SPN → Prediction")
    print()
    print("The aggregator learns to:")
    print("  1. Assess evidence quality (low variance = high quality)")
    print("  2. Detect contradictions (similar posteriors = consistent)")
    print("  3. Weight evidence optimally for calibration")
    print()
    print("This is what makes LPF better than VAE-Only!")
    