"""
VAE Encoder Module - maps evidence to latent posteriors q_phi(z|e).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class LatentPosterior:
    evidence_id: str
    mu: np.ndarray
    sigma: np.ndarray
    confidence: float

    @property
    def logvar(self) -> np.ndarray:
        return 2.0 * np.log(self.sigma + 1e-8)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'evidence_id': self.evidence_id,
            'mu': self.mu.tolist() if isinstance(self.mu, np.ndarray) else self.mu,
            'sigma': self.sigma.tolist() if isinstance(self.sigma, np.ndarray) else self.sigma,
            'confidence': self.confidence
        }


class VAEEncoderNetwork(nn.Module):
    def __init__(self, embedding_dim: int, latent_dim: int,
                 hidden_dims: Optional[List[int]] = None,
                 dropout: float = 0.1, sigma_min: float = 1e-6):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.sigma_min = sigma_min
        if hidden_dims is None:
            hidden_dims = [256, 128]
        layers = []
        prev_dim = embedding_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_log_sigma = nn.Linear(prev_dim, latent_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(embedding)
        mu = self.fc_mu(h)
        log_sigma = self.fc_log_sigma(h)
        sigma = F.softplus(log_sigma) + self.sigma_min
        return mu, sigma

    def encode(self, embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward(embedding)


class VAEEncoder:
    def __init__(self, encoder_network: VAEEncoderNetwork,
                 embedding_model: Optional[Any] = None, device: str = "cpu"):
        self.encoder_network = encoder_network.to(device)
        self.embedding_model = embedding_model
        self.device = device
        self.encoder_network.eval()

    def encode(self, evidence_ids: List[str], evidence_index: Any) -> List[LatentPosterior]:
        posteriors = []
        with torch.no_grad():
            for eid in evidence_ids:
                meta = evidence_index.fetch_meta(eid)
                if meta is None:
                    continue
                if hasattr(meta, 'embedding_id') and meta.embedding_id is not None:
                    embedding = self._fetch_embedding_from_index(evidence_index, meta.embedding_id)
                else:
                    raw = evidence_index.fetch_raw(eid)
                    if raw is None:
                        continue
                    embedding = self._compute_embedding(raw)
                embedding_tensor = torch.FloatTensor(embedding).unsqueeze(0).to(self.device)
                mu, sigma = self.encoder_network(embedding_tensor)
                mu_np = mu.squeeze(0).cpu().numpy()
                sigma_np = sigma.squeeze(0).cpu().numpy()
                mean_sigma = float(np.mean(sigma_np))
                confidence = 1.0 / (1.0 + mean_sigma)
                posteriors.append(LatentPosterior(evidence_id=eid, mu=mu_np,
                                                   sigma=sigma_np, confidence=confidence))
        return posteriors

    def _fetch_embedding_from_index(self, evidence_index: Any, embedding_id: int) -> np.ndarray:
        vector = evidence_index.vector_store.index.reconstruct(int(embedding_id))
        return vector

    def _compute_embedding(self, text: str) -> np.ndarray:
        if self.embedding_model is None:
            raise ValueError("No embedding model provided")
        return self.embedding_model.encode(text)

    def encode_batch(self, embeddings: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        with torch.no_grad():
            t = torch.FloatTensor(embeddings).to(self.device)
            mu, sigma = self.encoder_network(t)
            mu_np = mu.cpu().numpy()
            sigma_np = sigma.cpu().numpy()
            return [(mu_np[i], sigma_np[i]) for i in range(len(embeddings))]

    def train_mode(self): self.encoder_network.train()
    def eval_mode(self): self.encoder_network.eval()

    def save(self, path: str):
        torch.save({'network_state_dict': self.encoder_network.state_dict(),
                    'embedding_dim': self.encoder_network.embedding_dim,
                    'latent_dim': self.encoder_network.latent_dim}, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder_network.load_state_dict(checkpoint['network_state_dict'])
        self.encoder_network.eval()


def compute_kl_divergence(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.sum(sigma**2 + mu**2 - 1.0 - torch.log(sigma**2 + 1e-8), dim=1)


def reparameterize(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return mu + sigma * torch.randn_like(sigma)


def sample_from_posterior(mu: np.ndarray, sigma: np.ndarray, n_samples: int = 1) -> np.ndarray:
    latent_dim = len(mu)
    epsilon = np.random.randn(n_samples, latent_dim)
    return mu[np.newaxis, :] + sigma[np.newaxis, :] * epsilon
