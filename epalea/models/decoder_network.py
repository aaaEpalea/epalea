"""Decoder Network - maps (z, predicate) to p(y|z, predicate)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from epalea.models.schema import Schema


class PredicateEmbedding(nn.Module):
    def __init__(self, predicate_names: List[str], embedding_dim: int = 32):
        super().__init__()
        self.predicate_to_idx = {name: idx for idx, name in enumerate(predicate_names)}
        self.embedding = nn.Embedding(len(predicate_names), embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, predicate_name: str) -> torch.Tensor:
        idx = self.predicate_to_idx.get(predicate_name)
        if idx is None:
            raise ValueError(f"Unknown predicate: {predicate_name}")
        return self.embedding(torch.tensor([idx], dtype=torch.long)).squeeze(0)


class DecoderNetwork(nn.Module):
    def __init__(self, latent_dim: int, schema: Schema,
                 hidden_dims: Optional[List[int]] = None,
                 predicate_embedding_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.schema = schema
        self.predicate_embedding_dim = predicate_embedding_dim
        if hidden_dims is None:
            hidden_dims = [128, 64]
        predicate_names = list(schema.predicates.keys())
        self.predicate_embedding = PredicateEmbedding(predicate_names, predicate_embedding_dim)
        input_dim = latent_dim + predicate_embedding_dim
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)
        self.output_heads = nn.ModuleDict()
        for pred_name, predicate in schema.predicates.items():
            self.output_heads[pred_name] = nn.Linear(prev_dim, len(predicate.domain))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, z: torch.Tensor, predicate: str) -> torch.Tensor:
        batch_size = z.shape[0]
        pred_emb = self.predicate_embedding(predicate)
        pred_emb_batch = pred_emb.unsqueeze(0).expand(batch_size, -1)
        x = torch.cat([z, pred_emb_batch], dim=1)
        h = self.mlp(x)
        if predicate not in self.output_heads:
            raise ValueError(f"Unknown predicate: {predicate}")
        logits = self.output_heads[predicate](h)
        return F.softmax(logits, dim=1)

    def decode(self, z: torch.Tensor, predicate: str) -> Dict[str, float]:
        if z.ndim == 1:
            z = z.unsqueeze(0)
        with torch.no_grad():
            probs = self.forward(z, predicate)
            probs_np = probs.squeeze(0).cpu().numpy()
        domain = self.schema.get_predicate_domain(predicate)
        return {value: float(prob) for value, prob in zip(domain, probs_np)}

    def decode_batch(self, z_batch: torch.Tensor, predicate: str) -> List[Dict[str, float]]:
        with torch.no_grad():
            probs = self.forward(z_batch, predicate)
            probs_np = probs.cpu().numpy()
        domain = self.schema.get_predicate_domain(predicate)
        return [{value: float(prob) for value, prob in zip(domain, probs_np[i])}
                for i in range(len(z_batch))]
