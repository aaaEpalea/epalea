"""
Evidence Index Module
FAISS-based vector store + JSONL metadata store for evidence retrieval.
Implements the EvidenceIndex from Algorithm 2.
"""

import jsonlines
import numpy as np
import faiss
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from sentence_transformers import SentenceTransformer


@dataclass
class EvidenceMetadata:
    """Metadata for a piece of evidence."""
    evidence_id: str
    entity_id: str
    predicate: str
    text_content: Optional[str] = None
    structured_data: Optional[Dict[str, Any]] = None
    credibility: float = 1.0
    timestamp: Optional[str] = None
    evidence_type: str = "text"
    source: str = ""
    embedding_id: Optional[int] = None  # Index in FAISS
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'EvidenceMetadata':
        return cls(**d)


class FaissVectorStore:
    """
    FAISS-based vector store for evidence embeddings.
    """
    
    def __init__(self, dimension: int, index_type: str = "Flat"):
        """
        Initialize FAISS vector store.
        
        Args:
            dimension: Embedding dimension
            index_type: FAISS index type ("Flat" for exact search, "IVF" for approximate)
        """
        self.dimension = dimension
        self.index_type = index_type
        
        if index_type == "Flat":
            self.index = faiss.IndexFlatL2(dimension)
        elif index_type == "IVF":
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        self.count = 0
    
    def add(self, embeddings: np.ndarray) -> List[int]:
        """
        Add embeddings to the index.
        
        Args:
            embeddings: Array of shape (n, dimension)
            
        Returns:
            List of indices for the added embeddings
        """
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        
        assert embeddings.shape[1] == self.dimension, \
            f"Expected dimension {self.dimension}, got {embeddings.shape[1]}"
        
        # Train index if needed (for IVF)
        if self.index_type == "IVF" and not self.index.is_trained:
            self.index.train(embeddings.astype('float32')) # type: ignore
        
        start_idx = self.count
        self.index.add(embeddings.astype('float32')) # type: ignore
        self.count += len(embeddings)
        
        return list(range(start_idx, self.count))
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors.
        
        Args:
            query_embedding: Query vector of shape (dimension,) or (1, dimension)
            k: Number of neighbors to retrieve
            
        Returns:
            Tuple of (distances, indices) arrays of shape (1, k)
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        assert query_embedding.shape[1] == self.dimension
        
        k = min(k, self.count)  # Don't ask for more than available
        if k == 0:
            return np.array([[]]), np.array([[]])
        
        distances, indices = self.index.search(query_embedding.astype('float32'), k) # type: ignore
        return distances, indices
    
    def save(self, path: str):
        """Save index to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, path)
    
    def load(self, path: str):
        """Load index from disk."""
        self.index = faiss.read_index(path)
        self.count = self.index.ntotal
    
    def __len__(self) -> int:
        return self.count


class MetadataStore:
    """
    JSONL-based metadata store.
    Maps evidence_id to metadata and embedding_id to evidence_id.
    """
    
    def __init__(self, path: Optional[str] = None):
        """
        Initialize metadata store.
        
        Args:
            path: Path to JSONL file. If None, uses in-memory storage.
        """
        self.path = path
        self.metadata: Dict[str, EvidenceMetadata] = {}
        self.embedding_to_evidence: Dict[int, str] = {}
        
        if path and Path(path).exists():
            self.load(path)
    
    def add(self, metadata: EvidenceMetadata):
        """Add metadata for an evidence item."""
        self.metadata[metadata.evidence_id] = metadata
        if metadata.embedding_id is not None:
            self.embedding_to_evidence[metadata.embedding_id] = metadata.evidence_id
    
    def get(self, evidence_id: str) -> Optional[EvidenceMetadata]:
        """Get metadata by evidence_id."""
        return self.metadata.get(evidence_id)
    
    def get_by_embedding_id(self, embedding_id: int) -> Optional[EvidenceMetadata]:
        """Get metadata by embedding_id (FAISS index)."""
        evidence_id = self.embedding_to_evidence.get(embedding_id)
        if evidence_id is None:
            return None
        return self.metadata.get(evidence_id)
    
    def save(self, path: Optional[str] = None):
        """Save metadata to JSONL file."""
        save_path = path or self.path
        if save_path is None:
            raise ValueError("No path specified for saving")
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with jsonlines.open(save_path, mode='w') as writer:
            for meta in self.metadata.values():
                writer.write(meta.to_dict())
    
    def load(self, path: str):
        """Load metadata from JSONL file."""
        self.metadata.clear()
        self.embedding_to_evidence.clear()
        
        with jsonlines.open(path) as reader:
            for obj in reader:
                meta = EvidenceMetadata.from_dict(obj)
                self.add(meta)
    
    def __len__(self) -> int:
        return len(self.metadata)


class EvidenceIndex:
    """
    Combined evidence index with vector search and metadata storage.
    Implements the EvidenceIndex interface from Algorithm 2.
    """
    
    def __init__(
        self,
        embedding_model: Optional[SentenceTransformer] = None,
        embedding_dim: int = 384,
        vector_store_path: Optional[str] = None,
        metadata_store_path: Optional[str] = None
    ):
        """
        Initialize evidence index.
        
        Args:
            embedding_model: Model for encoding text to embeddings
            embedding_dim: Dimension of embeddings
            vector_store_path: Path to save/load FAISS index
            metadata_store_path: Path to save/load metadata JSONL
        """
        self.embedding_model = embedding_model or SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = embedding_dim
        
        self.vector_store = FaissVectorStore(embedding_dim)
        self.metadata_store = MetadataStore(metadata_store_path)
        
        self.vector_store_path = vector_store_path
        self.metadata_store_path = metadata_store_path
        
        # Add entity-predicate index for efficient lookup
        self.entity_index: Dict[Tuple[str, str], List[str]] = {}
        
        # Load if paths exist
        if vector_store_path and Path(vector_store_path).exists():
            self.vector_store.load(vector_store_path)
        if metadata_store_path and Path(metadata_store_path).exists():
            self.metadata_store.load(metadata_store_path)
            # Rebuild entity index from loaded metadata
            self._rebuild_entity_index()
    
    def _rebuild_entity_index(self):
        """Rebuild entity index from metadata store."""
        self.entity_index.clear()
        for evidence_id, meta in self.metadata_store.metadata.items():
            key = (meta.entity_id, meta.predicate)
            if key not in self.entity_index:
                self.entity_index[key] = []
            self.entity_index[key].append(evidence_id)

    def add_evidence(
        self,
        evidence_id: str,
        entity_id: str,
        predicate: str,
        text_content: Optional[str] = None,
        structured_data: Optional[Dict[str, Any]] = None,
        credibility: float = 1.0,
        timestamp: Optional[str] = None,
        evidence_type: str = "text",
        source: str = "",
        embedding: Optional[np.ndarray] = None
    ) -> EvidenceMetadata:
        """
        Add evidence to the index.
        
        Args:
            evidence_id: Unique evidence identifier
            entity_id: Entity this evidence relates to
            predicate: Predicate this evidence supports
            text_content: Text content of evidence
            structured_data: Structured data (if applicable)
            credibility: Credibility score [0, 1]
            timestamp: ISO timestamp
            evidence_type: Type of evidence
            source: Source of evidence
            embedding: Pre-computed embedding (if None, will compute from text_content)
            
        Returns:
            EvidenceMetadata object
        """
        # Compute embedding if not provided
        if embedding is None:
            if text_content is None:
                raise ValueError("Either text_content or embedding must be provided")
            # Convert tensor to numpy array if needed
            embedding_result = self.embedding_model.encode(text_content)
            embedding = np.array(embedding_result) if not isinstance(embedding_result, np.ndarray) else embedding_result
        
        # Ensure embedding is numpy array (not tensor)
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)
        
        # Add to vector store
        embedding_ids = self.vector_store.add(embedding)
        embedding_id = embedding_ids[0]
        
        # Create metadata
        metadata = EvidenceMetadata(
            evidence_id=evidence_id,
            entity_id=entity_id,
            predicate=predicate,
            text_content=text_content,
            structured_data=structured_data,
            credibility=credibility,
            timestamp=timestamp,
            evidence_type=evidence_type,
            source=source,
            embedding_id=embedding_id
        )
        
        # Add to metadata store
        self.metadata_store.add(metadata)

        # Add to entity index
        key = (entity_id, predicate)
        if key not in self.entity_index:
            self.entity_index[key] = []
        self.entity_index[key].append(evidence_id)
        
        return metadata
    
    def search(
        self,
        entity_id: str,
        predicate: str,
        top_k: int = 10,
        query_text: Optional[str] = None
    ) -> List[str]:
        """
        Search for relevant evidence.
        
        Args:
            entity_id: Entity to search for
            predicate: Predicate to search for
            top_k: Number of results to return
            query_text: Optional query text for semantic search
            
        Returns:
            List of evidence_ids
        """
        # NEW: Use entity index for exact entity+predicate lookup
        key = (entity_id, predicate)
        
        if key in self.entity_index:
            candidate_ids = self.entity_index[key]
            
            # If no semantic query or few results, just return them
            if query_text is None or len(candidate_ids) <= top_k:
                return candidate_ids[:top_k]
            
            # Optional: Rank by semantic similarity if query_text provided
            embeddings = []
            valid_ids = []
            
            for eid in candidate_ids:
                meta = self.metadata_store.get(eid)
                if meta and meta.embedding_id is not None:
                    try:
                        emb = np.zeros(self.embedding_dim, dtype='float32')
                        self.vector_store.index.reconstruct(int(meta.embedding_id), emb)
                        embeddings.append(emb)
                        valid_ids.append(eid)
                    except:
                        continue
            
            if embeddings and query_text:
                # Compute query embedding
                query_embedding_result = self.embedding_model.encode(query_text)
                query_embedding = np.array(query_embedding_result) if not isinstance(query_embedding_result, np.ndarray) else query_embedding_result
                
                # Compute similarities
                embeddings_array = np.array(embeddings)
                similarities = np.dot(embeddings_array, query_embedding)
                
                # Sort by similarity
                sorted_indices = np.argsort(similarities)[::-1][:top_k]
                return [valid_ids[i] for i in sorted_indices]
            
            return valid_ids[:top_k]
        
        # Fallback: No evidence found for this entity+predicate
        return []
    
    def fetch_meta(self, evidence_id: str) -> Optional[EvidenceMetadata]:
        """Fetch metadata for an evidence item."""
        return self.metadata_store.get(evidence_id)
    
    def fetch_raw(self, evidence_id: str) -> Optional[str]:
        """Fetch raw text content for an evidence item."""
        meta = self.metadata_store.get(evidence_id)
        if meta is None:
            return None
        return meta.text_content
    
    def save(self):
        """Save index and metadata to disk."""
        if self.vector_store_path:
            self.vector_store.save(self.vector_store_path)
        if self.metadata_store_path:
            self.metadata_store.save(self.metadata_store_path)
    
    def __len__(self) -> int:
        return len(self.metadata_store)
    
    def __repr__(self) -> str:
        return (f"EvidenceIndex(evidence_count={len(self.metadata_store)}, "
                f"embedding_dim={self.embedding_dim})")