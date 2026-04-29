"""
Generic LPF Training Module
Trains encoder + conditional decoder for ANY domain.

This is a domain-agnostic trainer that can work with:
  - Compliance data (low/medium/high)
  - Quantum tomography (|0⟩, |1⟩, |+⟩, |-⟩, |i⟩, |-i⟩, mixed)
  - Healthcare data (healthy/at-risk/critical)
  - Any other categorical prediction task!

Key improvement: Schema is passed as argument, making this fully reusable.

USAGE:
  python train_generic_lpf.py \
    --domain quantum_1q \
    --data-dir ./data/quantum_1q \
    --checkpoint-dir ./checkpoints/quantum_1q \
    --predicate quantum_state_type \
    --domain-values "|0⟩" "|1⟩" "|+⟩" "|-⟩" "|i⟩" "|-i⟩" "mixed" \
    --n-seeds 7
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from epalea.models.vae_encoder import VAEEncoderNetwork, compute_kl_divergence, reparameterize
from epalea.models.decoder_network import DecoderNetwork
from epalea.models.schema import Schema
from epalea.core.evidence_index import EvidenceIndex


class LPFDataset(Dataset):
    """
    Dataset for LPF training: embeddings + predicate labels.
    Domain-agnostic!
    """
    
    def __init__(
        self,
        embeddings: np.ndarray,
        predicates: List[str],
        labels: List[str],
        schema: Schema
    ):
        """
        Initialize dataset.
        
        Args:
            embeddings: Evidence embeddings (n_samples, embedding_dim)
            predicates: Predicate names for each sample
            labels: Ground truth values (domain-specific)
            schema: Schema for encoding labels
        """
        self.embeddings = torch.FloatTensor(embeddings)
        self.predicates = predicates
        self.labels = labels.tolist() if isinstance(labels, np.ndarray) else labels
        
        # Convert labels to indices
        self.label_indices = []
        for pred, label in zip(predicates, self.labels):
            domain = schema.get_predicate_domain(pred)
            if label not in domain:
                raise ValueError(f"Label '{label}' not in domain for predicate '{pred}'")
            idx = domain.index(label)
            self.label_indices.append(idx)
        self.label_indices = torch.LongTensor(self.label_indices)
    
    def __len__(self) -> int:
        return len(self.embeddings)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, torch.Tensor]:
        return self.embeddings[idx], self.predicates[idx], self.label_indices[idx]


class LPFModel(nn.Module):
    """
    Complete LPF model: Encoder + Conditional Decoder.
    Domain-agnostic!
    """
    
    def __init__(
        self,
        encoder: VAEEncoderNetwork,
        decoder: DecoderNetwork
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(
        self,
        embeddings: torch.Tensor,
        predicates: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            embeddings: Input embeddings (batch_size, embedding_dim)
            predicates: List of predicate names
            
        Returns:
            Tuple of (logits, mu, sigma)
        """
        # Encode to latent posterior
        mu, sigma = self.encoder(embeddings)
        
        # Sample latent code
        z = reparameterize(mu, sigma)
        
        # Decode to predicate distribution
        predicate = predicates[0]  # Assume batch has same predicate
        logits = self.decoder(z, predicate)
        
        return logits, mu, sigma


def compute_lpf_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    kl_weight: float = 0.01
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute LPF training loss.
    
    Loss = CrossEntropy(logits, labels) + β·KL(q(z|e) || p(z))
    """
    # Classification loss
    ce_loss = F.cross_entropy(logits, labels)
    
    # KL divergence
    kl_loss = compute_kl_divergence(mu, sigma).mean()
    
    # Total loss
    total_loss = ce_loss + kl_weight * kl_loss
    
    # Accuracy
    preds = torch.argmax(logits, dim=1)
    accuracy = (preds == labels).float().mean()
    
    return total_loss, {
        'total': total_loss.item(),
        'cross_entropy': ce_loss.item(),
        'kl': kl_loss.item(),
        'accuracy': accuracy.item()
    }


class LPFTrainer:
    """
    Generic trainer for LPF models.
    Works with any domain!
    """
    
    def __init__(
        self,
        model: LPFModel,
        learning_rate: float = 1e-3,
        kl_weight: float = 0.01,
        device: str = "cpu"
    ):
        """
        Initialize trainer.
        
        Args:
            model: LPFModel (encoder + decoder)
            learning_rate: Learning rate
            kl_weight: KL regularization weight (β)
            device: Device to train on
        """
        self.model = model.to(device)
        self.device = device
        self.kl_weight = kl_weight
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )
        
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_losses = {
            'total': 0.0,
            'cross_entropy': 0.0,
            'kl': 0.0,
            'accuracy': 0.0
        }
        
        for embeddings, predicates, labels in train_loader:
            embeddings = embeddings.to(self.device)
            labels = labels.to(self.device)
            
            # Forward
            logits, mu, sigma = self.model(embeddings, predicates)
            
            # Loss
            loss, loss_dict = compute_lpf_loss(
                logits, labels, mu, sigma, self.kl_weight
            )
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Accumulate
            for key, val in loss_dict.items():
                epoch_losses[key] += val
        
        # Average
        n_batches = len(train_loader)
        epoch_losses = {k: v / n_batches for k, v in epoch_losses.items()}
        
        self.train_losses.append(epoch_losses)
        return epoch_losses
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate on validation set."""
        self.model.eval()
        
        epoch_losses = {
            'total': 0.0,
            'cross_entropy': 0.0,
            'kl': 0.0,
            'accuracy': 0.0
        }
        
        with torch.no_grad():
            for embeddings, predicates, labels in val_loader:
                embeddings = embeddings.to(self.device)
                labels = labels.to(self.device)
                
                # Forward
                logits, mu, sigma = self.model(embeddings, predicates)
                
                # Loss
                loss, loss_dict = compute_lpf_loss(
                    logits, labels, mu, sigma, self.kl_weight
                )
                
                # Accumulate
                for key, val in loss_dict.items():
                    epoch_losses[key] += val
        
        # Average
        n_batches = len(val_loader)
        epoch_losses = {k: v / n_batches for k, v in epoch_losses.items()}
        
        self.val_losses.append(epoch_losses)
        return epoch_losses
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        checkpoint_dir: Optional[str] = None
    ) -> Dict[str, List[Dict[str, float]]]:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            early_stopping_patience: Patience for early stopping
            checkpoint_dir: Directory to save checkpoints
            
        Returns:
            Training history
        """
        if checkpoint_dir:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print("=" * 70)
        print("Training LPF Model (Encoder + Decoder)")
        print("=" * 70)
        
        for epoch in range(epochs):
            # Train
            train_losses = self.train_epoch(train_loader)
            
            # Validate
            if val_loader:
                val_losses = self.validate(val_loader)
                
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Train - Loss: {train_losses['total']:.4f}, "
                      f"CE: {train_losses['cross_entropy']:.4f}, "
                      f"KL: {train_losses['kl']:.4f}, "
                      f"Acc: {train_losses['accuracy']:.2%}")
                print(f"  Val   - Loss: {val_losses['total']:.4f}, "
                      f"CE: {val_losses['cross_entropy']:.4f}, "
                      f"KL: {val_losses['kl']:.4f}, "
                      f"Acc: {val_losses['accuracy']:.2%}")
                
                # Early stopping
                if val_losses['total'] < best_val_loss:
                    best_val_loss = val_losses['total']
                    patience_counter = 0
                    
                    if checkpoint_dir:
                        self.save(f"{checkpoint_dir}/best_model.pt")
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
            else:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Train - Loss: {train_losses['total']:.4f}, "
                      f"CE: {train_losses['cross_entropy']:.4f}, "
                      f"KL: {train_losses['kl']:.4f}, "
                      f"Acc: {train_losses['accuracy']:.2%}")
            
            # Save periodic checkpoint
            if checkpoint_dir and (epoch + 1) % 10 == 0:
                self.save(f"{checkpoint_dir}/model_epoch_{epoch+1}.pt")
        
        print("\n" + "=" * 70)
        print("✅ Training Complete!")
        print("=" * 70)
        if val_loader:
            print(f"Best val loss: {best_val_loss:.4f}")
            print(f"Final train accuracy: {self.train_losses[-1]['accuracy']:.2%}")
            print(f"Final val accuracy: {self.val_losses[-1]['accuracy']:.2%}")
        
        return {
            'train': self.train_losses,
            'val': self.val_losses
        }
    
    def save(self, path: str):
        """Save trained model."""
        torch.save({
            'encoder_state_dict': self.model.encoder.state_dict(),
            'decoder_state_dict': self.model.decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'kl_weight': self.kl_weight
        }, path)
        print(f"✓ Model saved to {path}")
    
    def load(self, path: str):
        """Load trained model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.model.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.kl_weight = checkpoint.get('kl_weight', 0.01)
        print(f"✓ Model loaded from {path}")


def prepare_training_data_from_files(
    data_dir: str,
    evidence_index: EvidenceIndex,
    schema: Schema,
    predicate: str,
    label_key: str,
    split: str = "train"
) -> LPFDataset:
    """
    Prepare training data from JSON files.
    GENERIC version that works with any domain!
    
    Args:
        data_dir: Directory with data files
        evidence_index: Evidence index with embeddings
        schema: Schema for label encoding
        predicate: Predicate to train on
        label_key: Key for label in company/entity dict (e.g., 'compliance_level', 'quantum_state_type')
        split: Data split to load
        
    Returns:
        LPFDataset ready for training
    """
    # Load entities (companies/systems/patients/etc.)
    entities_file = f"{data_dir}/{split}_companies.json"
    with open(entities_file) as f:
        entities = json.load(f)
    
    # Create entity_id -> label mapping
    entity_labels = {
        e['company_id']: e[label_key]
        for e in entities
    }
    
    # Load evidence
    evidence_file = f"{data_dir}/{split}_evidence.json"
    with open(evidence_file) as f:
        evidence_items = json.load(f)
    
    # Extract embeddings and labels
    embeddings = []
    predicates_list = []
    labels = []
    
    print(f"Loading {split} data from {data_dir}...")
    
    for item in evidence_items:
        entity_id = item['company_id']
        if entity_id not in entity_labels:
            continue
        
        # Get embedding from evidence index
        meta = evidence_index.fetch_meta(item['evidence_id'])
        if meta is None or meta.embedding_id is None:
            continue
        
        try:
            # FAISS reconstruct
            emb = np.zeros(evidence_index.embedding_dim, dtype='float32')
            evidence_index.vector_store.index.reconstruct(
                int(meta.embedding_id),
                emb
            )
            embeddings.append(emb)
            predicates_list.append(predicate)
            labels.append(entity_labels[entity_id])
        except Exception:
            continue
    
    if len(embeddings) == 0:
        raise ValueError(f"No valid embeddings found in {split} data!")
    
    embeddings_array = np.array(embeddings)
    
    print(f"✓ Prepared {len(embeddings)} {split} samples")
    print(f"  Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
    
    dataset = LPFDataset(
        embeddings_array,
        predicates_list,
        labels,
        schema
    )
    
    return dataset


def load_and_index_data(
    data_dir: str,
    evidence_index: EvidenceIndex,
    predicate: str,
    split: str = "train"
) -> int:
    """
    Load evidence into the evidence index.
    GENERIC version!
    """
    evidence_file = f"{data_dir}/{split}_evidence.json"
    
    print(f"Indexing {split} evidence from {evidence_file}...")
    
    with open(evidence_file) as f:
        evidence_items = json.load(f)
    
    count = 0
    failed = 0
    for item in evidence_items:
        text_content = item.get('text_content', f"Evidence for {item['company_id']}")
        
        try:
            evidence_index.add_evidence(
                evidence_id=item['evidence_id'],
                entity_id=item['company_id'],
                predicate=predicate,
                text_content=text_content,
                structured_data=item.get('structured_data'),
                credibility=item.get('credibility', 1.0),
                timestamp=item.get('timestamp'),
                evidence_type=item.get('evidence_type', 'text'),
                source=f"{split}"
            )
            count += 1
        except Exception as e:
            failed += 1
    
    print(f"✓ Indexed {count} evidence items ({failed} failed)")
    return count


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_schema_from_args(
    variable_name: str,
    predicate_name: str,
    domain_values: List[str]
) -> Schema:
    """
    Create schema from command-line arguments.
    
    Args:
        variable_name: Variable name (e.g., 'compliance', 'quantum_state')
        predicate_name: Predicate name (e.g., 'compliance_level', 'quantum_state_type')
        domain_values: List of possible values
        
    Returns:
        Schema object
    """
    schema = Schema()
    schema.add_variable(variable_name, domain_values)
    schema.add_predicate(predicate_name, [variable_name], domain_values)
    return schema


def train_with_seed_search(
    domain: str,
    data_dir: str,
    checkpoint_dir: str,
    results_dir: str,
    schema: Schema,
    predicate: str,
    label_key: str,
    n_seeds: int = 7,
    embedding_dim: int = 384,
    latent_dim: int = 64
):
    """
    Train LPF models with multiple seeds and save best.
    GENERIC VERSION with optimized evidence index loading!
    """
    import csv
    import shutil
    
    print("\n" + "="*70)
    print(f"🔍 SEED SEARCH MODE: Testing {n_seeds} seeds for {domain.upper()}")
    print("="*70)
    
    # Create results directory
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    # Seeds to try
    seeds = [42, 123, 456, 789, 2024, 2025, 314159][:n_seeds]
    
    # ============================================================
    # INITIALIZE EVIDENCE INDEX ONCE (OUTSIDE SEED LOOP)
    # ============================================================
    print("\n📊 Setting up evidence index (one-time)...")
    
    vector_store_path = Path(data_dir) / "vector_store.faiss"
    metadata_store_path = Path(data_dir) / "metadata.jsonl"
    
    evidence_index = EvidenceIndex(
        embedding_dim=embedding_dim,
        vector_store_path=str(vector_store_path),
        metadata_store_path=str(metadata_store_path)
    )
    
    # Check if we need to index or reindex
    needs_indexing = False
    
    if len(evidence_index) == 0:
        print("  ⚠ Evidence index is empty!")
        needs_indexing = True
    else:
        print(f"  ✓ Loaded existing index: {len(evidence_index)} items")
        
        # ============================================================
        # VERIFICATION: Check train and val data coverage
        # ============================================================
        print("\n  [VERIFICATION] Checking evidence coverage...")
        
        # Load sample evidence files to get company IDs
        train_evidence_file = Path(data_dir) / "train_evidence.json"
        val_evidence_file = Path(data_dir) / "val_evidence.json"
        
        train_missing = False
        val_missing = False
        
        # Check train data
        if train_evidence_file.exists():
            with open(train_evidence_file) as f:
                train_evidence_data = json.load(f)
            
            if train_evidence_data:
                sample_train_id = train_evidence_data[0]['company_id']
                train_evidence_results = evidence_index.search(
                    sample_train_id, predicate, top_k=6
                )
                print(f"  Sample train company {sample_train_id}: {len(train_evidence_results)} evidence items")
                
                if len(train_evidence_results) == 0:
                    print("  ⚠ WARNING: No evidence found for training companies!")
                    train_missing = True
        
        # Check val data
        if val_evidence_file.exists():
            with open(val_evidence_file) as f:
                val_evidence_data = json.load(f)
            
            if val_evidence_data:
                sample_val_id = val_evidence_data[0]['company_id']
                val_evidence_results = evidence_index.search(
                    sample_val_id, predicate, top_k=6
                )
                print(f"  Sample val company {sample_val_id}: {len(val_evidence_results)} evidence items")
                
                if len(val_evidence_results) == 0:
                    print("  ⚠ WARNING: No evidence found for validation companies!")
                    val_missing = True
        
        if train_missing or val_missing:
            print("\n  ⚠ Evidence index incomplete, reindexing...")
            needs_indexing = True
    
    # ============================================================
    # INDEX DATA IF NEEDED
    # ============================================================
    if needs_indexing:
        print("\n  📥 Indexing evidence data...")
        
        # Clear existing index if it has partial data
        if len(evidence_index) > 0:
            print("  Clearing existing index...")
            # Reinitialize fresh index (will clear files on disk)
            evidence_index = EvidenceIndex(
                embedding_dim=embedding_dim,
                vector_store_path=str(vector_store_path),
                metadata_store_path=str(metadata_store_path)
            )
        
        # Index train data
        print("  Indexing training data...")
        train_count = load_and_index_data(data_dir, evidence_index, predicate, split="train")
        
        # Index val data
        print("  Indexing validation data...")
        val_count = load_and_index_data(data_dir, evidence_index, predicate, split="val")
        
        # CRITICAL: Rebuild entity index after adding all evidence
        print("  Rebuilding entity index...")
        evidence_index._rebuild_entity_index()
        
        # Save
        evidence_index.save()
        print(f"  ✓ Indexed {train_count} train + {val_count} val = {len(evidence_index)} total items")
    else:
        print("  ✓ Evidence index verified and ready!")
    
    print(f"  Entity index has {len(evidence_index.entity_index)} keys")
    
    # ============================================================
    # SEED SEARCH LOOP
    # ============================================================
    
    best_seed = None
    best_val_acc = 0.0
    best_val_loss = float('inf')
    all_results = []
    
    for i, seed in enumerate(seeds, 1):
        print("\n" + "="*70)
        print(f"Training with SEED {i}/{n_seeds}: {seed}")
        print("="*70)
        
        # Set seed
        set_seed(seed)
        
        # Prepare datasets from existing index (REUSE!)
        print(f"\n📊 Preparing datasets with seed {seed}...")
        train_dataset = prepare_training_data_from_files(
            data_dir, evidence_index, schema, 
            predicate=predicate, label_key=label_key, split="train"
        )
        val_dataset = prepare_training_data_from_files(
            data_dir, evidence_index, schema, 
            predicate=predicate, label_key=label_key, split="val"
        )
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64)
        
        # Create models
        encoder = VAEEncoderNetwork(
            embedding_dim=embedding_dim,
            latent_dim=latent_dim,
            hidden_dims=[256, 128]
        )
        
        decoder = DecoderNetwork(
            latent_dim=latent_dim,
            schema=schema
        )
        
        model = LPFModel(encoder, decoder)
        
        # Create trainer
        trainer = LPFTrainer(
            model=model,
            learning_rate=1e-3,
            kl_weight=0.01,
            device="cpu"
        )
        
        # Train
        print(f"\n🚀 Training with seed {seed}...")
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=50,
            early_stopping_patience=15,
            checkpoint_dir=f"{checkpoint_dir}/seed_{seed}"
        )
        
        # Extract metrics
        final_train = history['train'][-1]
        final_val = history['val'][-1]
        
        best_epoch_val_loss = min(h['total'] for h in history['val'])
        best_epoch_val_acc = max(h['accuracy'] for h in history['val'])
        
        result = {
            'seed': seed,
            'final_train_loss': final_train['total'],
            'final_train_acc': final_train['accuracy'],
            'final_val_loss': final_val['total'],
            'final_val_acc': final_val['accuracy'],
            'best_val_loss': best_epoch_val_loss,
            'best_val_acc': best_epoch_val_acc,
            'n_epochs': len(history['train'])
        }
        all_results.append(result)
        
        print(f"\n📊 SEED {seed} Results:")
        print(f"   Final Val Acc: {final_val['accuracy']:.4f}")
        print(f"   Best Val Acc: {best_epoch_val_acc:.4f}")
        
        # Track best seed
        if best_epoch_val_acc > best_val_acc:
            best_val_acc = best_epoch_val_acc
            best_val_loss = best_epoch_val_loss
            best_seed = seed
            print(f"   🌟 NEW BEST SEED!")
            
            # Copy best model
            src = f"{checkpoint_dir}/seed_{seed}/best_model.pt"
            dst = f"{checkpoint_dir}/best_model.pt"
            if Path(src).exists():
                shutil.copy(src, dst)
                print(f"   ✓ Copied to {dst}")
    
    # Statistics
    val_accs = [r['best_val_acc'] for r in all_results]
    mean_val_acc = np.mean(val_accs)
    std_val_acc = np.std(val_accs)
    
    # Print summary
    print("\n" + "="*70)
    print(f"TRAINING SEED SEARCH SUMMARY ({domain.upper()})")
    print("="*70)
    print(f"{'Seed':<10} {'Val Acc':<12} {'Val Loss':<12} {'Epochs':<8}")
    print("-"*70)
    for r in all_results:
        marker = "🌟" if r['seed'] == best_seed else "  "
        print(f"{marker} {r['seed']:<10} {r['best_val_acc']:<12.4f} "
              f"{r['best_val_loss']:<12.4f} {r['n_epochs']:<8}")
    
    print(f"\nBest seed: {best_seed} with val accuracy: {best_val_acc*100:.2f}%")
    print(f"Mean ± Std: {mean_val_acc*100:.1f}±{std_val_acc*100:.1f}%")
    
    # Save results
    results_file = f"{results_dir}/training_seed_search.json"
    with open(results_file, 'w') as f:
        json.dump({
            'domain': domain,
            'all_results': all_results,
            'best_seed': best_seed,
            'statistics': {
                'mean': mean_val_acc,
                'std': std_val_acc,
                'best': best_val_acc
            }
        }, f, indent=2)
    
    print(f"\n✓ Results saved to {results_file}")
    print(f"✓ Best model saved to {checkpoint_dir}/best_model.pt")


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="Train LPF models (generic, domain-agnostic)")
    
    parser.add_argument("--domain", type=str, required=True,
                       help="Domain name (e.g., 'quantum_1q', 'compliance', 'healthcare')")
    parser.add_argument("--data-dir", type=str, required=True,
                       help="Data directory")
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                       help="Checkpoint directory")
    parser.add_argument("--predicate", type=str, required=True,
                       help="Predicate name (e.g., 'quantum_state_type', 'compliance_level')")
    parser.add_argument("--label-key", type=str, default=None,
                       help="Key for label in entity dict (defaults to predicate name)")
    parser.add_argument("--variable-name", type=str, default=None,
                       help="Schema variable name (defaults to domain)")
    parser.add_argument("--domain-values", nargs='+', required=True,
                       help="Domain values (e.g., '|0⟩' '|1⟩' '|+⟩' for quantum)")
    parser.add_argument("--n-seeds", type=int, default=7,
                       help="Number of seeds to try")
    parser.add_argument("--embedding-dim", type=int, default=384,
                       help="Embedding dimension")
    parser.add_argument("--latent-dim", type=int, default=64,
                       help="Latent dimension")
    parser.add_argument("--single-seed", type=int, default=None,
                       help="Train with single fixed seed (skips search)")
    
    args = parser.parse_args()
    
    # Defaults
    label_key = args.label_key or args.predicate
    variable_name = args.variable_name or args.domain
    results_dir = f"./lpf_training_results/{args.domain}"
    
    # Create schema
    print(f"\n📋 Creating schema for {args.domain}...")
    print(f"   Variable: {variable_name}")
    print(f"   Predicate: {args.predicate}")
    print(f"   Domain: {args.domain_values}")
    
    schema = create_schema_from_args(
        variable_name=variable_name,
        predicate_name=args.predicate,
        domain_values=args.domain_values
    )
    
    if args.single_seed is not None:
        # ========================================================================
        # SINGLE SEED TRAINING MODE - WITH INDEX REUSE (FIXED VERSION!)
        # ========================================================================
        print("=" * 70)
        print(f"Training with FIXED SEED: {args.single_seed}")
        print("=" * 70)
        
        set_seed(args.single_seed)
        
        # ============================================================
        # INITIALIZE EVIDENCE INDEX (REUSE IF EXISTS)
        # ============================================================
        print("\n📊 Setting up evidence index...")
        
        vector_store_path = Path(args.data_dir) / "vector_store.faiss"
        metadata_store_path = Path(args.data_dir) / "metadata.jsonl"
        
        evidence_index = EvidenceIndex(
            embedding_dim=args.embedding_dim,
            vector_store_path=str(vector_store_path),
            metadata_store_path=str(metadata_store_path)
        )
        
        # Check if we need to index or reindex
        needs_indexing = False
        
        if len(evidence_index) == 0:
            print("  ⚠ Evidence index is empty!")
            needs_indexing = True
        else:
            print(f"  ✓ Loaded existing index: {len(evidence_index)} items")
            
            # Verify evidence coverage
            print("\n  [VERIFICATION] Checking evidence coverage...")
            
            train_evidence_file = Path(args.data_dir) / "train_evidence.json"
            val_evidence_file = Path(args.data_dir) / "val_evidence.json"
            
            train_missing = False
            val_missing = False
            
            # Check train data
            if train_evidence_file.exists():
                with open(train_evidence_file) as f:
                    train_evidence_data = json.load(f)
                
                if train_evidence_data:
                    sample_train_id = train_evidence_data[0]['company_id']
                    train_evidence_results = evidence_index.search(
                        sample_train_id, args.predicate, top_k=6
                    )
                    print(f"  Sample train company {sample_train_id}: {len(train_evidence_results)} evidence items")
                    
                    if len(train_evidence_results) == 0:
                        print("  ⚠ WARNING: No evidence found for training companies!")
                        train_missing = True
            
            # Check val data
            if val_evidence_file.exists():
                with open(val_evidence_file) as f:
                    val_evidence_data = json.load(f)
                
                if val_evidence_data:
                    sample_val_id = val_evidence_data[0]['company_id']
                    val_evidence_results = evidence_index.search(
                        sample_val_id, args.predicate, top_k=6
                    )
                    print(f"  Sample val company {sample_val_id}: {len(val_evidence_results)} evidence items")
                    
                    if len(val_evidence_results) == 0:
                        print("  ⚠ WARNING: No evidence found for validation companies!")
                        val_missing = True
            
            if train_missing or val_missing:
                print("\n  ⚠ Evidence index incomplete, reindexing...")
                needs_indexing = True
        
        # Index data if needed
        if needs_indexing:
            print("\n  📥 Indexing evidence data...")
            
            # Clear existing index if it has partial data
            if len(evidence_index) > 0:
                print("  Clearing existing index...")
                evidence_index = EvidenceIndex(
                    embedding_dim=args.embedding_dim,
                    vector_store_path=str(vector_store_path),
                    metadata_store_path=str(metadata_store_path)
                )
            
            # Index train and val data
            print("  Indexing training data...")
            train_count = load_and_index_data(args.data_dir, evidence_index, args.predicate, split="train")
            
            print("  Indexing validation data...")
            val_count = load_and_index_data(args.data_dir, evidence_index, args.predicate, split="val")
            
            # Rebuild entity index
            print("  Rebuilding entity index...")
            evidence_index._rebuild_entity_index()
            
            # Save
            evidence_index.save()
            print(f"  ✓ Indexed {train_count} train + {val_count} val = {len(evidence_index)} total items")
        else:
            print("  ✓ Evidence index verified and ready!")
        
        print(f"  Entity index has {len(evidence_index.entity_index)} keys")
        
        # ============================================================
        # PREPARE DATASETS AND TRAIN
        # ============================================================
        train_dataset = prepare_training_data_from_files(
            args.data_dir, evidence_index, schema,
            predicate=args.predicate, label_key=label_key, split="train"
        )
        val_dataset = prepare_training_data_from_files(
            args.data_dir, evidence_index, schema,
            predicate=args.predicate, label_key=label_key, split="val"
        )
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64)
        
        encoder = VAEEncoderNetwork(
            embedding_dim=args.embedding_dim,
            latent_dim=args.latent_dim,
            hidden_dims=[256, 128]
        )
        decoder = DecoderNetwork(latent_dim=args.latent_dim, schema=schema)
        model = LPFModel(encoder, decoder)
        
        trainer = LPFTrainer(model=model, learning_rate=1e-3, kl_weight=0.01)
        
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=50,
            early_stopping_patience=15,
            checkpoint_dir=args.checkpoint_dir
        )
        
        print("\n✅ Training complete!")
        print(f"Model saved to: {args.checkpoint_dir}/best_model.pt")
    
    else:
        # Seed search mode
        train_with_seed_search(
            domain=args.domain,
            data_dir=args.data_dir,
            checkpoint_dir=args.checkpoint_dir,
            results_dir=results_dir,
            schema=schema,
            predicate=args.predicate,
            label_key=label_key,
            n_seeds=args.n_seeds,
            embedding_dim=args.embedding_dim,
            latent_dim=args.latent_dim
        )


if __name__ == "__main__":
    main()
    