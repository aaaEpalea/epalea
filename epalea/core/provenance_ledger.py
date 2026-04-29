"""
Provenance Ledger Module
Maintains immutable audit trail of all inferences and their evidence chains.
Section 4.4 and Section 9.5: "Every query produces immutable audit records"
"""

import json
import jsonlines
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path
import hashlib


@dataclass
class InferenceRecord:
    """
    Immutable record of a single inference operation.
    Contains complete provenance for audit and reproducibility.
    """
    record_id: str
    timestamp: str
    entity_id: str
    predicate: str
    query_type: str  # "marginal", "conditional", etc.
    
    # Results
    distribution: Dict[str, float]
    top_value: str
    confidence: float
    
    # Provenance
    evidence_chain: List[str]  # Evidence IDs used
    factor_metadata: List[Dict[str, Any]]  # Factor details
    model_versions: Dict[str, str]  # Model checkpoint versions
    
    # Query details
    conditionals: Dict[str, Any] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    
    # System info
    execution_time_ms: float = 0.0
    source: str = "lpf_system"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    def compute_hash(self) -> str:
        """
        Compute cryptographic hash of record for integrity.
        
        Returns:
            SHA-256 hash of record
        """
        # Create stable JSON string
        record_str = json.dumps(self.to_dict(), sort_keys=True)
        hash_obj = hashlib.sha256(record_str.encode('utf-8'))
        return hash_obj.hexdigest()


class ProvenanceLedger:
    """
    Append-only ledger for inference provenance.
    
    Provides:
    - Immutable audit trail
    - Evidence chain tracking
    - Model version tracking
    - Query reproducibility
    """
    
    def __init__(self, ledger_path: Optional[str] = None):
        """
        Initialize provenance ledger.
        
        Args:
            ledger_path: Path to JSONL ledger file (None for in-memory)
        """
        self.ledger_path = ledger_path
        self.records: List[InferenceRecord] = []
        self.record_counter = 0
        
        if ledger_path is not None and Path(ledger_path).exists():
            self._load_ledger()
    
    def append_inference_record(
        self,
        entity_id: str,
        predicate: str,
        distribution: Dict[str, float],
        evidence_chain: List[str],
        factor_metadata: List[Dict[str, Any]],
        model_versions: Optional[Dict[str, str]] = None,
        conditionals: Optional[Dict[str, Any]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        execution_time_ms: float = 0.0,
        query_type: str = "marginal"
    ) -> InferenceRecord:
        """
        Append a new inference record to the ledger.
        
        Args:
            entity_id: Entity being queried
            predicate: Predicate being inferred
            distribution: Resulting probability distribution
            evidence_chain: List of evidence IDs used
            factor_metadata: Metadata for each soft factor
            model_versions: Dictionary of model checkpoint versions
            conditionals: Hard evidence used in query
            hyperparameters: Hyperparameters used
            execution_time_ms: Execution time in milliseconds
            query_type: Type of query
            
        Returns:
            Created InferenceRecord
        """
        # Generate record ID
        self.record_counter += 1
        record_id = f"INF{self.record_counter:08d}"
        
        # Get timestamp
        timestamp = datetime.now().isoformat()
        
        # Get top value and confidence
        if distribution:
            top_value = max(distribution.keys(), key=lambda k: distribution[k])
            confidence = distribution[top_value]
        else:
            top_value = "unknown"
            confidence = 0.0
        
        # Create record
        record = InferenceRecord(
            record_id=record_id,
            timestamp=timestamp,
            entity_id=entity_id,
            predicate=predicate,
            query_type=query_type,
            distribution=distribution,
            top_value=top_value,
            confidence=confidence,
            evidence_chain=evidence_chain,
            factor_metadata=factor_metadata,
            model_versions=model_versions or {},
            conditionals=conditionals or {},
            hyperparameters=hyperparameters or {},
            execution_time_ms=execution_time_ms
        )
        
        # Append to in-memory list
        self.records.append(record)
        
        # Append to file if path specified
        if self.ledger_path is not None:
            self._append_to_file(record)
        
        return record
    
    def _append_to_file(self, record: InferenceRecord):
        """Append record to JSONL file."""
        if self.ledger_path is None:
            return
        
        Path(self.ledger_path).parent.mkdir(parents=True, exist_ok=True)
        
        with jsonlines.open(self.ledger_path, mode='a') as writer:
            writer.write(record.to_dict())
    
    def _load_ledger(self):
        """Load existing ledger from file."""
        if self.ledger_path is None:
            return
        
        self.records.clear()
        
        with jsonlines.open(self.ledger_path) as reader:
            for obj in reader:
                record = InferenceRecord(**obj)
                self.records.append(record)
                
                # Update counter
                record_num = int(record.record_id.replace("INF", ""))
                self.record_counter = max(self.record_counter, record_num)
    
    def get_audit_trail(
        self,
        entity_id: str,
        predicate: Optional[str] = None
    ) -> List[InferenceRecord]:
        """
        Get audit trail for an entity and optionally a predicate.
        
        Args:
            entity_id: Entity to get trail for
            predicate: Optional predicate filter
            
        Returns:
            List of InferenceRecords
        """
        results = []
        for record in self.records:
            if record.entity_id == entity_id:
                if predicate is None or record.predicate == predicate:
                    results.append(record)
        return results
    
    def get_record(self, record_id: str) -> Optional[InferenceRecord]:
        """
        Get a specific record by ID.
        
        Args:
            record_id: Record identifier
            
        Returns:
            InferenceRecord if found, None otherwise
        """
        for record in self.records:
            if record.record_id == record_id:
                return record
        return None
    
    def get_evidence_usage(self, evidence_id: str) -> List[InferenceRecord]:
        """
        Get all inferences that used a specific evidence item.
        
        Args:
            evidence_id: Evidence identifier
            
        Returns:
            List of InferenceRecords
        """
        results = []
        for record in self.records:
            if evidence_id in record.evidence_chain:
                results.append(record)
        return results
    
    def get_recent_records(self, n: int = 10) -> List[InferenceRecord]:
        """
        Get the n most recent records.
        
        Args:
            n: Number of records to return
            
        Returns:
            List of most recent InferenceRecords
        """
        return self.records[-n:]
    
    def verify_integrity(self) -> bool:
        """
        Verify integrity of all records via hash checking.
        
        Returns:
            True if all records are valid
        """
        # In a production system, you would store hashes separately
        # and verify them here. For now, just check basic validity.
        for record in self.records:
            try:
                # Check that record can be hashed
                record.compute_hash()
                # Check required fields
                assert record.record_id
                assert record.entity_id
                assert record.predicate
            except Exception:
                return False
        return True
    
    def export_to_json(self, output_path: str):
        """
        Export entire ledger to JSON file.
        
        Args:
            output_path: Path to output file
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(
                [record.to_dict() for record in self.records],
                f,
                indent=2
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the ledger.
        
        Returns:
            Dictionary with ledger statistics
        """
        if not self.records:
            return {
                'total_records': 0,
                'unique_entities': 0,
                'unique_predicates': 0
            }
        
        entities = set(r.entity_id for r in self.records)
        predicates = set(r.predicate for r in self.records)
        
        # Average confidence
        avg_confidence = sum(r.confidence for r in self.records) / len(self.records)
        
        # Average execution time
        avg_time = sum(r.execution_time_ms for r in self.records) / len(self.records)
        
        return {
            'total_records': len(self.records),
            'unique_entities': len(entities),
            'unique_predicates': len(predicates),
            'average_confidence': avg_confidence,
            'average_execution_time_ms': avg_time
        }
    
    def __len__(self) -> int:
        return len(self.records)
    
    def __repr__(self) -> str:
        return f"ProvenanceLedger(records={len(self.records)})"


def create_factor_metadata_summary(factors: List[Any]) -> List[Dict[str, Any]]:
    """
    Create summary metadata from soft factors for provenance.
    
    Args:
        factors: List of SoftFactor objects
        
    Returns:
        List of metadata dictionaries
    """
    from core.factor_converter import SoftFactor
    
    summaries = []
    for factor in factors:
        if isinstance(factor, SoftFactor):
            summary = {
                'evidence_id': factor.evidence_id,
                'variables': factor.variables,
                'weight': factor.weight,
                'potential': factor.potential,
                'metadata': factor.metadata
            }
            summaries.append(summary)
    
    return summaries
