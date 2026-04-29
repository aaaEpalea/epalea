"""
Canonical Database Module
Fast key-value store for authoritative facts with staleness checking.
Implements the "canonical fast path" from Section 5.8.
"""

import json
import sqlite3
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class CanonicalRecord:
    """A canonical fact record."""
    entity_id: str
    predicate: str
    value: str
    confidence: float
    timestamp: str  # ISO format
    source: str = "canonical"
    metadata: Dict[str, Any] | None = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CanonicalRecord':
        return cls(**d)
    
    def age_days(self) -> float:
        """Calculate age of record in days."""
        timestamp_dt = datetime.fromisoformat(self.timestamp)
        now = datetime.now()
        delta = now - timestamp_dt
        return delta.total_seconds() / 86400.0


class CanonicalDB:
    """
    Fast canonical database for authoritative facts.
    Uses SQLite for persistence with in-memory caching.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize canonical database.
        
        Args:
            db_path: Path to SQLite database file. If None, uses in-memory DB.
        """
        if db_path is None:
            self.db_path = ":memory:"
        else:
            self.db_path = db_path
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_db()
        
        # In-memory cache for fast lookups
        self._cache: Dict[Tuple[str, str], CanonicalRecord] = {}
    
    def _init_db(self):
        """Initialize database schema."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS canonical_facts (
                entity_id TEXT NOT NULL,
                predicate TEXT NOT NULL,
                value TEXT NOT NULL,
                confidence REAL NOT NULL,
                timestamp TEXT NOT NULL,
                source TEXT NOT NULL,
                metadata TEXT,
                PRIMARY KEY (entity_id, predicate)
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_entity 
            ON canonical_facts(entity_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON canonical_facts(timestamp)
        """)
        self.conn.commit()
    
    def set(
        self,
        entity_id: str,
        predicate: str,
        value: str,
        confidence: float = 1.0,
        timestamp: Optional[str] = None,
        source: str = "canonical",
        metadata: Optional[Dict[str, Any]] = None
    ) -> CanonicalRecord:
        """
        Set a canonical fact.
        
        Args:
            entity_id: Entity identifier
            predicate: Predicate name
            value: Predicate value
            confidence: Confidence score [0, 1]
            timestamp: ISO timestamp (defaults to now)
            source: Source of the fact
            metadata: Additional metadata
            
        Returns:
            The created CanonicalRecord
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        record = CanonicalRecord(
            entity_id=entity_id,
            predicate=predicate,
            value=value,
            confidence=confidence,
            timestamp=timestamp,
            source=source,
            metadata=metadata or {}
        )
        
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO canonical_facts
            (entity_id, predicate, value, confidence, timestamp, source, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            entity_id,
            predicate,
            value,
            confidence,
            timestamp,
            source,
            json.dumps(metadata or {})
        ))
        self.conn.commit()
        
        # Update cache
        self._cache[(entity_id, predicate)] = record
        
        return record
    
    def get(
        self,
        entity_id: str,
        predicate: str
    ) -> Optional[CanonicalRecord]:
        """
        Get a canonical fact.
        
        Args:
            entity_id: Entity identifier
            predicate: Predicate name
            
        Returns:
            CanonicalRecord if found, None otherwise
        """
        # Check cache first
        key = (entity_id, predicate)
        if key in self._cache:
            return self._cache[key]
        
        # Query database
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM canonical_facts
            WHERE entity_id = ? AND predicate = ?
        """, (entity_id, predicate))
        
        row = cursor.fetchone()
        if row is None:
            return None
        
        record = CanonicalRecord(
            entity_id=row['entity_id'],
            predicate=row['predicate'],
            value=row['value'],
            confidence=row['confidence'],
            timestamp=row['timestamp'],
            source=row['source'],
            metadata=json.loads(row['metadata']) if row['metadata'] else {}
        )
        
        # Update cache
        self._cache[key] = record
        
        return record
    
    def is_stale(
        self,
        entity_id: str,
        predicate: str,
        max_age_days: float
    ) -> bool:
        """
        Check if a canonical fact is stale.
        
        Args:
            entity_id: Entity identifier
            predicate: Predicate name
            max_age_days: Maximum age in days
            
        Returns:
            True if record doesn't exist or is older than max_age_days
        """
        record = self.get(entity_id, predicate)
        if record is None:
            return True
        
        return record.age_days() > max_age_days
    
    def delete(self, entity_id: str, predicate: str) -> bool:
        """
        Delete a canonical fact.
        
        Args:
            entity_id: Entity identifier
            predicate: Predicate name
            
        Returns:
            True if record was deleted, False if not found
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            DELETE FROM canonical_facts
            WHERE entity_id = ? AND predicate = ?
        """, (entity_id, predicate))
        self.conn.commit()
        
        deleted = cursor.rowcount > 0
        
        # Remove from cache
        key = (entity_id, predicate)
        if key in self._cache:
            del self._cache[key]
        
        return deleted
    
    def get_all_for_entity(self, entity_id: str) -> List[CanonicalRecord]:
        """
        Get all canonical facts for an entity.
        
        Args:
            entity_id: Entity identifier
            
        Returns:
            List of CanonicalRecords
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM canonical_facts
            WHERE entity_id = ?
        """, (entity_id,))
        
        records = []
        for row in cursor.fetchall():
            record = CanonicalRecord(
                entity_id=row['entity_id'],
                predicate=row['predicate'],
                value=row['value'],
                confidence=row['confidence'],
                timestamp=row['timestamp'],
                source=row['source'],
                metadata=json.loads(row['metadata']) if row['metadata'] else {}
            )
            records.append(record)
        
        return records
    
    def clear(self):
        """Clear all records from the database."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM canonical_facts")
        self.conn.commit()
        self._cache.clear()
    
    def close(self):
        """Close database connection."""
        self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def __repr__(self) -> str:
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM canonical_facts")
        count = cursor.fetchone()[0]
        return f"CanonicalDB(records={count}, cache_size={len(self._cache)})"
    