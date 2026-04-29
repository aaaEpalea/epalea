# Evidence Index

The evidence index is a FAISS-based vector store combined with a JSON Lines metadata store
and an in-memory entity-predicate lookup table.

## Structure

```
evidence_index/
├── vector_store.faiss   # FAISS flat L2 index of 384d embeddings
└── metadata.jsonl       # One JSON object per evidence item
```

## How indexing works

1. Each evidence item's `text_content` is encoded by `all-MiniLM-L6-v2` (384d)
2. The embedding is added to the FAISS index
3. Metadata (entity_id, predicate, credibility, text, etc.) is stored in JSONL
4. After all evidence is added, `_rebuild_entity_index()` is called to build
   an in-memory `dict[(entity_id, predicate)] → [evidence_id, ...]` lookup

> **Important:** `_rebuild_entity_index()` must be called after loading an existing index
> from disk. This is handled automatically by `epalea index` and `load_model()`.

## Search

At query time, the index retrieves `top_k` evidence items for a given `(entity_id, predicate)` pair:

```python
evidence_ids = index.search(entity_id="C0001", predicate="compliance_level", top_k=5)
```

If `query_text` is provided, results are ranked by semantic similarity within the entity's
evidence set.

## Embedding model

Default: `all-MiniLM-L6-v2` from Sentence Transformers (384d).
This model is baked into `EvidenceIndex` — no separate configuration required.
