"""
memory/chroma.py — Long-term memory backed by ChromaDB.

Each agent gets its own namespace (collection) so memories never bleed
between agents. Supports semantic search via sentence-transformers.

If sentence-transformers is not installed, falls back to ChromaDB's
built-in default embeddings (no extra dependency, slightly lower quality).

Usage:
    mem = MemoryManager(agent_name="researcher")
    mem.store("RAG stands for Retrieval-Augmented Generation")
    results = mem.search("what is RAG", n_results=3)
"""

import hashlib
import os
from datetime import datetime, timezone
from typing import Optional

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", ".chroma")


class MemoryManager:
    """
    Per-agent long-term memory backed by ChromaDB.
    Stores text chunks with metadata and retrieves by semantic similarity.
    """

    def __init__(
        self,
        agent_name: str,
        persist_dir: str = CHROMA_PERSIST_DIR,
        use_default_embeddings: bool = False,
    ):
        """
        agent_name           : used as the ChromaDB collection name (namespace)
        persist_dir          : where ChromaDB stores its data on disk
        use_default_embeddings: True = use ChromaDB built-in (no torch needed)
                               False = use sentence-transformers (better quality)
        """
        self.agent_name = agent_name
        self.persist_dir = persist_dir
        self._client = None
        self._collection = None
        self._use_default = use_default_embeddings
        self._ef = None  # embedding function

    # ── Lazy init ─────────────────────────────────────────────────────────────

    def _ensure_init(self):
        """Initialise ChromaDB client and collection on first use."""
        if self._collection is not None:
            return

        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError(
                "chromadb not installed. Run: pip install chromadb"
            )

        self._client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        self._ef = self._build_embedding_function()

        self._collection = self._client.get_or_create_collection(
            name=self._safe_collection_name(self.agent_name),
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    def _build_embedding_function(self):
        if self._use_default:
            return None  # ChromaDB uses its own default

        try:
            from chromadb.utils.embedding_functions import (
                SentenceTransformerEmbeddingFunction,
            )
            return SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"  # fast, small, good quality
            )
        except ImportError:
            # Graceful fallback if sentence-transformers not installed
            return None

    @staticmethod
    def _safe_collection_name(name: str) -> str:
        """ChromaDB collection names must be 3–63 chars, alphanumeric + hyphens."""
        safe = "".join(c if c.isalnum() else "-" for c in name.lower())
        safe = safe.strip("-")
        if len(safe) < 3:
            safe = safe + "-mem"
        return safe[:63]

    # ── Public API ─────────────────────────────────────────────────────────────

    def store(self, text: str, metadata: Optional[dict] = None) -> str:
        """
        Store a text chunk in long-term memory.
        Returns the document ID.
        """
        self._ensure_init()

        doc_id = self._make_id(text)
        meta = {
            "agent": self.agent_name,
            "ts": datetime.now(timezone.utc).isoformat(),
            **(metadata or {}),
        }

        # Upsert — won't duplicate if same content stored twice
        self._collection.upsert(
            ids=[doc_id],
            documents=[text],
            metadatas=[meta],
        )
        return doc_id

    def search(self, query: str, n_results: int = 3) -> list[dict]:
        """
        Retrieve the top-n most semantically similar memories.

        Returns a list of dicts:
            [{"text": ..., "score": ..., "metadata": {...}}, ...]
        Score is cosine distance (lower = more similar).
        """
        self._ensure_init()

        count = self._collection.count()
        if count == 0:
            return []

        n = min(n_results, count)

        results = self._collection.query(
            query_texts=[query],
            n_results=n,
            include=["documents", "distances", "metadatas"],
        )

        output = []
        docs = results["documents"][0]
        distances = results["distances"][0]
        metadatas = results["metadatas"][0]

        for doc, dist, meta in zip(docs, distances, metadatas):
            output.append({
                "text": doc,
                "score": round(1 - dist, 4),  # convert distance → similarity
                "metadata": meta,
            })

        # Sort by similarity descending
        output.sort(key=lambda x: x["score"], reverse=True)
        return output

    def search_as_context(self, query: str, n_results: int = 3, min_score: float = 0.3) -> str:
        """
        Convenience method — returns memories formatted as a context string
        ready to inject into the agent's system prompt or conversation.
        Returns empty string if no relevant memories found.
        """
        results = self.search(query, n_results=n_results)
        relevant = [r for r in results if r["score"] >= min_score]

        if not relevant:
            return ""

        lines = ["[Relevant memories from previous runs:]"]
        for i, r in enumerate(relevant, 1):
            lines.append(f"  {i}. {r['text']} (relevance: {r['score']})")

        return "\n".join(lines)

    def delete(self, doc_id: str):
        """Delete a specific memory by ID."""
        self._ensure_init()
        self._collection.delete(ids=[doc_id])

    def clear(self):
        """Wipe all memories for this agent's namespace."""
        self._ensure_init()
        self._client.delete_collection(
            self._safe_collection_name(self.agent_name)
        )
        self._collection = None  # force re-init on next use

    @property
    def count(self) -> int:
        """Number of stored memories."""
        self._ensure_init()
        return self._collection.count()

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _make_id(text: str) -> str:
        """Deterministic ID from content — prevents duplicates."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def __repr__(self):
        return (
            f"<MemoryManager agent={self.agent_name} "
            f"persist_dir={self.persist_dir}>"
        )
