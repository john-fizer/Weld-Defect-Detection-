"""Vector store for agent memory."""

import os
from typing import List, Dict, Any, Optional
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False


class VectorStore:
    """Vector store for long-term agent memory.

    Stores learned playbooks, trade diaries, and insights.
    Uses ChromaDB for embedding storage and similarity search.
    """

    def __init__(self, persist_directory: Optional[str] = None):
        """Initialize vector store.

        Args:
            persist_directory: Directory to persist data
        """
        if not CHROMA_AVAILABLE:
            print("⚠️  ChromaDB not available - memory features disabled")
            self.client = None
            self.collection = None
            return

        persist_dir = persist_directory or os.getenv(
            "VECTORSTORE_PATH",
            "./data/vectorstore"
        )

        # Initialize ChromaDB client
        self.client = chromadb.Client(Settings(
            persist_directory=persist_dir,
            anonymized_telemetry=False,
        ))

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="trading_memory",
            metadata={"description": "Agent trading memory and insights"}
        )

    def add_memory(
        self,
        text: str,
        metadata: Dict[str, Any],
        memory_id: Optional[str] = None,
    ):
        """Add memory to vector store.

        Args:
            text: Memory text content
            metadata: Metadata dict
            memory_id: Optional memory ID
        """
        if not self.collection:
            return

        if not memory_id:
            from datetime import datetime
            memory_id = f"mem_{datetime.utcnow().timestamp()}"

        self.collection.add(
            documents=[text],
            metadatas=[metadata],
            ids=[memory_id],
        )

    def search_memories(
        self,
        query: str,
        n_results: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for relevant memories.

        Args:
            query: Search query
            n_results: Number of results to return
            filter: Metadata filter

        Returns:
            List of memory dicts
        """
        if not self.collection:
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=filter,
        )

        memories = []
        for i, doc in enumerate(results["documents"][0]):
            memories.append({
                "id": results["ids"][0][i],
                "text": doc,
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i] if "distances" in results else 0.0,
            })

        return memories

    def get_strategy_playbook(self, strategy: str) -> List[str]:
        """Get learned playbook for strategy.

        Args:
            strategy: Strategy name

        Returns:
            List of relevant insights
        """
        memories = self.search_memories(
            query=f"{strategy} strategy insights",
            n_results=10,
            filter={"strategy": strategy},
        )

        return [m["text"] for m in memories]

    def log_trade_diary(
        self,
        ticker: str,
        strategy: str,
        outcome: str,
        notes: str,
    ):
        """Log trade diary entry.

        Args:
            ticker: Ticker symbol
            strategy: Strategy used
            outcome: Trade outcome
            notes: Additional notes
        """
        text = f"Trade: {ticker} using {strategy}. Outcome: {outcome}. Notes: {notes}"

        metadata = {
            "type": "trade_diary",
            "ticker": ticker,
            "strategy": strategy,
            "outcome": outcome,
        }

        self.add_memory(text, metadata)
