from __future__ import annotations
from typing import List, Dict, Tuple
from .embedding_store import EmbeddingStore


def retrieve_top_k(store: EmbeddingStore, query: str, k: int = 5) -> List[Tuple[float, Dict]]:
    return store.search(query, k)
