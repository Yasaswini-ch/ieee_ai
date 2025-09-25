from __future__ import annotations
from typing import List, Dict, Tuple

# Cross-encoder re-ranking using sentence-transformers
# Default model is small and fast; you can change MODEL_NAME via parameter.
_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_ce = None
_ce_name = None

def _get_ce(model_name: str):
    global _ce, _ce_name
    if _ce is None or _ce_name != model_name:
        from sentence_transformers import CrossEncoder
        _ce = CrossEncoder(model_name)
        _ce_name = model_name
    return _ce


def rerank(query: str, records: List[Dict], model_name: str = _DEFAULT_MODEL, top_m: int = 5, batch_size: int = 32) -> List[Tuple[float, Dict]]:
    """
    Score (query, text) pairs with a cross-encoder and return top_m (score, record) sorted desc.
    records must each include a 'text' field.
    """
    if not records:
        return []
    ce = _get_ce(model_name)
    pairs = [(query, r.get("text", "")) for r in records]
    scores = ce.predict(pairs, batch_size=batch_size)
    # Zip and sort
    scored = list(zip([float(s) for s in scores], records))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_m]
