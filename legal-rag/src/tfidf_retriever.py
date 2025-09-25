from __future__ import annotations
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def tfidf_retrieve(records: List[Dict], query: str, k: int = 5) -> List[Tuple[float, Dict]]:
    """Rank records by TF-IDF cosine similarity to the query.
    Returns list of (score, record) sorted descending.
    """
    if not records:
        return []
    texts = [r.get("text", "") for r in records]
    # Fit vectorizer on corpus; small and fast for moderate sizes
    vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
    X = vec.fit_transform(texts)
    qv = vec.transform([query])
    sims = cosine_similarity(qv, X)[0]
    top_idx = np.argsort(-sims)[:k]
    out: List[Tuple[float, Dict]] = []
    for idx in top_idx:
        score = float(sims[idx])
        out.append((score, records[idx]))
    return out
