from __future__ import annotations
from typing import List, Dict, Tuple
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

_sentence_re = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    # Simple sentence splitter; avoids heavy dependencies
    parts = _sentence_re.split(text)
    # Clean and filter very short sentences
    sents = [s.strip() for s in parts if len(s.strip()) > 20]
    return sents


def build_corpus_sentences(records: List[Dict]) -> Tuple[List[str], List[Tuple[int, int, str]]]:
    """
    Take records (with title,text, doc_id, section_id, version) and produce
    a list of sentences, along with mapping back to (record_index, sentence_index, section_title).
    Returns (sentences, meta_map)
    """
    sentences: List[str] = []
    meta: List[Tuple[int, int, str]] = []
    for i, rec in enumerate(records):
        sents = split_sentences(rec.get("text", ""))
        for j, s in enumerate(sents):
            sentences.append(s)
            meta.append((i, j, rec.get("title", "Section")))
    return sentences, meta


def mmr_select(query_vec: np.ndarray, sent_vecs: np.ndarray, k: int = 6, lambda_mult: float = 0.7) -> List[int]:
    """
    Maximal Marginal Relevance selection over sentence vectors to balance relevance and diversity.
    query_vec: (d,)
    sent_vecs: (n, d)
    Returns selected indices.
    """
    if sent_vecs.shape[0] == 0:
        return []
    # Normalize
    def norm(a):
        n = np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8
        return a / n
    q = norm(query_vec.reshape(1, -1))
    V = norm(sent_vecs)
    # Cosine similarities
    rel = (V @ q.T).reshape(-1)  # (n,)
    selected: List[int] = []
    candidates = list(range(len(rel)))
    while candidates and len(selected) < k:
        if not selected:
            idx = int(np.argmax(rel[candidates]))
            selected.append(candidates[idx])
            candidates.pop(idx)
            continue
        sel_vecs = V[selected]
        # Diversity: max similarity to any already selected
        div = np.max(sel_vecs @ V[candidates].T, axis=0)
        mmr = lambda_mult * rel[candidates] - (1 - lambda_mult) * div
        idx = int(np.argmax(mmr))
        selected.append(candidates[idx])
        candidates.pop(idx)
    return selected


def summarize_records(records: List[Dict], sentences_max: int = 6) -> Dict:
    """
    Build an extractive summary card for a set of section records belonging to one document.
    Returns a dict with keys: title, doc_id, version_set, bullets, coverage_count
    """
    if not records:
        return {"title": "", "doc_id": "", "version_set": [], "bullets": [], "coverage_count": 0}

    # Aggregate info
    doc_id = records[0].get("doc_id", "")
    title = records[0].get("title", "Policy")
    versions = sorted({r.get("version", "") for r in records if r.get("version")})

    # Build sentence corpus and TF-IDF sentence vectors
    corpus_sents, meta = build_corpus_sentences(records)
    if not corpus_sents:
        return {"title": title, "doc_id": doc_id, "version_set": versions, "bullets": [], "coverage_count": 0}

    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(corpus_sents)  # (n_sents, d)
    sent_vecs = X.toarray().astype("float32")

    # Query vector: centroid of all sentences (document theme)
    query_vec = np.mean(sent_vecs, axis=0)

    sel_idx = mmr_select(query_vec, sent_vecs, k=sentences_max, lambda_mult=0.7)
    # Keep original order within the document by sentence index order
    sel_idx_sorted = sorted(sel_idx)

    bullets: List[str] = ["• " + corpus_sents[i] for i in sel_idx_sorted]
    coverage = len({meta[i][2] for i in sel_idx_sorted})  # number of unique section titles covered

    return {
        "title": title,
        "doc_id": doc_id,
        "version_set": versions,
        "bullets": bullets,
        "coverage_count": coverage,
    }
