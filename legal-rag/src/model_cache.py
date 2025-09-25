from __future__ import annotations
from functools import lru_cache
from typing import Any

# Centralized, process-level caches for heavy ML resources.
# These caches persist across Streamlit reruns in the same session.

@lru_cache(maxsize=4)
def get_sentence_transformer(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


@lru_cache(maxsize=4)
def get_cross_encoder(model_name: str):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name)


@lru_cache(maxsize=2)
def get_qa_pipeline(model_name: str):
    from transformers import pipeline
    return pipeline("question-answering", model=model_name)
