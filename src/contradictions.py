from __future__ import annotations
from typing import List, Dict, Tuple


def detect_contradictions(chunks: List[Dict]) -> List[Tuple[Tuple[int, int], float]]:
    """
    Placeholder contradiction detection. To keep initial dependencies light and fast on Windows,
    we avoid loading a heavy Transformer by default. This function returns an empty list.
    You can later replace with a real NLI model via transformers.

    Returns list of ((i, j), score) pairs where higher score indicates stronger contradiction.
    """
    return []

# Example implementation (commented) if you want to use transformers locally:
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
# _nli = None
# def _get_nli():
#     global _nli
#     if _nli is None:
#         model = "MoritzLaurer/DeBERTa-v3-base-mnli"
#         _nli = pipeline("text-classification", model=model, return_all_scores=True)
#     return _nli
#
# def detect_contradictions(chunks: List[Dict]) -> List[Tuple[Tuple[int, int], float]]:
#     nli = _get_nli()
#     results = []
#     for i in range(len(chunks)):
#         for j in range(i+1, len(chunks)):
#             a = chunks[i]["text"][:1000]
#             b = chunks[j]["text"][:1000]
#             scores = nli({"text": a, "text_pair": b})[0]
#             label_to_score = {s["label"].lower(): s["score"] for s in scores}
#             contra = max(label_to_score.get("contradiction", 0.0), label_to_score.get("CONTRADICTION", 0.0))
#             if contra > 0.6:
#                 results.append(((i, j), float(contra)))
#     return results
