from __future__ import annotations
from typing import List, Dict, Tuple

# Lightweight extractive QA using Hugging Face transformers
# Defaults to a small SQuAD-tuned model. You can change MODEL_NAME for higher accuracy.
MODEL_NAME = "distilbert-base-cased-distilled-squad"

_qa_pipe = None

def _get_pipeline():
    global _qa_pipe
    if _qa_pipe is None:
        from transformers import pipeline
        _qa_pipe = pipeline("question-answering", model=MODEL_NAME)
    return _qa_pipe


def best_span(question: str, contexts: List[Dict], top_n: int = 3) -> Tuple[str, List[Tuple[float, Dict]]]:
    """
    Run extractive QA over up to top_n contexts and return the best answer text with scored contexts.
    contexts: list of metadata dicts including a 'text' field.
    Returns (answer_text, [(score, ctx_meta), ...])
    """
    qa = _get_pipeline()
    scored: List[Tuple[float, Dict]] = []
    best_answer = ""
    best_score = float("-inf")

    for ctx in contexts[:top_n]:
        context_text = ctx.get("text", "")
        if not context_text.strip():
            continue
        try:
            out = qa(question=question, context=context_text)
            score = float(out.get("score", 0.0))
            answer = out.get("answer", "")
            scored.append((score, ctx))
            if score > best_score and answer:
                best_score = score
                best_answer = answer
        except Exception:
            # Skip problematic context silently
            continue

    # Sort contexts by score desc
    scored.sort(key=lambda x: x[0], reverse=True)
    return best_answer, scored
