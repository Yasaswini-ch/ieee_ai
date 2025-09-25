from __future__ import annotations
from typing import List, Dict, Optional
from openai import OpenAI
from .utils import get_env
import requests
import os


def build_context(chunks: List[Dict]) -> str:
    lines = []
    for c in chunks:
        cid = f"{c['doc_id']} | {c['section_id']} | {c['version']}"
        title = c.get('title', '')
        text = c.get('text', '')
        lines.append(f"[Citation: {cid}] {title}\n{text}")
    return "\n\n".join(lines)


def answer_with_openai(query: str, chunks: List[Dict], explain: bool = False) -> str:
    client = OpenAI()
    chat_model = get_env("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    system = (
        "You are a legal/policy assistant. Use only the provided context to answer. "
        "Cite exact references as doc_id/section_id/version when relevant. If unsure, say you don't know."
    )
    if explain:
        system += " Also provide a plain-English explanation suitable for a non-expert."
    ctx = build_context(chunks)
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {query}\n\nAnswer with citations."},
    ]
    try:
        resp = client.chat.completions.create(model=chat_model, messages=msgs, temperature=0.2)
        return resp.choices[0].message.content
    except Exception as e:
        raise RuntimeError(
            "Failed to generate an answer via OpenAI. "
            "Please check API key/billing, model access, or try again later. "
            f"Details: {e}"
        ) from e


def answer_with_ollama(query: str, chunks: List[Dict], model: str = "llama3.1:8b", explain: bool = False, base_url: str = "http://localhost:11434") -> str:
    """Call a local Ollama model via HTTP API to generate an answer using provided context.
    Requires Ollama to be installed and the model pulled. Refer: https://github.com/ollama/ollama
    """
    system = (
        "You are a legal/policy assistant. Use only the provided context to answer. "
        "Cite exact references as doc_id/section_id/version when relevant. If unsure, say you don't know."
    )
    if explain:
        system += " Also provide a plain-English explanation suitable for a non-expert."
    ctx = build_context(chunks)
    prompt = f"Context:\n{ctx}\n\nQuestion: {query}\n\nAnswer with citations."
    payload = {
        "model": model,
        "system": system,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2}
    }
    try:
        resp = requests.post(f"{base_url}/api/generate", json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "")
    except Exception as e:
        raise RuntimeError(
            "Failed to generate an answer via local Ollama. "
            "Ensure Ollama is running (default http://localhost:11434) and the model is pulled. "
            f"Details: {e}"
        ) from e


def answer_with_gemini(query: str, chunks: List[Dict], explain: bool = False, model: Optional[str] = None) -> str:
    try:
        import google.generativeai as genai  # type: ignore
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set in environment.")
        genai.configure(api_key=api_key)
        gemini_model = model or get_env("GEMINI_CHAT_MODEL", "gemini-1.5-flash")
        system = (
            "You are a legal/policy assistant. Use only the provided context to answer. "
            "Cite exact references as doc_id/section_id/version when relevant. If unsure, say you don't know."
        )
        if explain:
            system += " Also provide a plain-English explanation suitable for a non-expert."
        ctx = build_context(chunks)
        prompt = f"Context:\n{ctx}\n\nQuestion: {query}\n\nAnswer with citations."
        model_obj = genai.GenerativeModel(gemini_model, system_instruction=system)
        resp = model_obj.generate_content(prompt)
        return getattr(resp, "text", None) or (resp.candidates[0].content.parts[0].text if getattr(resp, "candidates", None) else "")
    except Exception as e:
        raise RuntimeError(
            "Failed to generate an answer via Gemini. "
            "Please check GEMINI_API_KEY, model access/name, or try again later. "
            f"Details: {e}"
        ) from e


def answer(
    query: str,
    chunks: List[Dict],
    explain: bool = False,
    llm_provider: str | None = None,
    local_model: str | None = None,
    gemini_model: str | None = None,
    use_local_llm: bool | None = None,
) -> str:
    """Unified answer function that switches between OpenAI, Gemini, and Ollama.
    Backwards compatibility: if use_local_llm=True, prefer Ollama.
    """
    provider = (llm_provider or ("ollama" if use_local_llm else "openai")).lower()
    if provider == "ollama":
        return answer_with_ollama(query, chunks, model=(local_model or "llama3.1:8b"), explain=explain)
    if provider == "gemini":
        return answer_with_gemini(query, chunks, explain=explain, model=gemini_model)
    return answer_with_openai(query, chunks, explain=explain)


# ---------- General LLM (no-context) fallback ----------
def answer_general_openai(query: str) -> str:
    client = OpenAI()
    chat_model = get_env("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    system = (
        "You are a legal/policy assistant. Answer the user's question directly and clearly. "
        "If citations are not provided, do not fabricate any."
    )
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": query},
    ]
    try:
        resp = client.chat.completions.create(model=chat_model, messages=msgs, temperature=0.3)
        return resp.choices[0].message.content
    except Exception as e:
        raise RuntimeError(
            "Failed to generate a general answer via OpenAI. "
            f"Details: {e}"
        ) from e


def answer_general_gemini(query: str, model: Optional[str] = None) -> str:
    try:
        import google.generativeai as genai  # type: ignore
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set in environment.")
        genai.configure(api_key=api_key)
        gemini_model = model or get_env("GEMINI_CHAT_MODEL", "gemini-1.5-flash")
        system = (
            "You are a legal/policy assistant. Answer the user's question directly and clearly. "
            "If citations are not provided, do not fabricate any."
        )
        model_obj = genai.GenerativeModel(gemini_model, system_instruction=system)
        resp = model_obj.generate_content(query)
        return getattr(resp, "text", None) or (resp.candidates[0].content.parts[0].text if getattr(resp, "candidates", None) else "")
    except Exception as e:
        raise RuntimeError(
            "Failed to generate a general answer via Gemini. "
            f"Details: {e}"
        ) from e


def general_answer(query: str, provider: str, gemini_model: Optional[str] = None) -> str:
    p = (provider or "openai").lower()
    if p == "gemini":
        return answer_general_gemini(query, model=gemini_model)
    # default to OpenAI
    return answer_general_openai(query)
