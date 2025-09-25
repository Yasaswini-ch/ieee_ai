from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import numpy as np
import faiss
from pathlib import Path
import json

from openai import OpenAI
from .utils import get_env, ensure_dir, save_json, load_json
from .model_cache import get_sentence_transformer


class EmbeddingStore:
    def __init__(
        self,
        index_dir: str | Path,
        use_local: bool = False,
        local_model: Optional[str] = None,
        provider: str | None = None,
        gemini_model: Optional[str] = None,
    ):
        self.index_dir = Path(index_dir)
        ensure_dir(self.index_dir)
        self.meta_path = self.index_dir / "metadata.jsonl"
        self.faiss_path = self.index_dir / "index.faiss"
        self.dim = None
        self.index = None
        self.metadata: List[Dict] = []
        # Embedding backends
        # provider: 'openai' (default), 'gemini', or 'local'
        self.provider = (provider or ("local" if use_local else "openai")).lower()
        self.local_model_name = local_model or "sentence-transformers/all-MiniLM-L6-v2"
        self._local_model = None  # lazy init
        # OpenAI (when provider == 'openai')
        self.client = None  # lazy init only if provider == 'openai'
        self.embedding_model = get_env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        # Gemini (when provider == 'gemini')
        self._gemini_configured = False
        self.gemini_model = gemini_model or get_env("GEMINI_EMBED_MODEL", "models/text-embedding-004")

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts using OpenAI, returning a float32 numpy array.
        Raises RuntimeError with a helpful message on API errors so callers can handle gracefully.
        """
        if not texts:
            return np.zeros((0, 0), dtype="float32")
        if self.provider == "local":
            try:
                if self._local_model is None:
                    # Use cached model
                    self._local_model = get_sentence_transformer(self.local_model_name)
                arr = self._local_model.encode(texts, normalize_embeddings=False, convert_to_numpy=True).astype(
                    "float32"
                )
                if self.dim is None and arr.size > 0:
                    self.dim = arr.shape[1]
                return arr
            except Exception as e:
                raise RuntimeError(
                    "Failed to generate embeddings via local Sentence-Transformers model. "
                    "Ensure 'sentence-transformers' is installed and the model name is correct. "
                    f"Details: {e}"
                ) from e
        elif self.provider == "gemini":
            try:
                if not self._gemini_configured:
                    import google.generativeai as genai  # type: ignore
                    api_key = get_env("GEMINI_API_KEY")
                    if not api_key:
                        raise RuntimeError("GEMINI_API_KEY is not set in environment.")
                    genai.configure(api_key=api_key)
                    self._genai = genai
                    self._gemini_configured = True
                # Gemini embed API: embed_content per item
                vecs = []
                for t in texts:
                    resp = self._genai.embed_content(model=self.gemini_model, content=t)
                    v = resp["embedding"]["values"] if isinstance(resp, dict) else resp.embedding
                    vecs.append(v)
                arr = np.array(vecs, dtype="float32")
                if self.dim is None and arr.size > 0:
                    self.dim = arr.shape[1]
                return arr
            except Exception as e:
                raise RuntimeError(
                    "Failed to generate embeddings via Gemini. "
                    "Check GEMINI_API_KEY, model name, and network. "
                    f"Details: {e}"
                ) from e
        else:  # OpenAI
            try:
                if self.client is None:
                    api_key = get_env("OPENAI_API_KEY")
                    if not api_key:
                        raise RuntimeError("OPENAI_API_KEY is not set in environment.")
                    self.client = OpenAI(api_key=api_key)
                resp = self.client.embeddings.create(model=self.embedding_model, input=texts)
                vecs = [d.embedding for d in resp.data]
                arr = np.array(vecs, dtype="float32")
                if self.dim is None and arr.size > 0:
                    self.dim = arr.shape[1]
                return arr
            except Exception as e:  # covers RateLimitError, AuthenticationError, APIStatusError, etc.
                raise RuntimeError(
                    "Failed to generate embeddings via OpenAI. "
                    "Please check your API key/billing, model access, and network. "
                    f"Details: {e}"
                ) from e

    def add(self, records: List[Dict]) -> int:
        texts = [r["text"] for r in records]
        # Batch to reduce request size and be friendlier to rate limits
        batch_size = 64
        all_embs: List[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embs = self._embed_batch(batch)
            all_embs.append(embs)
        if not all_embs:
            return 0
        embs = np.concatenate(all_embs, axis=0) if len(all_embs) > 1 else all_embs[0]
        if embs.size == 0:
            return 0
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dim)
        # Normalize for cosine similarity
        faiss.normalize_L2(embs)
        self.index.add(embs)
        self.metadata.extend(records)
        return len(records)

    def save(self) -> None:
        if self.index is not None:
            faiss.write_index(self.index, str(self.faiss_path))
        with open(self.meta_path, "w", encoding="utf-8") as f:
            for m in self.metadata:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    def load(self) -> bool:
        if self.faiss_path.exists() and self.meta_path.exists():
            self.index = faiss.read_index(str(self.faiss_path))
            self.metadata = []
            with open(self.meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.metadata.append(json.loads(line))
            if self.index.ntotal != len(self.metadata):
                # mismatch; reset
                self.index = None
                self.metadata = []
                return False
            self.dim = self.index.d
            return True
        return False

    def search(self, query: str, k: int = 5) -> List[Tuple[float, Dict]]:
        if self.index is None or self.index.ntotal == 0:
            return []
        q = self._embed_batch([query])
        faiss.normalize_L2(q)
        scores, idxs = self.index.search(q, k)
        out: List[Tuple[float, Dict]] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            meta = self.metadata[idx]
            out.append((float(score), meta))
        return out

