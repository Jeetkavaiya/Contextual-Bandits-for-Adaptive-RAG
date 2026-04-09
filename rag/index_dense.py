from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .types import Chunk, Hit


class DenseIndex:
    """
    Dense vector search using:
      - Ollama embeddings API (local): POST /api/embed
      - FAISS IndexFlatIP with L2-normalized vectors (cosine-like)

    Defaults:
      OLLAMA_BASE_URL = http://localhost:11434
      OLLAMA_EMBED_MODEL = nomic-embed-text
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        embed_model: Optional[str] = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
        self.embed_model = embed_model or os.getenv("OLLAMA_EMBED_MODEL") or "nomic-embed-text"

        self._index = None
        self._dim: Optional[int] = None
        self._chunks: List[Chunk] = []

    @staticmethod
    def _normalize(X: np.ndarray) -> np.ndarray:
        denom = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        return X / denom

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Calls Ollama embed endpoint:
          POST {base_url}/api/embed
          { "model": "...", "input": [ ... ] }
        Returns float32 matrix [len(texts), dim]
        """
        try:
            import requests  # type: ignore
        except Exception as e:
            raise ImportError("Missing dependency requests. Install: pip install requests") from e

        url = f"{self.base_url}/api/embed"
        payload = {"model": self.embed_model, "input": texts}

        r = requests.post(url, json=payload, timeout=600)
        r.raise_for_status()
        data = r.json()

        embs = data.get("embeddings", None)
        if embs is None:
            raise RuntimeError(f"Ollama embed response missing 'embeddings'. Got keys: {list(data.keys())}")

        X = np.array(embs, dtype=np.float32)
        return X

    def _cache_key(self, chunks: List[Chunk]) -> str:
        """
        Cache based on:
        - embed model name
        - chunk ids + small hash of texts
        """
        h = hashlib.sha256()
        h.update(self.embed_model.encode("utf-8"))
        for c in chunks:
            h.update(c.chunk_id.encode("utf-8"))
            h.update(c.text[:200].encode("utf-8", errors="ignore"))  # partial text for speed
        return h.hexdigest()[:16]

    def build(
        self,
        chunks: List[Chunk],
        batch_size: int = 32,
        cache_dir: str = "data/cache",
        use_cache: bool = True,
        normalize: bool = True,
    ) -> None:
        """
        Builds FAISS index.
        If use_cache=True, caches embeddings to disk so rebuild is fast.
        """
        try:
            import faiss  # type: ignore
        except Exception as e:
            raise ImportError("Missing dependency faiss. Install: pip install faiss-cpu") from e

        self._chunks = list(chunks)
        os.makedirs(cache_dir, exist_ok=True)

        key = self._cache_key(self._chunks)
        emb_path = os.path.join(cache_dir, f"emb_{self.embed_model.replace(':','_')}_{key}.npy")

        if use_cache and os.path.exists(emb_path):
            X = np.load(emb_path).astype(np.float32)
        else:
            # Embed all chunk texts
            mats: List[np.ndarray] = []
            texts = [c.text for c in self._chunks]
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                Xb = self._embed_batch(batch)
                mats.append(Xb)
            X = np.vstack(mats).astype(np.float32)
            if use_cache:
                np.save(emb_path, X)

        if normalize:
            X = self._normalize(X)

        self._dim = int(X.shape[1])
        self._index = faiss.IndexFlatIP(self._dim)
        self._index.add(X)

    def search(self, query: str, k: int = 10) -> List[Hit]:
        if self._index is None or self._dim is None:
            raise RuntimeError("DenseIndex not built. Call build(chunks) first.")
        if k <= 0:
            return []

        try:
            import faiss  # type: ignore
        except Exception as e:
            raise ImportError("faiss required for search. Install: pip install faiss-cpu") from e

        qv = self._embed_batch([query]).astype(np.float32)
        qv = self._normalize(qv)

        scores, idxs = self._index.search(qv, k)  # type: ignore
        idxs = idxs[0].tolist()
        scores = scores[0].tolist()

        hits: List[Hit] = []
        for pos, s in zip(idxs, scores):
            if pos < 0:
                continue
            c = self._chunks[pos]
            hits.append(
                Hit(
                    chunk_id=c.chunk_id,
                    text=c.text,
                    retriever="dense",
                    score=float(s),        # cosine-like similarity (inner product on normalized vectors)
                    dense_score=float(s),
                    meta=c.meta,
                )
            )
        return hits