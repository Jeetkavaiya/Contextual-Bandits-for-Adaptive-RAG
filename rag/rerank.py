from __future__ import annotations

from typing import List, Optional

from .types import Hit


class CrossEncoderReranker:
    """
    Cross-encoder reranker (HuggingFace Transformers).

    Default model is strong and common:
      BAAI/bge-reranker-base

    Requirements:
      pip install torch transformers
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: Optional[str] = None,
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length

        self._tok = None
        self._model = None

    def _load(self) -> None:
        if self._tok is not None and self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except Exception as e:
            raise ImportError("Install reranker deps: pip install torch transformers") from e

        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._model.to(self.device)  # type: ignore
        self._model.eval()  # type: ignore

    def rerank(self, query: str, hits: List[Hit], topR: int) -> List[Hit]:
        """
        Rerank topR hits; returns reranked hits (len = min(topR, len(hits))).
        Output hits have retriever="rerank" and score = reranker logit.
        """
        if topR <= 0 or not hits:
            return []

        self._load()
        import torch

        top_hits = hits[:topR]
        pairs = [(query, h.text) for h in top_hits]

        scores: List[float] = []

        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i : i + self.batch_size]
            q_batch = [p[0] for p in batch]
            d_batch = [p[1] for p in batch]

            enc = self._tok(  # type: ignore
                q_batch,
                d_batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}  # type: ignore

            with torch.no_grad():
                out = self._model(**enc)  # type: ignore
                logits = out.logits.squeeze(-1)
                logits = logits.detach().float().cpu().numpy().tolist()

            if isinstance(logits, float):
                logits = [logits]

            scores.extend([float(s) for s in logits])

        reranked: List[Hit] = []
        for h, s in zip(top_hits, scores):
            reranked.append(
                Hit(
                    chunk_id=h.chunk_id,
                    text=h.text,
                    retriever="rerank",
                    score=float(s),
                    bm25_score=h.bm25_score,
                    dense_score=h.dense_score,
                    bm25_norm=h.bm25_norm,
                    dense_norm=h.dense_norm,
                    meta=h.meta,
                )
            )

        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked