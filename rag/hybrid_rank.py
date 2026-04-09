from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .types import Hit


def _min_max_norm(values: List[float]) -> List[float]:
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if abs(vmax - vmin) < 1e-12:
        return [0.0 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


def hybrid_rank(
    bm25_hits: List[Hit],
    dense_hits: List[Hit],
    x_bm25: float,
    y_dense: float,
    topk: Optional[int] = None,
) -> List[Hit]:
    """
    Combine BM25 + Dense hits by:
        score = x_bm25 * bm25_norm + y_dense * dense_norm

    Notes:
    - bm25_norm and dense_norm are min-max normalized within each list.
    - We merge by chunk_id and keep the best score from each retriever per chunk.
    """
    if topk is not None and topk <= 0:
        return []

    # Raw scores
    bm25_scores = [h.bm25_score if h.bm25_score is not None else h.score for h in bm25_hits]
    dense_scores = [h.dense_score if h.dense_score is not None else h.score for h in dense_hits]

    # Normalized scores
    bm25_norms = _min_max_norm([float(s) for s in bm25_scores])
    dense_norms = _min_max_norm([float(s) for s in dense_scores])

    # Best per chunk_id
    best_bm25: Dict[str, Tuple[Hit, float, float]] = {}
    for h, raw, norm in zip(bm25_hits, bm25_scores, bm25_norms):
        prev = best_bm25.get(h.chunk_id)
        if prev is None or raw > prev[1]:
            best_bm25[h.chunk_id] = (h, float(raw), float(norm))

    best_dense: Dict[str, Tuple[Hit, float, float]] = {}
    for h, raw, norm in zip(dense_hits, dense_scores, dense_norms):
        prev = best_dense.get(h.chunk_id)
        if prev is None or raw > prev[1]:
            best_dense[h.chunk_id] = (h, float(raw), float(norm))

    all_ids = set(best_bm25.keys()) | set(best_dense.keys())

    merged: List[Hit] = []
    for cid in all_ids:
        bm25_pack = best_bm25.get(cid)
        dense_pack = best_dense.get(cid)

        # Pick representative text/meta (prefer dense hit if exists)
        rep = dense_pack[0] if dense_pack is not None else bm25_pack[0]  # type: ignore

        bm25_raw = bm25_pack[1] if bm25_pack is not None else None
        bm25_norm = bm25_pack[2] if bm25_pack is not None else 0.0

        dense_raw = dense_pack[1] if dense_pack is not None else None
        dense_norm = dense_pack[2] if dense_pack is not None else 0.0

        combined = (x_bm25 * float(bm25_norm)) + (y_dense * float(dense_norm))

        merged.append(
            Hit(
                chunk_id=rep.chunk_id,
                text=rep.text,
                retriever="hybrid",
                score=float(combined),
                bm25_score=bm25_raw,
                dense_score=dense_raw,
                bm25_norm=float(bm25_norm),
                dense_norm=float(dense_norm),
                meta=rep.meta,
            )
        )

    merged.sort(key=lambda h: h.score, reverse=True)
    if topk is not None:
        merged = merged[:topk]
    return merged