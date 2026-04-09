from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List

from rag.config import RAGConfig


@dataclass(frozen=True)
class Action:
    rewrite: int          # 0/1
    nq: int               # 1/3 (if rewrite=0 force nq=1)
    rerank: int           # 0/1
    k_final: int          # 5/10/20
    x_bm25: float         # 0.2/0.5/0.8
    # y_dense = 1 - x_bm25


def build_action_space() -> List[Action]:
    rewrites = [0, 1]
    nqs = [1, 3]
    reranks = [0, 1]
    ks = [5, 10, 20]
    xs = [0.2, 0.5, 0.8]

    actions: List[Action] = []
    for rw in rewrites:
        for nq in nqs:
            for rr in reranks:
                for k in ks:
                    for x in xs:
                        if rw == 0 and nq != 1:
                            continue
                        actions.append(Action(rewrite=rw, nq=nq, rerank=rr, k_final=k, x_bm25=x))
    return actions


def action_to_config(action: Action, base: RAGConfig) -> RAGConfig:
    """
    Convert an Action into a concrete RAGConfig used by pipeline.
    """
    y = 1.0 - float(action.x_bm25)
    nq = 1 if action.rewrite == 0 else action.nq

    cfg = replace(
        base,
        rewrite=bool(action.rewrite),
        nq=int(nq),
        rerank=bool(action.rerank),
        k_final=int(action.k_final),
        x_bm25=float(action.x_bm25),
        y_dense=float(y),
    )
    return cfg