from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from rag.config import RAGConfig
from rag.evalset import EvalItem
from rag.judge import OllamaJudge
from rag.metrics import MetricsRow, score_one
from rag.pipeline import RAGPipeline

from .action_space import Action, action_to_config, build_action_space
from .state_features import extract_state, state_to_vector


class RAGEnv:
    """
    1-step RL environment for checkpoint 1.

    Episode = one query.
      reset(item) -> state_vector
      step(action_idx) -> (next_state_vector, reward, done, info)

    done is always True after step(), because it's a 1-step decision problem.
    """

    def __init__(
        self,
        pipe: RAGPipeline,
        base_config: Optional[RAGConfig] = None,
        judge_model: Optional[str] = None,
        reward_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.pipe = pipe
        self.base_config = base_config or RAGConfig()
        self.actions: List[Action] = build_action_space()
        self.judge = OllamaJudge(model=judge_model)

        # Reward = quality + retrieval - cost (simple + paper-friendly)
        # You can tune these later.
        self.w = reward_weights or {
            "w_correctness": 1.0,     # correctness in [0, 1] after /2
            "w_faithfulness": 0.5,    # faithfulness in {0,1}
            "w_recall": 0.5,          # evidence_recall@k in {0,1} (in-domain only)
            "w_time": 0.05,           # seconds penalty
            "w_tokens": 0.002,        # token penalty (tokens/1000)
            "w_chunks": 0.01,         # k_final penalty
            "w_rerank": 0.05,         # rerank penalty
            "w_rewrite": 0.02,        # rewrite penalty
        }

        self._cur_item: Optional[EvalItem] = None
        self._cur_state_vec: Optional[List[float]] = None

    def action_space_n(self) -> int:
        return len(self.actions)

    def reset(self, item: EvalItem) -> List[float]:
        """
        Stores the current EvalItem and returns the state vector.
        State is extracted from RAW query (cheap features + probe retrieval stats).
        """
        self._cur_item = item

        st = extract_state(
            raw_query=item.raw_query,
            bm25=self.pipe.bm25,
            dense=self.pipe.dense,
        )
        self._cur_state_vec = state_to_vector(st)
        return list(self._cur_state_vec)

    def step(self, action_idx: int) -> Tuple[List[float], float, bool, Dict[str, Any]]:
        if self._cur_item is None or self._cur_state_vec is None:
            raise RuntimeError("Call reset(item) before step(action_idx).")

        if action_idx < 0 or action_idx >= len(self.actions):
            raise ValueError(f"Invalid action_idx {action_idx}, must be in [0, {len(self.actions)-1}]")

        item = self._cur_item
        action = self.actions[action_idx]
        cfg = action_to_config(action, base=self.base_config)

        # Run RAG with chosen config
        res = self.pipe.run_rag(
            session_id=item.qid,
            raw_query=item.raw_query,
            chat_history=item.chat_history,
            config=cfg,
        )

        # Score with metrics (uses Ollama judge)
        row: MetricsRow = score_one(item, res, judge=self.judge)

        # Reward calculation
        correctness = (float(row.correctness) / 2.0) if row.correctness is not None else 0.0
        faithfulness = float(row.faithfulness) if row.faithfulness is not None else 0.0
        recall = float(row.evidence_recall_at_k) if row.evidence_recall_at_k is not None else 0.0

        # Costs
        t = float(row.total_time_s)
        tok = float(row.total_tokens) if row.total_tokens is not None else 0.0
        tok_k = tok / 1000.0  # scale
        k_final = float(row.k_final)
        rerank_on = float(row.rerank_enabled)
        rewrite_on = float(row.rewrite_enabled)

        reward = 0.0
        reward += self.w["w_correctness"] * correctness
        reward += self.w["w_faithfulness"] * faithfulness
        reward += self.w["w_recall"] * recall

        reward -= self.w["w_time"] * t
        reward -= self.w["w_tokens"] * tok_k
        reward -= self.w["w_chunks"] * k_final
        reward -= self.w["w_rerank"] * rerank_on
        reward -= self.w["w_rewrite"] * rewrite_on

        info: Dict[str, Any] = {
            "action_idx": action_idx,
            "action": asdict(action),
            "config": cfg.to_dict(),
            "metrics": asdict(row),
            "answer": res.answer,
            "context_chunk_ids": [c.chunk_id for c in res.context_chunks],
            "retrieval_query": res.retrieval_query,
            "timings": asdict(res.timings),
            "tokens": asdict(res.tokens),
        }

        done = True
        next_state = list(self._cur_state_vec)  # 1-step episode: next_state not used
        return next_state, float(reward), done, info