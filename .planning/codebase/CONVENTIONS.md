# Coding Conventions
_Last updated: 2026-04-30_

## Summary
The codebase is Python-only, organized into three packages (`bandits/`, `rag/`, `env/`) plus top-level scripts. All files use a consistent, typed, class-based style with no linting or formatting config files detected. The primary patterns are abstract base classes for algorithm variants, frozen dataclasses for immutable data, and verbose module-level docstrings on every file.

---

## Language and Version

- Python 3.x (exact version not pinned — no `.python-version` or `pyproject.toml`)
- All files start with `from __future__ import annotations` for forward-reference type hints
- No linting config (`.flake8`, `.pylintrc`), formatting config (`.black`, `.prettierrc`), or `pyproject.toml` detected

---

## File Header Pattern

Every source file begins with one of:
1. A module-level docstring (`"""..."""`) — used in `bandits/` and top-level scripts
2. No docstring but bare imports — used in `rag/` and `env/`

```python
# bandits/linucb.py — example with docstring
"""LinUCB Disjoint — the primary contextual bandit algorithm.
...
"""
from __future__ import annotations
import numpy as np
from .base import BaseBandit
```

```python
# rag/pipeline.py — example without docstring
from __future__ import annotations
import time
from typing import Dict, List, Tuple
from .config import RAGConfig
```

---

## Naming Conventions

**Files:** `snake_case.py` throughout (`epsilon_greedy.py`, `state_features.py`, `hybrid_rank.py`)

**Classes:** `PascalCase`
- `BaseBandit`, `EpsilonGreedy`, `LinUCBDisjoint`, `LinearThompsonSampling`
- `RAGPipeline`, `RAGConfig`, `OllamaLLM`, `OllamaJudge`, `OllamaRewriter`
- `BM25Index`, `DenseIndex`, `CrossEncoderReranker`

**Functions / methods:** `snake_case`
- `build_action_space()`, `action_to_config()`, `extract_state()`, `state_to_vector()`
- Private helpers prefixed with `_`: `_current_epsilon()`, `_ucb_scores()`, `_best_by_chunk_id_keep_max_score()`

**Variables:**
- `snake_case` for local variables and instance attributes
- Short single-letter names used in tight numeric loops: `N, K, d, t, x, r`
- NumPy arrays named by their mathematical meaning: `self.A`, `self.b`, `self.B`, `self.f`, `theta_a`

**Constants:** `UPPER_SNAKE_CASE` — e.g., `REPO_ROOT`, `STATE_KEYS`, `_WORD_RE`

**Type aliases:** `PascalCase` — e.g., `RetrieverName = Literal["bm25", "dense", "hybrid", "rerank"]`

---

## Type Annotations

All public APIs are fully annotated. Return types are explicit on every method:

```python
def select(self, context: Optional[np.ndarray] = None) -> int: ...
def train_bandit(...) -> Dict[str, np.ndarray]: ...
def action_to_config(action: Action, base: RAGConfig) -> RAGConfig: ...
```

`# type: ignore[override]` comments are used deliberately where subclasses intentionally narrow parameter types (e.g., `linucb.py` and `thompson.py` override `select`/`update` to require a non-optional context).

---

## Class Design

**Abstract base classes** via `abc.ABC` and `@abstractmethod` define the algorithm contract:
- `bandits/base.py` — `BaseBandit` with abstract `select()` and `update()`
- Concrete implementations inherit and add only algorithm-specific state

**Dataclasses** used for all data-holding objects:
- `@dataclass(frozen=True)` for immutable records: `Chunk`, `Hit`, `Action`
- `@dataclass` (mutable) for accumulation objects: `TimingInfo`, `TokenUsage`, `MetricsRow`, `RAGConfig`
- `RAGConfig.__post_init__` performs validation and auto-correction inline

**Plain classes** used for stateful components with behavior: `RAGPipeline`, `OllamaLLM`, `OllamaJudge`, `RAGEnv`, `BM25Index`, `DenseIndex`, `StateScaler`

---

## Code Organization Pattern — Section Separators

Methods within a class are grouped under dashed comment headers:

```python
# ------------------------------------------------------------------
# Core API
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# Convenience
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# Persistence
# ------------------------------------------------------------------
```

This pattern appears in `bandits/base.py`, `bandits/linucb.py`, `bandits/thompson.py`.

Top-level scripts (`train_bandit.py`, `precompute_rewards.py`) use numbered dashed sections:

```python
# ---------------------------------------------------------------------------
# 1. Load reward table
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 2. Fit state scaler
# ---------------------------------------------------------------------------
```

---

## Import Organization

1. `from __future__ import annotations` — always first
2. Standard library imports (alphabetical within group)
3. Third-party imports (`numpy`, `requests`, `torch`)
4. Local relative imports using dot notation: `from .base import BaseBandit`
5. Top-level scripts add `sys.path.insert(0, str(REPO_ROOT))` before local package imports

---

## Error Handling Patterns

**Validation errors** raised in `__post_init__` for configuration objects:
```python
# rag/config.py
if self.nq not in (1, 3):
    raise ValueError(f"nq must be 1 or 3 for checkpoint baseline, got {self.nq}")
```

**Dependency import errors** wrapped with helpful install instructions:
```python
# rag/index_bm25.py
try:
    from rank_bm25 import BM25Okapi
except Exception as e:
    raise ImportError("Missing dependency rank-bm25. Install: pip install rank-bm25") from e
```

**External call errors** use broad `except Exception` with silent fallback (particularly for LLM/JSON parsing):
```python
# rag/judge.py
try:
    obj = json.loads(text)
    if isinstance(obj, dict):
        return obj
except Exception:
    pass
```

**State precondition errors** use `RuntimeError`:
```python
# env/rag_env.py
raise RuntimeError("Call reset(item) before step(action_idx).")
```

**Fatal script errors** use `sys.exit()` with a `[ERROR]` prefix string:
```python
# train_bandit.py
sys.exit(f"[ERROR] Reward table not found at {table_path}\nRun precompute_rewards.py first ...")
```

---

## Logging Pattern

No `logging` module is used anywhere. All progress output is `print(..., flush=True)`:

```python
print(f"[setup] Loading chunks from {pdf_path} …", flush=True)
print(f"  elapsed          : {elapsed:.1f}s", flush=True)
print(f"  mean reward      : {s['mean_reward']:.4f}", flush=True)
```

The `flush=True` argument is used consistently on progress prints (important for SLURM/HPC buffering). Section dividers use `'='*60` and `'-'*60`.

---

## CLI Pattern

Top-level scripts use `argparse` with a dedicated `parse_args() -> argparse.Namespace` function and a `main() -> None` entry point guarded by `if __name__ == "__main__": main()`. No `click` or `typer` is used.

---

## Persistence Pattern

Bandit agents persist state via `pickle` through `BaseBandit.save(path)` / `load(path)` methods defined in `bandits/base.py`. Results are stored as `.npz` (NumPy compressed arrays) and `.json` summary files. Evaluation details are written as `.jsonl` (one JSON object per line).

---

## Gaps & Unknowns

- No linting or formatting configuration detected — code style is enforced only by convention
- No `pre-commit` hooks, `tox.ini`, or `Makefile` found
- Python version is not pinned anywhere (`requirements_ccr.txt` was not read in full)
- Notebook files (`notebooks/`) exist but were not analyzed for convention adherence
