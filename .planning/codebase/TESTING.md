# Testing
_Last updated: 2026-04-30_

## Summary
This project has no automated test suite. Correctness is validated at runtime through print-based diagnostics, assertion guards in key scripts, and manual inspection of intermediate outputs (reward matrices, retrieval results). All testing is manual and integration-level.

## Test Framework
- None detected. No `pytest`, `unittest`, `nose`, or equivalent configured.
- No `pytest.ini`, `setup.cfg [tool:pytest]`, or `pyproject.toml [tool.pytest]` found.
- No test files (`test_*.py`, `*_test.py`) exist anywhere in the repository.

## What is Validated (Runtime Checks)
| Component | Validation Method |
|-----------|-------------------|
| Reward precomputation | Print progress + shape assertions on output arrays |
| Retrieval (BM25, FAISS) | Manual inspection of top-k results during dev |
| Bandit algorithms | Cumulative reward curves in `analyze_results.py` plots |
| Ollama connectivity | HTTP errors surface at request time |
| SLURM jobs | Job logs in `slurm/logs/` (stdout/stderr) |

## Untested Components
- `bandits/` — all bandit algorithm implementations (UCB, Thompson Sampling, etc.)
- `rag/` — retrieval pipeline (indexing, querying, reranking)
- `env/` — RL environment logic
- `precompute_rewards.py` — reward computation correctness
- `train_bandit.py` — training loop correctness
- `analyze_results.py` — analysis and plotting

## How to Run Tests
No test runner configured. To run the pipeline end-to-end manually:
```bash
python precompute_rewards.py   # precompute reward matrix
python train_bandit.py         # train bandit agents
python analyze_results.py      # analyze and plot results
```

## Gaps & Unknowns
- No property-based or unit tests for bandit algorithm correctness (e.g., UCB confidence bounds, Thompson posterior updates).
- No regression tests to catch changes in retrieval quality or reward distribution.
- No CI pipeline — no automated test runs on commit or PR.
- Reproducibility depends entirely on fixed random seeds being set in scripts (not verified).
