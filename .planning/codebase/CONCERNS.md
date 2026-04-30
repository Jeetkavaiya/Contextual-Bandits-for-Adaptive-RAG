# Codebase Concerns
_Last updated: 2026-04-30_

## Summary

This project implements a contextual bandit system for RAG pipeline optimization. The core logic is reasonably clean, but several concerns exist around hardcoded values scattered across Python scripts, a fragile two-phase SLURM pipeline with no dependency enforcement, duplicate reward-weight definitions, missing no-test infrastructure, and a single-document scope that limits generalization. The LLM-as-judge approach introduces non-deterministic evaluation that can silently corrupt training data if the judge fails to parse JSON.

---

## Hardcoded Values

**Reward weights duplicated across two files:**
- Identical `reward_weights` dict defined independently in `precompute_rewards.py` (lines 196–205) and `env/rag_env.py` (lines 41–50).
- Risk: If weights are tuned in one file, the other silently diverges, making offline (precomputed) rewards inconsistent with online environment rewards.
- Fix: Extract to a shared constant in `rag/config.py` or a new `env/reward_config.py` and import from both.

**Chunk size and overlap hardcoded in `precompute_rewards.py`:**
- `build_pipeline()` calls `load_or_build_chunks(chunk_size=450, overlap=80)` at line 79 of `precompute_rewards.py`.
- `pdf_ingest.py` defaults are `chunk_size=1400, overlap=200` — a completely different parameterization.
- These values are not CLI-configurable. Running any other entry point will build a different chunk set, invalidating cached embeddings.
- Fix: Add `--chunk_size` and `--overlap` CLI args to `precompute_rewards.py` and expose them through `build_pipeline()`.

**LLM model fallback string hardcoded in two places:**
- `rag/llm.py` line 31 and `rag/rewrite.py` line 33 both default to `"hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF"` — a specific GGUF variant on HuggingFace.
- This default will silently fail if Ollama does not have that exact model pulled, and there is no startup check.
- Fix: Require the model name to be set via environment variable and raise a clear error if unset.

**SLURM array size tied to evalset size by hand:**
- `slurm/0_precompute.sh` has `#SBATCH --array=0-9` (10 tasks) and `CHUNK_SIZE=10` to cover 100 items. These two values must be kept in sync manually. If the evalset grows or changes the array bounds need a matching edit.
- Fix: Compute array bounds programmatically or add a guard that verifies task coverage before submitting.

**Ollama port hardcoded:**
- `OLLAMA_PORT=11434` in `slurm/0_precompute.sh` line 35. Multiple concurrent SLURM array tasks running on the same node would collide on this port.
- Fix: Derive port from `SLURM_JOB_ID % 1000 + 11000` or similar to guarantee uniqueness per task.

**Email address committed to SLURM script:**
- `#SBATCH --mail-user=devchira@buffalo.edu` in `slurm/0_precompute.sh` line 28.
- Not a security risk but is institution-specific. Any other team member running this job will not receive failure notifications.
- Fix: Replace with `--mail-user=${SBATCH_MAIL_USER:-devchira@buffalo.edu}` or parameterize via a shared config file.

---

## Technical Debt

**Duplicate state-extraction during precompute:**
- `compute_rewards_for_item()` in `precompute_rewards.py` calls `extract_state()` once and then calls `pipe.run_rag()` K=60 times. For every RAG run, `run_rag()` does NOT call `extract_state()` again — this is correct. However, `env/rag_env.py` `reset()` also calls `extract_state()`, then `step()` calls `run_rag()`. The state is therefore computed inconsistently: precompute uses the raw probe, online env uses identical logic but at a different call site. Any future change to `extract_state()` must be applied carefully in both paths.

**`RAGEnv` is unused in the bandit training pipeline:**
- `env/rag_env.py` implements a full gym-style environment, but `train_bandit.py` never imports or uses it. Training operates directly on the precomputed reward table. The `RAGEnv` class is effectively dead code for the current workflow.
- Risk: The class may fall out of sync with the rest of the codebase (e.g., reward weights drift).

**`StateScaler` is a manual reimplementation of scikit-learn's `StandardScaler`:**
- `train_bandit.py` lines 74–91 implement mean/std normalization from scratch. This is fine for removing a `sklearn` dependency, but it lacks `inverse_transform`, partial-fit support, and is not serialized with the checkpoint (only `state_mean` and `state_std` arrays are saved to the NPZ file separately).
- Risk: At inference time, a consumer must manually reconstruct the scaler from the saved arrays; there is no `scaler.pkl` checkpoint.

**`precompute_rewards.py` backward-compat `_Args` class:**
- Lines 303–316 define an inline `_Args` class to handle the case where no subcommand is given. This is a code smell that exists to maintain backward compatibility with an older CLI that has been superseded by the `compute`/`merge` subcommand design. The fallback silently runs with `evalset_25_gold.jsonl` which may confuse users who expect the 100-item set.

**LinUCB runs `np.linalg.solve` in a Python loop over all arms:**
- `linucb.py` `_ucb_scores()` (lines 69–79) solves two linear systems per arm in a Python `for` loop. With K=60 arms and d=11 dimensions this is fast today, but is architecturally O(K) individual LAPACK calls. If K or d grows, this becomes a bottleneck.
- Similarly, `thompson.py` `select()` (lines 74–91) loops over all arms calling `multivariate_normal` per arm.

---

## Missing Error Handling

**Ollama readiness check has no abort if server never starts:**
- `slurm/0_precompute.sh` lines 82–88 poll for Ollama readiness for 60 seconds but silently continue if the server is never ready. The subsequent Python call will fail with a `ConnectionRefusedError` rather than a descriptive message.
- Fix: Add `exit 1` after the loop if the readiness check never succeeds.

**`precompute_rewards.py` catches all exceptions silently:**
- `compute_rewards_for_item()` line 161: `except Exception as exc:` prints a `[WARN]` and stores `NaN`, but the exception is swallowed. A systematic failure (e.g., Ollama server crash mid-run) will silently fill the reward table with NaN rather than aborting.
- Fix: Track failure count and abort (or raise) if more than a configurable fraction of actions fail for a given item.

**`OllamaJudge` falls back to `correctness=0, faithfulness=0` on parse failure:**
- `rag/judge.py` lines 102–103: When the judge LLM produces unparseable JSON, the item is silently scored as `(0, 0)`. This is indistinguishable from a genuinely wrong, unfaithful answer. If judge parse failures are frequent, they will bias the reward table toward pessimistic values.
- Fix: Track a `judge_parse_failed` flag in `MetricsRow` (the `notes` field already contains `"judge_parse_failed"` but is not aggregated) and surface the parse failure rate in training logs.

**`analyze_results.py` approximates number of arms incorrectly:**
- Line 47: `n_arms_total = int(hist["chosen_arms"].max()) + 1` — this estimates the arm count from the max observed arm index rather than from the action space size. If the highest-indexed arms are never selected (which is the metric being measured), the denominator is wrong and `arms_explored_frac` is inflated.

---

## Reproducibility Concerns

**Random seed is partially applied:**
- `train_bandit.py` uses `np.random.RandomState(args.seed)` (line 215), which governs episode query sampling.
- However, `LinearThompsonSampling.select()` (line 82) calls `np.random.multivariate_normal` — the global NumPy random state — not the seeded `rng` passed to `train_bandit()`. Thompson Sampling results are therefore not reproducible even with `--seed 42`.
- Fix: Pass `rng` into `LinearThompsonSampling` and use it for sampling instead of the global state.

**SLURM array tasks produce independent partial files, no atomicity:**
- If a SLURM task is preempted or fails mid-run, `results/rewards/rewards_{start}_{end}.npz` may be written partially (numpy writes atomically, but a task restart will overwrite a valid partial file without warning).
- There is no checksum or manifest file to detect missing or corrupted partial shards before the merge step.

**Data files committed to the repository:**
- `data/cache/` contains `.jsonl` chunk files and `.npy` embedding files that are checked into git. These are derived artifacts that depend on the PDF and chunking parameters. Committing them alongside source risks silently serving stale cache if parameters change but the cache key hashes happen to match (unlikely but possible).
- `results/` contains baseline run outputs (`baseline_5_details.jsonl`, `baseline_5_metrics.csv`, `baseline_5_metrics.json`) which are also committed. These will not reflect future retraining automatically.

**Single evaluation document:**
- The entire experiment is based on `data/prudentservices_dataset_rag.pdf`, a single proprietary PDF. There is no mechanism to swap in a different corpus, no abstract data interface, and no version tag on the PDF itself. Results are therefore not generalizable and cannot be reproduced without access to that exact file.

---

## Security Considerations

**Personal email hardcoded in SLURM script:**
- `slurm/0_precompute.sh` line 28: `devchira@buffalo.edu` is committed to version control. Low risk, but should be parameterized (see Hardcoded Values section above).

**No input validation on evalset items:**
- `rag/evalset.py` (loaded by `precompute_rewards.py`) reads JSONL directly and passes `raw_query` strings to Ollama without sanitization. If the evalset is sourced from untrusted input, prompt injection is possible. Not a concern for the current controlled academic setting, but worth noting for any future deployment.

**Pickle used for agent checkpoints:**
- `bandits/base.py` lines 55–62 serialize/deserialize agent state with `pickle`. Loading a checkpoint from an untrusted source would allow arbitrary code execution.
- Fix: Use `numpy.savez` for the numeric arrays and JSON for the scalar hyperparameters, or validate the source before loading.

---

## Performance Bottlenecks

**Dense embedding calls are single-item at query time:**
- `DenseIndex.search()` in `rag/index_dense.py` line 136 calls `_embed_batch([query])` — a single-item batch — for every query. When `extract_state()` probes both BM25 and dense for state extraction (called once per item per precompute run), this adds one additional single-item embed call per item outside the main reward loop, which is acceptable. However, if state extraction is called at every training step in an online setting it would be very expensive.

**LinUCB matrix solves in Python loop:**
- Already noted above. At K=60 arms this takes microseconds. For sweeps with larger action spaces this will become the training bottleneck.

**No batching in precompute across items:**
- Each item is processed serially within a SLURM task (`precompute_rewards.py` lines 213–222). Within a task there is no concurrency across actions. With 60 actions × ~30s each = ~30 min per 10-item task, this is already near the expected 4-hour wall time, leaving little margin.

---

## Missing Documentation

**No `requirements.txt` for local development:**
- `requirements_ccr.txt` targets the CCR HPC cluster. There is no general `requirements.txt`, `setup.py`, `pyproject.toml`, or `environment.yml` for local setup. New contributors cannot reproduce the environment without reading the README closely.

**No docstring on `env/rag_env.py`'s module level:**
- `env/rag_env.py` has a class docstring but no module-level docstring explaining when/why to use this class vs. the offline precompute path.

**`rag/reporting.py` not documented or connected:**
- `rag/reporting.py` exists but is not imported from any other module in the training pipeline. Its purpose and usage context are unclear.

**No documentation on how to run the hyperparameter sweep:**
- `slurm/2_train_bandit.sh` mentions `sbatch --array=0-7` for a sweep, but the README does not cover this workflow. The mapping from `TASK_ID` to `(alpha, v)` combinations is undocumented outside the script itself.

---

## Incomplete Implementations

**`nq` is constrained to `{1, 3}` with a hard validation error:**
- `rag/config.py` line 47: `if self.nq not in (1, 3): raise ValueError(...)`. This restriction is documented as "for checkpoint baseline," implying it is a temporary constraint. The action space in `env/action_space.py` also only generates `nq in [1, 3]`. Supporting arbitrary `nq` values would require removing this guard and expanding the action grid.

**`allow_retrieve_zero` config flag is never exercised:**
- `RAGConfig.allow_retrieve_zero` is defined with a comment "RL later may set to 0" but `build_action_space()` in `env/action_space.py` never generates an action that sets `k_final=0`. The zero-retrieval path is unreachable in the current system.

**`RAGEnv.step()` always returns `done=True`:**
- `env/rag_env.py` line 131: `done = True` always. The class is designed for a 1-step environment but returns `next_state` anyway (line 132). For multi-step or trajectory-based RL extensions this design would need to be revisited.

**Multi-query (`nq=3`) fallback is fragile:**
- `rag/rewrite.py` lines 120–133: When the LLM fails to return valid JSON for multi-query generation, the fallback duplicates the base query `nq` times (line 133: `return [retrieval_query.strip()] * nq`). This silently degrades the retrieval quality without logging a warning.

---

## Gaps & Unknowns

- **Ollama model availability on CCR nodes:** The scripts assume `llama3.2:3b` and `nomic-embed-text` are pre-pulled in `$OLLAMA_MODELS`. There is no script to verify or pull these models, and the README covers it only briefly.
- **Port collision risk:** On a shared HPC node, a hardcoded port of 11434 could conflict with another user's Ollama instance. This was not tested.
- **Cache invalidation correctness:** The embedding cache key in `DenseIndex._cache_key()` hashes only the first 200 characters of each chunk's text. A chunk edit beyond character 200 would not invalidate the cache, silently serving stale embeddings.
- **`rag/reporting.py` purpose:** This file exists but is not used anywhere in the pipeline. It may be dead code from an earlier iteration or intended for future use — cannot determine from static analysis alone.
- **Notebook state:** `notebooks/final_project_checkpoint_jeetkava_devchira_vanshpra_FIXED.ipynb` contains output cells; it is unclear whether those outputs reflect the current codebase or a prior iteration.
