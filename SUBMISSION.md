# Final Submission — Team 4

**CSE 546, Reinforcement Learning, Spring 2026, University at Buffalo**

**Members:** Jeet Kavaiya · Dev Desai · Vansh Thakkar

**GitHub repository:** https://github.com/ub-cse546-s26/final-project-team_4

**GitHub Project board (Kanban, To Do / In Progress / Done):** https://github.com/orgs/ub-cse546-s26/projects/21/views/1

## What to grade

| Item | Path |
|---|---|
| Final report (PDF) | `final_project_jeetkava_devchira_vanshpra.pdf` |
| Final notebook with results saved | `final_project_jeetkava_devchira_vanshpra.ipynb` |
| Report LaTeX source | `report/final_project_jeetkava_devchira_vanshpra.tex` |
| Architecture diagrams (Mermaid + PNG) | `report/diagrams/` |
| Bandit + RL environment + RAG source code | `bandits/`, `env/`, `rag/` |
| Top-level scripts | `precompute_rewards.py`, `train_bandit.py`, `analyze_results.py`, `run_compare.py` |
| Saved bandit training data (5000 episodes, 4 algorithms) | `results/task_0/bandit_results.npz` + `bandit_summary.json` |
| Final apples-to-apples comparison artifacts (10 queries) | `results/compare_fair/` |
| SLURM scripts used on UB CCR | `slurm/` |
| Earlier checkpoint notebook (baseline pipeline trace) | `notebooks/final_project_checkpoint_jeetkava_devchira_vanshpra_FIXED.ipynb` |
| Project README with setup, structure, and results | `README.md` |
| GitHub Project board (Kanban) screenshots | `screenshots/` |

## Headline result

The trained Linear Thompson Sampling policy was compared to the fixed advanced RAG baseline on the same 10 evaluation queries (6 in-domain, 4 out-of-domain), running on the same warm Ollama instance. Per-query head-to-head wins:

| Metric | RL wins | Baseline wins | Ties |
|---|---|---|---|
| Correctness (0–2) | 0 | 1 | 9 |
| Faithfulness (0–1) | 0 | 1 | 9 |
| Evidence recall@k | 1 | 1 | 4 |
| **Total latency (s)** | **9** | **1** | **0** |
| **Total tokens** | **6** | **4** | **0** |
| **Reward (scalar)** | **9** | **1** | **0** |

Quality stays within noise, cost goes down a lot, scalar reward is higher 9 out of 10 times. Full numbers, plots, and discussion are in the report and notebook.

## Reproducibility

To re-run the apples-to-apples comparison locally (Ollama with `llama3.2:3b` and `nomic-embed-text` must be running):

```bash
python run_compare.py --bandit thompson \
    --evalset data/evalset_100_gold.jsonl \
    --ckpt_dir results/task_0 \
    --out_dir results/compare_fair
```

To rebuild the final notebook with executed outputs from the saved artifacts:

```bash
python scripts/build_final_notebook.py
```

The bandit training itself (Phase 2 on UB CCR) is replayed via `train_bandit.py` from the precomputed reward table; see `README.md` and `slurm/` for the full HPC pipeline.
