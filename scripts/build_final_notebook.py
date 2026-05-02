"""Build the final submission notebook with executed outputs.

Run from repo root:
    python scripts/build_final_notebook.py

Reads saved artifacts from results/compare_fair/ and results/task_0/, builds a
clean Jupyter notebook, executes it with nbclient so all outputs are baked in,
and writes notebooks/final_project_jeetkava_devchira_vanshpra.ipynb.
"""
from __future__ import annotations

import os
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "notebooks" / "final_project_jeetkava_devchira_vanshpra.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text)


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()

    nb.cells.append(md(
        "# RL-RAG: Final Project Notebook\n"
        "\n"
        "**Team 4 — Jeet Kavaiya, Dev Desai, Vansh Thakkar**\n"
        "\n"
        "**CSE 546 (Reinforcement Learning), University at Buffalo, Spring 2026**\n"
        "\n"
        "This notebook loads the *saved* artifacts from training and the final "
        "apples to apples comparison, and reproduces the key tables and plots "
        "shown in the report. It does **not** re-run RAG or bandit training, "
        "those are expensive and were run separately:\n"
        "\n"
        "* **Bandit training:** `train_bandit.py` (5000 episodes, all 4 algorithms) "
        "  produced `results/task_0/bandit_results.npz`.\n"
        "* **Live comparison:** `run_compare.py` (10 queries, fixed baseline vs. "
        "  trained Thompson Sampling policy) produced the CSVs and JSON in "
        "  `results/compare_fair/`.\n"
    ))

    nb.cells.append(code(
        "import json\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "REPO = Path('..').resolve() if Path('..').name else Path('.').resolve()\n"
        "if not (REPO / 'results').exists():\n"
        "    REPO = Path('.').resolve()\n"
        "    if not (REPO / 'results').exists():\n"
        "        REPO = REPO.parent\n"
        "print('repo root:', REPO)\n"
        "print('results dir exists:', (REPO / 'results').exists())\n"
    ))

    nb.cells.append(md(
        "## 1. Per-query head-to-head\n"
        "\n"
        "We picked 10 diverse queries (6 in-domain, 4 out-of-domain). For each "
        "query, the fixed baseline RAG and the trained Thompson Sampling policy "
        "were run on the same warm Ollama instance back to back. The table "
        "below joins both runs per `qid`.\n"
    ))

    nb.cells.append(code(
        "base = pd.read_csv(REPO / 'results' / 'compare_fair' / 'baseline_metrics.csv')\n"
        "rl   = pd.read_csv(REPO / 'results' / 'compare_fair' / 'rl_thompson_metrics.csv')\n"
        "\n"
        "join = (\n"
        "    base.merge(rl, on=['qid', 'domain'], suffixes=('_base', '_rl'))\n"
        "        [[\n"
        "            'qid', 'domain',\n"
        "            'correctness_base', 'correctness_rl',\n"
        "            'faithfulness_base', 'faithfulness_rl',\n"
        "            'evidence_recall_at_k_base', 'evidence_recall_at_k_rl',\n"
        "            'total_time_s_base', 'total_time_s_rl',\n"
        "            'total_tokens_base', 'total_tokens_rl',\n"
        "            'reward_base', 'reward_rl',\n"
        "        ]]\n"
        ")\n"
        "join.round(3)\n"
    ))

    nb.cells.append(md(
        "## 2. Aggregate comparison and per-metric winners\n"
        "\n"
        "Means over the 10 queries plus per-query head to head wins. Latency, "
        "tokens (median), evidence recall and reward go to RL on a per query "
        "basis, which matches the headline in the report.\n"
    ))

    nb.cells.append(code(
        "summary = json.loads((REPO / 'results' / 'compare_fair' / 'summary_thompson.json').read_text())\n"
        "\n"
        "agg_rows = [\n"
        "    ('Correctness (0-2)',       summary['baseline_all']['correctness'],         summary['rl_all']['correctness']),\n"
        "    ('Faithfulness (0-1)',      summary['baseline_all']['faithfulness'],        summary['rl_all']['faithfulness']),\n"
        "    ('Evidence recall@k',       summary['baseline_all']['evidence_recall_at_k'], summary['rl_all']['evidence_recall_at_k']),\n"
        "    ('Total latency (s)',       summary['baseline_all']['total_time_s'],         summary['rl_all']['total_time_s']),\n"
        "    ('Total tokens (mean)',     summary['baseline_all']['total_tokens'],         summary['rl_all']['total_tokens']),\n"
        "    ('Reward (scalar)',         summary['baseline_all']['reward'],              summary['rl_all']['reward']),\n"
        "]\n"
        "agg = pd.DataFrame(agg_rows, columns=['metric', 'baseline_mean', 'rl_mean'])\n"
        "agg['delta_rl_minus_baseline'] = agg['rl_mean'] - agg['baseline_mean']\n"
        "print('--- aggregate means ---')\n"
        "print(agg.round(3).to_string(index=False))\n"
    ))

    nb.cells.append(code(
        "def per_query_winner(col_base: str, col_rl: str, *, higher_is_better: bool):\n"
        "    b = base[col_base].astype(float).fillna(np.nan).values\n"
        "    r = rl[col_rl].astype(float).fillna(np.nan).values\n"
        "    mask = ~(np.isnan(b) | np.isnan(r))\n"
        "    b, r = b[mask], r[mask]\n"
        "    if higher_is_better:\n"
        "        return int((r > b).sum()), int((b > r).sum()), int((r == b).sum()), len(b)\n"
        "    return int((r < b).sum()), int((b < r).sum()), int((r == b).sum()), len(b)\n"
        "\n"
        "rows = []\n"
        "rows.append(('Correctness',        *per_query_winner('correctness',        'correctness',        higher_is_better=True)))\n"
        "rows.append(('Faithfulness',       *per_query_winner('faithfulness',       'faithfulness',       higher_is_better=True)))\n"
        "rows.append(('Evidence recall@k',  *per_query_winner('evidence_recall_at_k','evidence_recall_at_k', higher_is_better=True)))\n"
        "rows.append(('Total latency (s)',  *per_query_winner('total_time_s',       'total_time_s',       higher_is_better=False)))\n"
        "rows.append(('Total tokens',       *per_query_winner('total_tokens',       'total_tokens',       higher_is_better=False)))\n"
        "rows.append(('Reward (scalar)',    *per_query_winner('reward',             'reward',             higher_is_better=True)))\n"
        "\n"
        "h2h = pd.DataFrame(rows, columns=['metric', 'rl_wins', 'baseline_wins', 'ties', 'n_compared'])\n"
        "print('--- per-query head-to-head wins (10 queries) ---')\n"
        "print(h2h.to_string(index=False))\n"
    ))

    nb.cells.append(md(
        "## 3. Bandit training, cumulative regret\n"
        "\n"
        "All four bandits replay the same 5000 episodes drawn from the precomputed "
        "reward table. Lower cumulative regret is better. Linear Thompson "
        "Sampling has the lowest final regret, followed by LinUCB, "
        "epsilon-greedy, and UCB1. Numbers come straight from "
        "`results/task_0/bandit_summary.json`.\n"
    ))

    nb.cells.append(code(
        "with open(REPO / 'results' / 'task_0' / 'bandit_summary.json') as f:\n"
        "    bs = json.load(f)\n"
        "summary_df = pd.DataFrame(bs)[['algo', 'mean_reward', 'mean_reward_last10p', 'final_cum_regret', 'arms_explored_frac']]\n"
        "print(summary_df.round(3).to_string(index=False))\n"
    ))

    nb.cells.append(code(
        "data = np.load(REPO / 'results' / 'task_0' / 'bandit_results.npz', allow_pickle=True)\n"
        "print('keys:', list(data.keys())[:20], '...')\n"
        "\n"
        "algos = ['epsilon_greedy', 'ucb1', 'linucb', 'thompson']\n"
        "fig, ax = plt.subplots(figsize=(7, 4))\n"
        "for algo in algos:\n"
        "    key = f'{algo}/cum_regrets'\n"
        "    if key in data.files:\n"
        "        ax.plot(data[key], label=algo, linewidth=1.6)\n"
        "ax.set_xlabel('Episode')\n"
        "ax.set_ylabel('Cumulative regret')\n"
        "ax.set_title('Cumulative regret over 5000 training episodes')\n"
        "ax.grid(True, alpha=0.3)\n"
        "ax.legend()\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
    ))

    nb.cells.append(md(
        "## 4. Action selection by Thompson during training\n"
        "\n"
        "Histogram of the 54 arms picked by the Thompson sampler over 5000 "
        "training episodes. Note the policy concentrates mass on a handful of "
        "configurations rather than collapsing to a single arm.\n"
    ))

    nb.cells.append(code(
        "thompson_arms = data['thompson/chosen_arms']\n"
        "n_arms = int(thompson_arms.max()) + 1\n"
        "counts = np.bincount(thompson_arms, minlength=n_arms)\n"
        "fig, ax = plt.subplots(figsize=(9, 3.5))\n"
        "ax.bar(np.arange(n_arms), counts, width=0.85, color='#2D75D6', alpha=0.85)\n"
        "ax.set_xlabel('Arm index (0..53)')\n"
        "ax.set_ylabel('Times selected')\n"
        "ax.set_title('Thompson Sampling arm selection over 5000 episodes')\n"
        "ax.grid(True, axis='y', alpha=0.3)\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
        "\n"
        "top5 = np.argsort(counts)[::-1][:5]\n"
        "print('top 5 most-picked arms by Thompson:', [(int(a), int(counts[a])) for a in top5])\n"
    ))

    nb.cells.append(md(
        "## 5. Per-query RL action choices on the comparison set\n"
        "\n"
        "For each of the 10 evaluation queries the trained Thompson policy used "
        "the state vector to pick one of the 54 RAG configurations. The table "
        "below shows the chosen arm and its decoded settings.\n"
    ))

    nb.cells.append(code(
        "rl_choices = rl[[\n"
        "    'qid', 'domain', 'arm_idx',\n"
        "    'rewrite_enabled', 'nq', 'rerank_enabled', 'k_final', 'x_bm25', 'y_dense',\n"
        "    'total_time_s', 'total_tokens', 'reward'\n"
        "]].copy()\n"
        "rl_choices.round(3)\n"
    ))

    nb.cells.append(md(
        "## 6. Conclusion\n"
        "\n"
        "On the same 10 query evaluation set, the trained Thompson Sampling "
        "policy beats the fixed advanced RAG baseline on **4 of 6 metrics by "
        "per-query head to head wins**: latency, total tokens (median), "
        "evidence recall at k, and scalar reward. Quality (correctness and "
        "faithfulness) ties on most queries with one outlier. The bandit "
        "learning is real (see cumulative regret curves and the action "
        "distribution) and the resulting policy is cheap at inference time, "
        "which matches the project goal of getting same or better quality at "
        "lower cost. Full discussion is in the report.\n"
    ))

    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python"}
    return nb


def main() -> None:
    nb = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"executing notebook ({len(nb.cells)} cells)...")
    client = NotebookClient(nb, timeout=180, kernel_name="python3")
    client.execute(cwd=str(OUT.parent))
    nbf.write(nb, str(OUT))
    print(f"wrote {OUT}")
    print(f"size: {OUT.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
