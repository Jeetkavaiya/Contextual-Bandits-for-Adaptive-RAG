"""
Phase 2 — Train bandit algorithms on the pre-computed reward table.

This script is pure NumPy (no Ollama needed) and runs fast on CCR CPU nodes.

Training protocol
-----------------
Each *episode* (step):
  1. Sample a query at random (with replacement) from the reward table.
  2. Agent observes the state vector for that query.
  3. Agent selects an action (arm).
  4. Agent receives the pre-computed reward for (query, action).
  5. Agent updates its model.

We repeat for `--n_episodes` steps and record cumulative regret.

Oracle / optimal action
  For each query the *oracle* plays the action with highest reward.
  Regret_t = oracle_reward_t - agent_reward_t.
  Cumulative regret = sum of per-step regrets.

Algorithms trained
  - epsilon_greedy  (non-contextual baseline)
  - ucb1            (non-contextual baseline)
  - linucb          (contextual — main algorithm)
  - thompson        (contextual — Bayesian)

Output
  results/bandit_results.npz   — arrays for plotting
  results/bandit_summary.json  — final metrics table
  checkpoints/<algo>.pkl       — saved agent state

Usage
-----
# Basic run (uses results/reward_table.npz)
python train_bandit.py

# Full options
python train_bandit.py \\
    --reward_table results/reward_table.npz \\
    --n_episodes 5000 \\
    --seed 42 \\
    --alpha 1.0 \\
    --v 1.0 \\
    --epsilon 0.15 \\
    --ucb_c 1.0 \\
    --out_dir results \\
    --checkpoint_dir checkpoints
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from bandits import EpsilonGreedy, UCB1, LinUCBDisjoint, LinearThompsonSampling
from bandits.base import BaseBandit
from env.state_features import STATE_KEYS


# ---------------------------------------------------------------------------
# State normalisation (online StandardScaler, fit from table)
# ---------------------------------------------------------------------------

class StateScaler:
    """Fit a StandardScaler on the full state matrix, then transform on-line."""

    def __init__(self) -> None:
        self.mean_: np.ndarray = None  # type: ignore
        self.std_:  np.ndarray = None  # type: ignore

    def fit(self, X: np.ndarray) -> "StateScaler":
        self.mean_ = X.mean(axis=0)
        self.std_  = X.std(axis=0)
        self.std_[self.std_ < 1e-8] = 1.0   # avoid division by zero
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_bandit(
    agent: BaseBandit,
    states: np.ndarray,       # (N_items, d)
    rewards: np.ndarray,      # (N_items, K_actions)  NaN = failed run
    n_episodes: int,
    scaler: StateScaler,
    rng: np.random.RandomState,
    contextual: bool,
) -> Dict[str, np.ndarray]:
    """
    Run the bandit for n_episodes steps against the pre-computed reward table.

    Returns a dict of per-step arrays:
        chosen_arms   (n_episodes,)
        agent_rewards (n_episodes,)
        oracle_rewards(n_episodes,)
        regrets       (n_episodes,)
        cum_regrets   (n_episodes,)
        item_indices  (n_episodes,)
    """
    N, K = rewards.shape
    assert agent.n_arms == K, f"Agent has {agent.n_arms} arms but table has {K}"

    # Oracle reward for each item (best valid action, ignoring NaN)
    oracle_rewards = np.nanmax(rewards, axis=1)   # (N,)

    chosen_arms    = np.zeros(n_episodes, dtype=int)
    agent_rwds     = np.zeros(n_episodes, dtype=float)
    oracle_rwds    = np.zeros(n_episodes, dtype=float)
    item_indices   = np.zeros(n_episodes, dtype=int)

    for t in range(n_episodes):
        # Sample a query
        i = rng.randint(0, N)
        state_raw = states[i]                         # (d,)
        state_norm = scaler.transform(state_raw)       # (d,)

        # Agent selects arm
        context = state_norm if contextual else None
        arm = agent.select(context)

        # Reward: if NaN (failed run), substitute -1.0 as penalty
        r_raw = rewards[i, arm]
        reward = float(r_raw) if not np.isnan(r_raw) else -1.0

        # Update agent
        agent.update(arm, reward, context)

        chosen_arms[t]  = arm
        agent_rwds[t]   = reward
        oracle_rwds[t]  = float(oracle_rewards[i])
        item_indices[t] = i

    regrets     = oracle_rwds - agent_rwds
    cum_regrets = np.cumsum(regrets)

    return {
        "chosen_arms":    chosen_arms,
        "agent_rewards":  agent_rwds,
        "oracle_rewards": oracle_rwds,
        "regrets":        regrets,
        "cum_regrets":    cum_regrets,
        "item_indices":   item_indices,
    }


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def summarise(name: str, hist: Dict[str, np.ndarray], n_arms: int) -> dict:
    N = len(hist["regrets"])
    ar = hist["agent_rewards"]
    return {
        "algo":                name,
        "n_episodes":          N,
        "mean_reward":         float(ar.mean()),
        "std_reward":          float(ar.std()),
        "mean_reward_last10p": float(ar[int(0.9 * N):].mean()),
        "final_cum_regret":    float(hist["cum_regrets"][-1]),
        "mean_regret":         float(hist["regrets"].mean()),
        # Action diversity: fraction of arms tried at least once
        "arms_explored_frac":  float(np.unique(hist["chosen_arms"]).size / n_arms),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train bandit algorithms offline")
    p.add_argument("--reward_table",    default="results/reward_table.npz")
    p.add_argument("--n_episodes",      type=int,   default=5000)
    p.add_argument("--seed",            type=int,   default=42)
    # LinUCB
    p.add_argument("--alpha",           type=float, default=1.0,
                   help="LinUCB exploration coefficient")
    p.add_argument("--reg_lambda",      type=float, default=1.0,
                   help="Ridge regularisation for LinUCB / Thompson")
    # Thompson
    p.add_argument("--v",               type=float, default=1.0,
                   help="Thompson Sampling posterior variance scale")
    # ε-greedy
    p.add_argument("--epsilon",         type=float, default=0.15)
    p.add_argument("--epsilon_decay",   type=float, default=0.0,
                   help="Annealing rate for epsilon.  0 = fixed.")
    # UCB1
    p.add_argument("--ucb_c",           type=float, default=1.0)
    # Output
    p.add_argument("--out_dir",         default="results")
    p.add_argument("--checkpoint_dir",  default="checkpoints")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.RandomState(args.seed)

    # -----------------------------------------------------------------------
    # 1. Load reward table
    # -----------------------------------------------------------------------
    table_path = REPO_ROOT / args.reward_table
    if not table_path.exists():
        # Try relative
        table_path = Path(args.reward_table)
    if not table_path.exists():
        sys.exit(
            f"[ERROR] Reward table not found at {table_path}\n"
            "Run precompute_rewards.py first (see slurm/0_precompute.sh)"
        )

    data   = np.load(str(table_path), allow_pickle=True)
    states  = data["states"].astype(np.float32)   # (N, 11)
    rewards = data["rewards"].astype(np.float32)  # (N, K)
    ids     = data["ids"]

    N, K = rewards.shape
    d    = states.shape[1]

    print(f"Reward table: {N} items × {K} actions  (state_dim={d})", flush=True)
    print(f"STATE_KEYS  : {STATE_KEYS}", flush=True)
    valid_frac = np.sum(~np.isnan(rewards)) / rewards.size
    print(f"Valid entries: {valid_frac*100:.1f}%", flush=True)

    # Replace remaining NaN with -1 penalty in stats (keep raw NaN for oracle calc)
    # -----------------------------------------------------------------------
    # 2. Fit state scaler
    # -----------------------------------------------------------------------
    scaler = StateScaler().fit(states)

    # -----------------------------------------------------------------------
    # 3. Define agents
    # -----------------------------------------------------------------------
    agents: List[Tuple[str, BaseBandit, bool]] = [  # (name, agent, contextual)
        ("epsilon_greedy",  EpsilonGreedy(K, epsilon=args.epsilon,
                                          decay_rate=args.epsilon_decay), False),
        ("ucb1",            UCB1(K, c=args.ucb_c),                         False),
        ("linucb",          LinUCBDisjoint(K, d=d,
                                           alpha=args.alpha,
                                           reg_lambda=args.reg_lambda),    True),
        ("thompson",        LinearThompsonSampling(K, d=d,
                                                   v=args.v,
                                                   reg_lambda=args.reg_lambda), True),
    ]

    # -----------------------------------------------------------------------
    # 4. Train each agent
    # -----------------------------------------------------------------------
    out_dir  = REPO_ROOT / args.out_dir
    ckpt_dir = REPO_ROOT / args.checkpoint_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    all_histories: Dict[str, Dict] = {}
    summaries: List[dict] = []

    for name, agent, contextual in agents:
        print(f"\n{'='*60}", flush=True)
        print(f"Training {name}  (n_episodes={args.n_episodes})", flush=True)
        t0 = time.time()

        hist = train_bandit(
            agent      = agent,
            states     = states,
            rewards    = rewards,
            n_episodes = args.n_episodes,
            scaler     = scaler,
            rng        = rng,
            contextual = contextual,
        )

        elapsed = time.time() - t0
        s = summarise(name, hist, K)
        summaries.append(s)
        all_histories[name] = hist

        print(f"  elapsed          : {elapsed:.1f}s", flush=True)
        print(f"  mean reward      : {s['mean_reward']:.4f}", flush=True)
        print(f"  mean reward(last 10%) : {s['mean_reward_last10p']:.4f}", flush=True)
        print(f"  final cum regret : {s['final_cum_regret']:.2f}", flush=True)
        print(f"  arms explored    : {s['arms_explored_frac']*100:.1f}%", flush=True)

        # Save checkpoint
        ckpt_path = ckpt_dir / f"{name}.pkl"
        agent.save(str(ckpt_path))
        print(f"  checkpoint saved → {ckpt_path}", flush=True)

    # -----------------------------------------------------------------------
    # 5. Save results
    # -----------------------------------------------------------------------
    # NPZ with per-step arrays for each algorithm
    np_payload = {}
    for name, hist in all_histories.items():
        for key, arr in hist.items():
            np_payload[f"{name}/{key}"] = arr
    np_payload["state_mean"] = scaler.mean_
    np_payload["state_std"]  = scaler.std_

    results_file = out_dir / "bandit_results.npz"
    np.savez_compressed(str(results_file), **np_payload)
    print(f"\nResults → {results_file}", flush=True)

    # JSON summary
    summary_file = out_dir / "bandit_summary.json"
    summary_file.write_text(json.dumps(summaries, indent=2))
    print(f"Summary → {summary_file}", flush=True)

    # -----------------------------------------------------------------------
    # 6. Print final leaderboard
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print(f"{'ALGO':<20} {'MEAN_REWARD':>12} {'MEAN_R(LAST10%)':>16} {'CUM_REGRET':>12}")
    print("-"*60)
    for s in sorted(summaries, key=lambda x: -x["mean_reward_last10p"]):
        print(f"{s['algo']:<20} {s['mean_reward']:>12.4f} "
              f"{s['mean_reward_last10p']:>16.4f} "
              f"{s['final_cum_regret']:>12.2f}")
    print("="*60)


if __name__ == "__main__":
    main()
