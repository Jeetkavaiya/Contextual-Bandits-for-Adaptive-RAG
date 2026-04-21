"""
Analyse and plot bandit training results.

Run locally after downloading results/bandit_results.npz from CCR.

Usage
-----
python analyze_results.py --results results/bandit_results.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))


def smooth(arr: np.ndarray, window: int = 50) -> np.ndarray:
    """Uniform moving average."""
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def load_results(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    # Reconstruct per-algo dicts
    algos = {}
    for key in data.files:
        if "/" in key:
            algo, metric = key.split("/", 1)
            algos.setdefault(algo, {})[metric] = data[key]
    return algos


def print_summary(algos: dict) -> None:
    print(f"\n{'ALGORITHM':<22} {'MEAN_R':>10} {'FINAL_CUM_REG':>15} {'ARMS%':>8}")
    print("-" * 60)
    rows = []
    for name, hist in algos.items():
        ar = hist["agent_rewards"]
        cr = hist["cum_regrets"]
        n_arms_total = int(hist["chosen_arms"].max()) + 1  # approx
        n_tried = len(np.unique(hist["chosen_arms"]))
        rows.append((name, ar.mean(), cr[-1], n_tried / n_arms_total))

    for name, mr, cr, af in sorted(rows, key=lambda x: -x[1]):
        print(f"{name:<22} {mr:>10.4f} {cr:>15.2f} {af*100:>7.1f}%")


def plot_results(algos: dict, out_dir: str, window: int = 100) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots.")
        return

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    colors = {"linucb": "C0", "thompson": "C1",
              "ucb1": "C2", "epsilon_greedy": "C3"}

    # ── 1. Cumulative Regret ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, hist in algos.items():
        ax.plot(hist["cum_regrets"],
                label=name, color=colors.get(name), linewidth=1.5)
    ax.set_xlabel("Episode"); ax.set_ylabel("Cumulative Regret")
    ax.set_title("Cumulative Regret — all algorithms")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out / "cumulative_regret.png"), dpi=150)
    plt.close(fig)

    # ── 2. Smoothed per-step reward ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, hist in algos.items():
        s = smooth(hist["agent_rewards"], window)
        ax.plot(s, label=name, color=colors.get(name), linewidth=1.5)
    ax.set_xlabel("Episode"); ax.set_ylabel(f"Reward (MA-{window})")
    ax.set_title("Smoothed Agent Reward")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out / "smoothed_reward.png"), dpi=150)
    plt.close(fig)

    # ── 3. Action frequency heatmap for contextual bandits ───────────────────
    for name in ("linucb", "thompson"):
        if name not in algos:
            continue
        arms = algos[name]["chosen_arms"]
        K = int(arms.max()) + 1
        counts = np.bincount(arms, minlength=K)
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.bar(range(K), counts, color=colors.get(name, "C0"), alpha=0.8)
        ax.set_xlabel("Action index"); ax.set_ylabel("Times selected")
        ax.set_title(f"{name} — action selection frequency")
        fig.tight_layout()
        fig.savefig(str(out / f"{name}_action_freq.png"), dpi=150)
        plt.close(fig)

    print(f"\nPlots saved to {out}/")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results/bandit_results.npz")
    p.add_argument("--plot_dir", default="results/plots")
    p.add_argument("--window", type=int, default=100,
                   help="Smoothing window for reward plot")
    args = p.parse_args()

    path = REPO_ROOT / args.results
    if not path.exists():
        path = Path(args.results)
    if not path.exists():
        sys.exit(f"[ERROR] {path} not found")

    algos = load_results(str(path))
    print(f"Loaded results for: {list(algos.keys())}")
    print_summary(algos)
    plot_results(algos, args.plot_dir, args.window)


if __name__ == "__main__":
    main()
