"""
Train + evaluate lerobot DiffusionPolicy baseline across demo counts.

Folder layout:
    experiments/
        demos_50/
            policy.pth
            loss.png
        demos_100/
            ...
    baseline_results.json
    baseline_curve.png

Usage:
    python run_baseline.py
"""

import json
import os

from train_lerobot import train
from eval_lerobot import evaluate, load_seeds

DEMO_COUNTS = [50, 100, 200, 500, 996]
EXP_ROOT = "experiments"
RESULTS_FILE = "baseline_results.json"
PLOT_FILE = "baseline_curve.png"

# Steps equalized by dataset size: aim for ~300 passes through each dataset.
# ~120 samples/demo on average, batch=256 → steps ≈ (n * 120 * 300) / 256
def num_steps_for(n: int) -> int:
    raw = int(n * 120 * 300 / 256)
    return max(raw, 50_000)   # floor at 50k even for tiny sets


def exp_dir(n: int) -> str:
    return os.path.join(EXP_ROOT, f"demos_{n}")


def plot_results(results: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    demos = sorted(int(k) for k in results)
    rates = [results[str(k)] * 100 for k in demos]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(demos, rates, marker="o", linewidth=2, color="#2196F3", label="Diffusion Policy (lerobot)")
    ax.axhline(80, color="gray", linestyle="--", linewidth=1, label="80% target")

    for x, y in zip(demos, rates):
        ax.annotate(f"{y:.0f}%", (x, y), textcoords="offset points", xytext=(0, 8), ha="center")

    ax.set_xlabel("Number of demos")
    ax.set_ylabel("Success rate (%)")
    ax.set_title("PegInsertionSide-v1 — Diffusion Policy baseline")
    ax.set_ylim(0, 105)
    ax.set_xscale("log")
    ax.set_xticks(demos)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_FILE, dpi=150)
    plt.close(fig)
    print(f"Curve → {PLOT_FILE}")


def main():
    os.makedirs(EXP_ROOT, exist_ok=True)

    results = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)

    for n in DEMO_COUNTS:
        key = str(n)
        d = exp_dir(n)
        ckpt = os.path.join(d, "policy.pth")

        # --- train ---
        if not os.path.exists(ckpt):
            steps = num_steps_for(n)
            print(f"\n=== Training demos={n}  steps={steps} ===")
            train(
                num_demos=n,
                exp_name="policy",
                ckpt_dir=d,
                num_steps=steps,
                batch_size=256,
                lr=1e-4,
            )
        else:
            print(f"[skip train] demos={n} — {ckpt} exists")

        # --- evaluate ---
        if key not in results:
            eval_n = min(n, 100)
            seeds = load_seeds(eval_n)
            print(f"\n=== Evaluating demos={n}  episodes={eval_n} ===")
            rate = evaluate(ckpt, num_episodes=eval_n, seeds=seeds)
            results[key] = rate
            print(f"demos={n}  success={rate:.2%}")

            with open(RESULTS_FILE, "w") as f:
                json.dump(results, f, indent=2)

            if len(results) > 1:
                plot_results(results)
        else:
            print(f"[skip eval] demos={n} — already have result {results[key]:.2%}")

    print("\n=== Baseline Results ===")
    for k in sorted(results, key=int):
        print(f"  {int(k):>4} demos → {results[k]:.2%}")

    plot_results(results)


if __name__ == "__main__":
    main()
