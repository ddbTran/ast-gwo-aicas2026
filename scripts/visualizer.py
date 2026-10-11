#!/usr/bin/env python3

import os
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "legend.title_fontsize": 12,
    "lines.linewidth": 2.0,
})

ROOT = Path("results/runs/ast_gwo/conference")

BENCHMARKS = [
    "adder",
    "multiplier",
]

def summarize_results():
    print("\n================ Summary ================\n")

    for benchmark in BENCHMARKS:

        print(f"\n===== {benchmark.upper()} =====")
        print(
            f"{'k_range':<12}"
            f"{'QoR Mean':>10}"
            f"{'QoR Std':>10}"
            f"{'Best':>10}"
            f"{'Worst':>10}"
            f"{'Div Mean':>12}"
            f"{'Div Std':>10}"
            f"{'Div Max':>10}"
            f"{'Div Min':>10}"
        )

        for exp in sorted(ROOT.glob("lut_*")):

            label = exp.name.split("k_range_")[-1].replace("_", "-")

            qors = []
            divs = []

            for seed in range(5):

                run_dir = exp / benchmark / str(seed)

                # QoR
                df = pd.read_csv(run_dir / "iter.log")
                qors.append(df["best_qor"].iloc[-1])

                # Diversity
                div_curve = load_diversity(run_dir)
                divs.append(div_curve[-1])      # diversity ở iteration cuối

            qors = np.asarray(qors)
            divs = np.asarray(divs)

            print(
                f"{label:<12}"
                f"{qors.mean():10.6f}"
                f"{qors.std(ddof=1):10.6f}"
                f"{qors.min():10.6f}"
                f"{qors.max():10.6f}"
                f"{divs.mean():12.4f}"
                f"{divs.std(ddof=1):10.4f}"
                f"{divs.max():10.4f}"
                f"{divs.min():10.4f}"
            )

def pairwise_diversity(bitstrings):
    """
    Mean normalized pairwise Hamming distance.
    """

    arr = np.array(
        [list(map(int, s)) for s in bitstrings],
        dtype=np.uint8,
    )

    n = len(arr)

    if n < 2:
        return 0

    dist = []

    for i, j in combinations(range(n), 2):
        d = np.count_nonzero(arr[i] != arr[j])
        dist.append(d)

    return np.mean(dist)


def load_convergence(run_dir):

    df = pd.read_csv(run_dir / "iter.log")

    return df["best_qor"].to_numpy()


def load_diversity(run_dir):

#    pop = pd.read_csv(run_dir / "pop.log")

    pop = pd.read_csv(
        run_dir / "pop.log",
        dtype={
            "iter": np.int32,
            "idx": np.int32,
            "bits": str,
            "qor": np.float64,
        },
    )

    diversity = []

    for it in sorted(pop["iter"].unique()):

        bits = pop.loc[
            pop["iter"] == it,
            "bits"
        ].tolist()

        diversity.append(pairwise_diversity(bits))

    return np.asarray(diversity)


def plot_convergence(benchmark):

    plt.figure(figsize=(8,5))

    for exp in sorted(ROOT.glob("lut_*")):

        label = exp.name.split("k_range_")[-1].replace("_", "–")

        runs = []

        for seed in range(5):
            run_dir = exp / benchmark / str(seed)
            runs.append(load_convergence(run_dir))

        runs = np.stack(runs)

        mean = runs.mean(axis=0)
        std = runs.std(axis=0)

        x = np.arange(len(mean))

        plt.plot(x, mean, label=label)
        plt.fill_between(
            x,
            mean-std,
            mean+std,
            alpha=0.2
        )

    plt.xlabel("Iteration")
    plt.ylabel("Best QoR")
    plt.title(f"{benchmark}: Convergence")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

#    plt.savefig(f"{benchmark}_convergence.pdf")
#    plt.savefig(f"{benchmark}_convergence.png", dpi=300)

#    plt.close()

    plt.tight_layout()
#    plt.show()
#    plt.close()

def plot_diversity(benchmark):

    plt.figure(figsize=(8,5))

    for exp in sorted(ROOT.glob("lut_*")):

        label = exp.name.split("k_range_")[-1].replace("_", "–")

        runs = []

        for seed in range(5):
            run_dir = exp / benchmark / str(seed)
            runs.append(load_diversity(run_dir))

        runs = np.stack(runs)

        mean = runs.mean(axis=0)
        std = runs.std(axis=0)

        x = np.arange(len(mean))

        plt.plot(x, mean, label=label)

        plt.fill_between(
            x,
            mean-std,
            mean+std,
            alpha=0.2
        )

    plt.xlabel("Iteration")
    plt.ylabel("Pairwise Hamming Distance")
    plt.title(f"{benchmark}: Population Diversity")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

#    plt.savefig(f"{benchmark}_diversity.pdf")
#    plt.savefig(f"{benchmark}_diversity.png", dpi=300)

#    plt.close()

    plt.tight_layout()
#    plt.show()
#    plt.close()

def plot_benchmark(benchmark, ax_conv, ax_div):

    # ================= Convergence =================
    for exp in sorted(ROOT.glob("lut_*")):

#        label = exp.name.split("k_range_")[-1].replace("_", "–")
        range_str = exp.name.split("k_range_")[-1]
        lo, hi = range_str.split("_")
        label = lo if lo == hi else f"{lo}–{hi}"
        
        runs = []

        for seed in range(5):
            run_dir = exp / benchmark / str(seed)

            df = pd.read_csv(run_dir / "iter.log")
            runs.append(df["best_improve"].to_numpy())

        runs = np.stack(runs)

        mean = runs.mean(axis=0)
        std = runs.std(axis=0)

        iters = np.arange(len(mean))

        ax_conv.plot(iters, mean, linewidth=2, label=label)
        ax_conv.fill_between(
            iters,
            mean - std,
            mean + std,
            alpha=0.2,
        )

    ax_conv.set_title(f"{benchmark.capitalize()} - Convergence")
    ax_conv.set_xlabel("Iterations")
    ax_conv.set_ylabel("QoR Improvements (%)")
    ax_conv.grid(alpha=0.3)

    # ================= Diversity =================
    for exp in sorted(ROOT.glob("lut_*")):

#        label = exp.name.split("k_range_")[-1].replace("_", "–")
        range_str = exp.name.split("k_range_")[-1]
        lo, hi = range_str.split("_")
        label = lo if lo == hi else f"{lo}–{hi}"

        runs = []

        for seed in range(5):
            run_dir = exp / benchmark / str(seed)
            runs.append(load_diversity(run_dir))

        runs = np.stack(runs)

        mean = runs.mean(axis=0)
        std = runs.std(axis=0)

        iters = np.arange(len(mean))

        ax_div.plot(iters, mean, linewidth=2, label=label)
        ax_div.fill_between(
            iters,
            mean - std,
            mean + std,
            alpha=0.2,
        )

    ax_div.set_title(f"{benchmark.capitalize()} - Diversity")
    ax_div.set_xlabel("Iterations")
    ax_div.set_ylabel("Pairwise Hamming Distance")
    ax_div.grid(alpha=0.3)

def plot_all():

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(8, 8),
    )

    plot_benchmark("adder", axes[0, 0], axes[0, 1])
    plot_benchmark("multiplier", axes[1, 0], axes[1, 1])

    handles, labels = axes[0, 0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=len(labels),
        frameon=True,
        fancybox=True,      # khung vuông
        edgecolor="black",
        facecolor="white",
        framealpha=1.0,
        columnspacing=1.0,
        handlelength=2.0,
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.99,
        bottom=0.08,
        top=0.90,
        wspace=0.20,
        hspace=0.30,
    )

def main():

    summarize_results()

    plot_all()

    plt.show()

    print("Done.")

if __name__ == "__main__":
    main()
