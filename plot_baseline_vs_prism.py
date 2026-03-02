import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot baseline vs contextual benchmark results.")
    parser.add_argument(
        "--csv",
        default="/data/kell8360/results_baseline_vs_prism.csv",
        help="Path to results_baseline_vs_prism.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="/data/kell8360/plots_baseline_vs_prism",
        help="Directory to write plots.",
    )
    return parser.parse_args()


def accuracy(rows: List[Dict[str, str]]) -> float:
    if not rows:
        return 0.0
    return sum(int(r["correct"]) for r in rows) / len(rows)


def group_accuracy(rows: List[Dict[str, str]], field: str, groups: List[str]) -> Dict[str, Tuple[int, int, float]]:
    counts = defaultdict(int)
    correct = defaultdict(int)
    for r in rows:
        value = (r.get(field) or "").strip().lower()
        if value in groups:
            counts[value] += 1
            correct[value] += int(r["correct"])
    out: Dict[str, Tuple[int, int, float]] = {}
    for g in groups:
        n = counts[g]
        c = correct[g]
        out[g] = (c, n, (c / n) if n else 0.0)
    return out


def bar_plot(labels: List[str], values: List[float], title: str, ylabel: str, outpath: str) -> None:
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values)
    plt.ylim(0, 1.0)
    plt.ylabel(ylabel)
    plt.title(title)
    for b, v in zip(bars, values):
        plt.text(b.get_x() + b.get_width() / 2, min(v + 0.02, 0.98), f"{v:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    baseline_rows = [r for r in rows if r.get("condition") == "baseline"]
    contextual_rows = [r for r in rows if r.get("condition") == "contextual"]

    baseline_acc = accuracy(baseline_rows)
    contextual_acc = accuracy(contextual_rows)

    race_stats = group_accuracy(contextual_rows, "race", ["black", "white"])
    gender_stats = group_accuracy(contextual_rows, "gender", ["male", "female"])
    inter_stats = group_accuracy(
        contextual_rows,
        "intersection",
        ["black|male", "black|female", "white|male", "white|female"],
    )

    bar_plot(
        labels=["Baseline", "Contextual"],
        values=[baseline_acc, contextual_acc],
        title="Overall Accuracy: Baseline vs Contextual",
        ylabel="Accuracy",
        outpath=os.path.join(args.output_dir, "overall_comparison.png"),
    )

    bar_plot(
        labels=["Black", "White", "Male", "Female"],
        values=[
            race_stats["black"][2],
            race_stats["white"][2],
            gender_stats["male"][2],
            gender_stats["female"][2],
        ],
        title="Contextual Accuracy by Race and Gender",
        ylabel="Accuracy",
        outpath=os.path.join(args.output_dir, "race_gender_comparison.png"),
    )

    bar_plot(
        labels=["Black Male", "Black Female", "White Male", "White Female"],
        values=[
            inter_stats["black|male"][2],
            inter_stats["black|female"][2],
            inter_stats["white|male"][2],
            inter_stats["white|female"][2],
        ],
        title="Contextual Accuracy by Intersectional Group",
        ylabel="Accuracy",
        outpath=os.path.join(args.output_dir, "intersectional_comparison.png"),
    )

    print(f"Wrote plots to: {args.output_dir}")


if __name__ == "__main__":
    main()
