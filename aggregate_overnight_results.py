import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict
from statistics import mean, pstdev
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate multi-seed baseline-vs-contextual benchmark outputs.")
    parser.add_argument(
        "--input-glob",
        default="/data/kell8360/overnight/results_seed_*.csv",
        help="Glob for per-seed result CSV files produced by benchmark_baseline_vs_prism.py",
    )
    parser.add_argument(
        "--output-dir",
        default="figures/overnight_aggregate",
        help="Directory for aggregate tables and figures.",
    )
    return parser.parse_args()


def wilson_ci(correct: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = correct / n
    denom = 1 + (z * z / n)
    center = (p + (z * z) / (2 * n)) / denom
    margin = (z / denom) * math.sqrt((p * (1 - p) / n) + ((z * z) / (4 * n * n)))
    return max(0.0, center - margin), min(1.0, center + margin)


def load_rows(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def acc(rows: List[Dict[str, str]]) -> Tuple[int, int, float]:
    n = len(rows)
    c = sum(int(r["correct"]) for r in rows)
    return c, n, (c / n) if n else 0.0


def grouped(rows: List[Dict[str, str]], field: str, allowed: List[str]) -> Dict[str, Tuple[int, int, float]]:
    counts = defaultdict(int)
    correct = defaultdict(int)
    for r in rows:
        value = (r.get(field) or "").strip().lower()
        if value in allowed:
            counts[value] += 1
            correct[value] += int(r["correct"])
    out: Dict[str, Tuple[int, int, float]] = {}
    for k in allowed:
        n = counts[k]
        c = correct[k]
        out[k] = (c, n, (c / n) if n else 0.0)
    return out


def bar_with_error(labels: List[str], values: List[float], yerr: List[float], title: str, outpath: str) -> None:
    plt.figure(figsize=(9, 5))
    bars = plt.bar(labels, values, yerr=yerr, capsize=5)
    plt.ylim(0, 1.0)
    plt.ylabel("Accuracy")
    plt.title(title)
    for b, v in zip(bars, values):
        plt.text(b.get_x() + b.get_width() / 2, min(v + 0.02, 0.98), f"{v:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    files = sorted(glob.glob(args.input_glob))
    if not files:
        raise FileNotFoundError(f"No files matched --input-glob: {args.input_glob}")

    seed_level_rows = []
    all_contextual: List[Dict[str, str]] = []
    all_baseline: List[Dict[str, str]] = []

    for path in files:
        rows = load_rows(path)
        baseline = [r for r in rows if r.get("condition") == "baseline"]
        contextual = [r for r in rows if r.get("condition") == "contextual"]
        bc, bn, ba = acc(baseline)
        cc, cn, ca = acc(contextual)
        seed_level_rows.append(
            {
                "file": path,
                "baseline_correct": bc,
                "baseline_n": bn,
                "baseline_accuracy": ba,
                "contextual_correct": cc,
                "contextual_n": cn,
                "contextual_accuracy": ca,
                "delta_contextual_minus_baseline": ca - ba,
            }
        )
        all_baseline.extend(baseline)
        all_contextual.extend(contextual)

    # pooled stats
    b_c, b_n, b_a = acc(all_baseline)
    c_c, c_n, c_a = acc(all_contextual)
    b_lo, b_hi = wilson_ci(b_c, b_n)
    c_lo, c_hi = wilson_ci(c_c, c_n)

    race_stats = grouped(all_contextual, "race", ["black", "white"])
    gender_stats = grouped(all_contextual, "gender", ["male", "female"])
    inter_stats = grouped(
        all_contextual,
        "intersection",
        ["black|male", "black|female", "white|male", "white|female"],
    )

    group_rows = []
    for name, (c, n, a) in {
        "black": race_stats["black"],
        "white": race_stats["white"],
        "male": gender_stats["male"],
        "female": gender_stats["female"],
        "black|male": inter_stats["black|male"],
        "black|female": inter_stats["black|female"],
        "white|male": inter_stats["white|male"],
        "white|female": inter_stats["white|female"],
    }.items():
        lo, hi = wilson_ci(c, n)
        group_rows.append(
            {
                "group": name,
                "correct": c,
                "n": n,
                "accuracy": a,
                "ci95_low": lo,
                "ci95_high": hi,
            }
        )

    # write tables
    seed_csv = os.path.join(args.output_dir, "seed_level_metrics.csv")
    with open(seed_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(seed_level_rows[0].keys()))
        writer.writeheader()
        writer.writerows(seed_level_rows)

    groups_csv = os.path.join(args.output_dir, "pooled_group_metrics.csv")
    with open(groups_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(group_rows[0].keys()))
        writer.writeheader()
        writer.writerows(group_rows)

    # figures
    baseline_accs = [r["baseline_accuracy"] for r in seed_level_rows]
    contextual_accs = [r["contextual_accuracy"] for r in seed_level_rows]
    base_mean = mean(baseline_accs)
    ctx_mean = mean(contextual_accs)
    base_std = pstdev(baseline_accs) if len(baseline_accs) > 1 else 0.0
    ctx_std = pstdev(contextual_accs) if len(contextual_accs) > 1 else 0.0

    bar_with_error(
        labels=["Baseline", "Contextual"],
        values=[base_mean, ctx_mean],
        yerr=[base_std, ctx_std],
        title="Seed-Level Mean Accuracy (Error Bars = Std Dev)",
        outpath=os.path.join(args.output_dir, "overall_seed_mean_std.png"),
    )

    # Pooled CI chart
    labels = ["Baseline", "Contextual"]
    values = [b_a, c_a]
    yerr = [max(b_a - b_lo, b_hi - b_a), max(c_a - c_lo, c_hi - c_a)]
    bar_with_error(
        labels=labels,
        values=values,
        yerr=yerr,
        title="Pooled Overall Accuracy (Error Bars = Wilson 95% CI)",
        outpath=os.path.join(args.output_dir, "overall_pooled_ci.png"),
    )

    inter_labels = ["Black Male", "Black Female", "White Male", "White Female"]
    inter_vals = [inter_stats["black|male"][2], inter_stats["black|female"][2], inter_stats["white|male"][2], inter_stats["white|female"][2]]
    inter_err = []
    for key in ["black|male", "black|female", "white|male", "white|female"]:
        c, n, a = inter_stats[key]
        lo, hi = wilson_ci(c, n)
        inter_err.append(max(a - lo, hi - a))
    bar_with_error(
        labels=inter_labels,
        values=inter_vals,
        yerr=inter_err,
        title="Contextual Intersectional Accuracy (Wilson 95% CI)",
        outpath=os.path.join(args.output_dir, "intersectional_pooled_ci.png"),
    )

    summary = {
        "input_files": files,
        "n_files": len(files),
        "pooled_overall": {
            "baseline": {"correct": b_c, "n": b_n, "accuracy": b_a, "ci95": [b_lo, b_hi]},
            "contextual": {"correct": c_c, "n": c_n, "accuracy": c_a, "ci95": [c_lo, c_hi]},
            "delta_contextual_minus_baseline": c_a - b_a,
        },
        "seed_level": {
            "baseline_mean": base_mean,
            "baseline_std": base_std,
            "contextual_mean": ctx_mean,
            "contextual_std": ctx_std,
            "delta_mean": mean([r["delta_contextual_minus_baseline"] for r in seed_level_rows]),
        },
        "tables": {"seed_level_metrics_csv": seed_csv, "pooled_group_metrics_csv": groups_csv},
        "figures": {
            "overall_seed_mean_std": os.path.join(args.output_dir, "overall_seed_mean_std.png"),
            "overall_pooled_ci": os.path.join(args.output_dir, "overall_pooled_ci.png"),
            "intersectional_pooled_ci": os.path.join(args.output_dir, "intersectional_pooled_ci.png"),
        },
    }
    summary_path = os.path.join(args.output_dir, "aggregate_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
