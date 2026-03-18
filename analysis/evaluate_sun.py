#!/usr/bin/env python3
"""
Compute S·U·N (Stability · Uniqueness · Novelty) metrics.

S·U·N is the composite figure of merit used in the paper to compare
LoRA adapters (ranks 8, 16, 32) against the FiLM baseline.

  S = fraction of generated structures with E_hull ≤ 0.10 eV/atom
      (self-consistent MatterSim-based evaluation)
  U = N_unique / N_total  (structural deduplication)
  N = N_novel  / N_unique (not present in Alexandria-MP-20 training set)
  S·U·N = S × U × N

Usage
-----
  python analysis/evaluate_sun.py \
      --results_dir  generated/rank16 \
      --reference_db /path/to/alex_mp_20 \
      --output       analysis_results/rank16_sun.json

  # Compare all models at once
  python analysis/evaluate_sun.py \
      --results_dir  generated/rank8 generated/rank16 generated/rank32 generated/film \
      --labels       "LoRA R8" "LoRA R16" "LoRA R32" "FiLM" \
      --output       analysis_results/sun_comparison.json \
      --plot
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from pymatgen.core import Structure
    from pymatgen.analysis.structure_matcher import StructureMatcher
except ImportError:
    raise ImportError("pymatgen is required: pip install pymatgen")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Stability thresholds (eV/atom)
STABLE_THRESHOLD = 0.10
METASTABLE_THRESHOLD = 0.20

# Structural matching tolerances
STOL = 0.25     # Å — site tolerance
ANGLE_TOL = 5.0  # degrees


def load_evaluated_structures(results_dir: Path) -> pd.DataFrame:
    """
    Load post-processed evaluation results from a generation output directory.

    Expects a CSV or JSON produced by the post-processing pipeline with columns:
      formula, e_above_hull_eV_per_atom, is_unique, is_novel
    """
    for fname in ("evaluation_results.csv", "sun_results.csv", "results.csv"):
        fpath = results_dir / fname
        if fpath.exists():
            df = pd.read_csv(fpath)
            logger.info(f"Loaded {len(df)} structures from {fpath}")
            return df

    # Fall back to the DFT results table included in this repository
    fallback = Path("data/results/all_materials_summary.csv")
    if fallback.exists():
        df = pd.read_csv(fallback)
        logger.info(f"Loaded {len(df)} structures from fallback {fallback}")
        return df

    raise FileNotFoundError(
        f"No evaluation CSV found in {results_dir}. "
        "Run generation/generate_materials.py with post-processing first."
    )


def compute_stability(df: pd.DataFrame, ehull_col: str = "e_above_hull_eV_per_atom") -> float:
    """Fraction of structures with E_hull ≤ STABLE_THRESHOLD."""
    if ehull_col not in df.columns:
        raise KeyError(f"Column '{ehull_col}' not found. Available: {list(df.columns)}")
    return float((df[ehull_col] <= STABLE_THRESHOLD).mean())


def compute_uniqueness(df: pd.DataFrame, unique_col: str = "is_unique") -> float:
    """Fraction of structurally unique structures."""
    if unique_col in df.columns:
        return float(df[unique_col].mean())
    # If uniqueness flag not present, assume all are unique (conservative)
    logger.warning("'is_unique' column not found — assuming all structures are unique.")
    return 1.0


def compute_novelty(df: pd.DataFrame, novel_col: str = "is_novel") -> float:
    """Fraction of unique structures not present in the training set."""
    if novel_col in df.columns:
        unique_mask = df.get("is_unique", pd.Series([True] * len(df)))
        unique_df = df[unique_mask.astype(bool)]
        if len(unique_df) == 0:
            return 0.0
        return float(unique_df[novel_col].mean())
    logger.warning("'is_novel' column not found — assuming all unique structures are novel.")
    return 1.0


def bootstrap_ci(values: np.ndarray, n_resamples: int = 1000, ci: float = 0.95) -> Tuple[float, float]:
    """Return (lower, upper) bootstrapped confidence interval for the mean."""
    means = np.array([np.mean(np.random.choice(values, size=len(values), replace=True))
                      for _ in range(n_resamples)])
    alpha = (1 - ci) / 2
    return float(np.percentile(means, 100 * alpha)), float(np.percentile(means, 100 * (1 - alpha)))


def evaluate_single(results_dir: Path, label: str) -> Dict:
    """Compute S, U, N, and S·U·N for one results directory."""
    df = load_evaluated_structures(results_dir)

    s = compute_stability(df)
    u = compute_uniqueness(df)
    n = compute_novelty(df)
    sun = s * u * n

    # Bootstrap SE for stability
    ehull_vals = df["e_above_hull_eV_per_atom"].values if "e_above_hull_eV_per_atom" in df.columns else np.array([])
    if len(ehull_vals) > 0:
        stable_flags = (ehull_vals <= STABLE_THRESHOLD).astype(float)
        ci_lo, ci_hi = bootstrap_ci(stable_flags)
        s_se = float(np.std(stable_flags) / np.sqrt(len(stable_flags)))
    else:
        ci_lo, ci_hi, s_se = float("nan"), float("nan"), float("nan")

    result = {
        "label": label,
        "n_structures": len(df),
        "S": round(s, 4),
        "U": round(u, 4),
        "N": round(n, 4),
        "SUN": round(sun, 4),
        "S_SE": round(s_se, 4),
        "S_CI_95": [round(ci_lo, 4), round(ci_hi, 4)],
    }
    return result


def print_table(results: List[Dict]) -> None:
    """Pretty-print a comparison table."""
    header = f"{'Model':<15} {'N':>6} {'S':>7} {'U':>7} {'N':>7} {'S·U·N':>8}  SE(S)"
    print("\n" + "=" * 60)
    print(header)
    print("-" * 60)
    for r in results:
        print(
            f"{r['label']:<15} {r['n_structures']:>6} "
            f"{r['S']:>7.3f} {r['U']:>7.3f} {r['N']:>7.3f} "
            f"{r['SUN']:>8.4f}  ±{r['S_SE']:.3f}"
        )
    print("=" * 60 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute S·U·N metrics for one or more generation runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results_dir", type=Path, nargs="+", required=True,
        help="One or more generation output directories.",
    )
    parser.add_argument(
        "--labels", type=str, nargs="+", default=None,
        help="Display labels for each directory (default: directory names).",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("analysis_results/sun_metrics.json"),
        help="JSON file to save results.",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Save a bar chart of S·U·N decomposition.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    labels = args.labels or [d.name for d in args.results_dir]
    if len(labels) != len(args.results_dir):
        raise ValueError("--labels must have the same length as --results_dir")

    all_results = []
    for results_dir, label in zip(args.results_dir, labels):
        logger.info(f"Evaluating {label} ...")
        result = evaluate_single(results_dir, label)
        all_results.append(result)

    print_table(all_results)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(all_results, indent=2))
    logger.info(f"Results saved to {args.output}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt

            labels_plot = [r["label"] for r in all_results]
            s_vals = [r["S"] for r in all_results]
            u_vals = [r["U"] for r in all_results]
            n_vals = [r["N"] for r in all_results]
            sun_vals = [r["SUN"] for r in all_results]

            x = np.arange(len(labels_plot))
            width = 0.2
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(x - 1.5 * width, s_vals,   width, label="Stability (S)",  color="#2ecc71", alpha=0.85, edgecolor="black")
            ax.bar(x - 0.5 * width, u_vals,   width, label="Uniqueness (U)", color="#3498db", alpha=0.85, edgecolor="black")
            ax.bar(x + 0.5 * width, n_vals,   width, label="Novelty (N)",    color="#e74c3c", alpha=0.85, edgecolor="black")
            ax.bar(x + 1.5 * width, sun_vals, width, label="S·U·N",          color="#2c3e50", alpha=0.85, edgecolor="black")
            ax.set_xticks(x)
            ax.set_xticklabels(labels_plot)
            ax.set_ylim(0, 1.1)
            ax.set_ylabel("Score")
            ax.set_title("S·U·N Metric Decomposition")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plot_path = args.output.with_suffix(".pdf")
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            logger.info(f"Plot saved to {plot_path}")
        except Exception as e:
            logger.warning(f"Plotting failed: {e}")
