"""Analyze saved D-Wave sampling batches.

Run from project root:
    python src/analyze_results.py

By default it reads:
    experiments/raw/samples_*.npz
and writes:
    experiments/analysis/
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def find_npz_files(raw_dir: Path) -> list[Path]:
    return sorted(raw_dir.glob("samples_*.npz"))


def extract_label(path: Path) -> str | None:
    # Supports both old and new names:
    # old: samples_<label>_batch0.npz
    # new: samples_<label>_batch0_<run_id>.npz
    match = re.match(r"samples_(.+)_batch\d+(?:_.+)?\.npz", path.name)
    if not match:
        return None
    return match.group(1)


def load_grouped_data(raw_dir: Path) -> dict[str, dict[str, list]]:
    grouped: dict[str, dict[str, list]] = {}

    for path in find_npz_files(raw_dir):
        label = extract_label(path)
        if label is None:
            continue

        data = np.load(path, allow_pickle=True)
        energies = data["energies"]
        samples = data["samples"]

        if label not in grouped:
            grouped[label] = {"files": [], "energies": [], "samples": [], "metadata": []}

        grouped[label]["files"].append(path.name)
        grouped[label]["energies"].append(energies)
        grouped[label]["samples"].append(samples)

        if "metadata_json" in data.files:
            try:
                grouped[label]["metadata"].append(json.loads(str(data["metadata_json"])))
            except Exception:
                grouped[label]["metadata"].append({})

    return grouped


def summarize(grouped: dict[str, dict[str, list]]) -> pd.DataFrame:
    rows = []

    for label, d in grouped.items():
        energies = np.concatenate(d["energies"])
        samples = np.concatenate(d["samples"], axis=0)
        ground = energies.min()
        ground_prob = np.mean(np.isclose(energies, ground))

        first_meta = d["metadata"][0] if d.get("metadata") else {}

        rows.append({
            "label": label,
            "num_batches": len(d["files"]),
            "num_samples": int(len(energies)),
            "num_variables": int(samples.shape[1]),
            "alpha": first_meta.get("alpha"),
            "annealing_time": first_meta.get("annealing_time"),
            "chain_strength": first_meta.get("chain_strength"),
            "solver": first_meta.get("solver"),
            "E_min": float(energies.min()),
            "E_mean": float(energies.mean()),
            "E_std": float(energies.std()),
            "E_5_percentile": float(np.quantile(energies, 0.05)),
            "E_median": float(np.quantile(energies, 0.50)),
            "E_95_percentile": float(np.quantile(energies, 0.95)),
            "ground_state_probability": float(ground_prob),
            "source_files": ";".join(d["files"]),
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("label")


def plot_energy_histograms(grouped: dict[str, dict[str, list]], out_dir: Path) -> None:
    for label, d in grouped.items():
        energies = np.concatenate(d["energies"])
        plt.figure(figsize=(8, 5))
        plt.hist(energies, bins=40)
        plt.xlabel("Energy")
        plt.ylabel("Count")
        plt.title(f"Energy distribution: {label}")
        plt.tight_layout()
        plt.savefig(out_dir / f"energy_hist_{label}.png", dpi=200)
        plt.close()


def plot_summary_comparison(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return

    x = np.arange(len(df))

    plt.figure(figsize=(10, 5))
    plt.errorbar(x, df["E_mean"], yerr=df["E_std"], fmt="o", capsize=4)
    plt.xticks(x, df["label"], rotation=45, ha="right")
    plt.ylabel("Energy")
    plt.title("Mean energy ± standard deviation")
    plt.tight_layout()
    plt.savefig(out_dir / "summary_energy_mean_std.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(x, df["ground_state_probability"])
    plt.xticks(x, df["label"], rotation=45, ha="right")
    plt.ylabel("Ground-state probability")
    plt.title("Ground-state probability by experiment")
    plt.tight_layout()
    plt.savefig(out_dir / "summary_ground_probability.png", dpi=200)
    plt.close()


def save_combined_npz(grouped: dict[str, dict[str, list]], out_dir: Path) -> None:
    for label, d in grouped.items():
        energies = np.concatenate(d["energies"])
        samples = np.concatenate(d["samples"], axis=0)
        np.savez(
            out_dir / f"combined_{label}.npz",
            energies=energies,
            samples=samples,
            source_files=np.array(d["files"]),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze saved chromatin D-Wave sampling batches.")
    parser.add_argument("--raw-dir", default="experiments/raw")
    parser.add_argument("--out-dir", default="experiments/analysis")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Could not find raw data directory: {raw_dir}")

    grouped = load_grouped_data(raw_dir)
    if not grouped:
        print(f"No samples_*.npz files found in {raw_dir}")
        return

    print("Found experiment labels:")
    for label, d in grouped.items():
        print(f"- {label}: {len(d['files'])} batch file(s)")

    df = summarize(grouped)
    summary_path = out_dir / "reconstructed_summary.csv"
    df.to_csv(summary_path, index=False)

    plot_energy_histograms(grouped, out_dir)
    plot_summary_comparison(df, out_dir)
    save_combined_npz(grouped, out_dir)

    print(f"\nSaved outputs to: {out_dir}")
    print("- reconstructed_summary.csv")
    print("- energy_hist_<label>.png")
    print("- summary_energy_mean_std.png")
    print("- summary_ground_probability.png")
    print("- combined_<label>.npz")
    print("\nSummary:")
    print(df)


if __name__ == "__main__":
    main()
