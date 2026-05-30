import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RAW_DIR = "experiments/raw"
OUT_DIR = "analysis_outputs"

os.makedirs(OUT_DIR, exist_ok=True)


def find_npz_files():
    files = []
    for f in os.listdir(RAW_DIR):
        if f.startswith("samples_") and f.endswith(".npz"):
            files.append(f)
    return sorted(files)


def load_grouped_data():
    grouped = {}

    for fname in find_npz_files():
        path = os.path.join(RAW_DIR, fname)
        data = np.load(path, allow_pickle=True)

        metadata = json.loads(str(data["metadata_json"]))
        label = metadata["label"]
        run_id = metadata["run_id"]

        energies = data["energies"]
        samples = data["samples"]

        if "num_occurrences" in data:
            occurrences = data["num_occurrences"]
        else:
            occurrences = np.ones_like(energies, dtype=int)

        chainbreaks = data["chain_break_fraction"] if "chain_break_fraction" in data else None

        if run_id not in grouped:
            grouped[run_id] = {
                "label": label,
                "metadata": metadata,
                "files": [],
                "energies": [],
                "samples": [],
                "occurrences": [],
                "chainbreaks": [],
            }

        grouped[run_id]["files"].append(fname)
        grouped[run_id]["energies"].append(energies)
        grouped[run_id]["samples"].append(samples)
        grouped[run_id]["occurrences"].append(occurrences)

        if chainbreaks is not None:
            grouped[run_id]["chainbreaks"].append(chainbreaks)

    return grouped


def weighted_quantile(values, quantiles, sample_weight):
    values = np.asarray(values)
    quantiles = np.asarray(quantiles)
    sample_weight = np.asarray(sample_weight)

    sorter = np.argsort(values)
    values = values[sorter]
    sample_weight = sample_weight[sorter]

    weighted_cdf = np.cumsum(sample_weight)
    weighted_cdf = weighted_cdf / weighted_cdf[-1]

    return np.interp(quantiles, weighted_cdf, values)


def summarize(grouped):
    rows = []

    for run_id, d in grouped.items():
        energies = np.concatenate(d["energies"])
        samples = np.concatenate(d["samples"], axis=0)
        occurrences = np.concatenate(d["occurrences"]).astype(float)

        total_occurrences = occurrences.sum()
        ground = energies.min()

        e_mean = np.average(energies, weights=occurrences)
        e_var = np.average((energies - e_mean) ** 2, weights=occurrences)
        e_std = np.sqrt(e_var)

        ground_prob = occurrences[np.isclose(energies, ground)].sum() / total_occurrences

        q05, q50, q95 = weighted_quantile(
            energies,
            [0.05, 0.50, 0.95],
            occurrences,
        )

        row = {
            "run_id": run_id,
            "label": d["label"],
            "num_batches": len(d["files"]),
            "num_unique_samples": len(energies),
            "num_occurrences_total": int(total_occurrences),
            "num_variables": samples.shape[1],
            "E_min": ground,
            "E_mean_weighted": e_mean,
            "E_std_weighted": e_std,
            "E_5_percentile_weighted": q05,
            "E_median_weighted": q50,
            "E_95_percentile_weighted": q95,
            "ground_state_probability_weighted": ground_prob,
        }

        metadata = d["metadata"]
        for key in [
            "experiment_group",
            "study_type",
            "repeat_id",
            "alpha",
            "annealing_time",
            "chain_strength",
            "git_commit",
            "solver",
        ]:
            row[key] = metadata.get(key, "")

        rows.append(row)

    return pd.DataFrame(rows).sort_values(["study_type", "label", "run_id"])


def plot_energy_histograms(grouped):
    for run_id, d in grouped.items():
        energies = np.concatenate(d["energies"])
        occurrences = np.concatenate(d["occurrences"])

        label = d["label"]

        plt.figure(figsize=(8, 5))
        plt.hist(energies, bins=40, weights=occurrences)
        plt.xlabel("Energy")
        plt.ylabel("Weighted count")
        plt.title(f"Weighted energy distribution: {label}")
        plt.tight_layout()

        out = os.path.join(OUT_DIR, f"energy_hist_weighted_{label}_{run_id}.png")
        plt.savefig(out, dpi=200)
        plt.close()


def plot_boltzmann_energy_test(grouped):
    rows = []

    for run_id, d in grouped.items():
        energies = np.concatenate(d["energies"])
        occurrences = np.concatenate(d["occurrences"]).astype(float)

        unique_E, counts = np.unique(energies, return_counts=False), None

        energy_counts = {}
        for E, w in zip(energies, occurrences):
            energy_counts[float(E)] = energy_counts.get(float(E), 0.0) + float(w)

        for E, count in energy_counts.items():
            rows.append({
                "run_id": run_id,
                "label": d["label"],
                "energy": E,
                "count": count,
                "probability": count / occurrences.sum(),
                "log_probability": np.log(count / occurrences.sum()),
                "alpha": d["metadata"].get("alpha", ""),
                "annealing_time": d["metadata"].get("annealing_time", ""),
                "chain_strength": d["metadata"].get("chain_strength", ""),
                "study_type": d["metadata"].get("study_type", ""),
            })

        df_one = pd.DataFrame([r for r in rows if r["run_id"] == run_id])
        df_one = df_one.sort_values("energy")

        plt.figure(figsize=(7, 5))
        plt.scatter(df_one["energy"], df_one["log_probability"], s=12)
        plt.xlabel("Energy")
        plt.ylabel("log P(E)")
        plt.title(f"Boltzmann check: {d['label']}")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"boltzmann_logP_vs_E_{d['label']}_{run_id}.png"), dpi=200)
        plt.close()

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "boltzmann_energy_distribution.csv"), index=False)


def save_combined_npz(grouped):
    for run_id, d in grouped.items():
        energies = np.concatenate(d["energies"])
        samples = np.concatenate(d["samples"], axis=0)
        occurrences = np.concatenate(d["occurrences"])

        np.savez(
            os.path.join(OUT_DIR, f"combined_{d['label']}_{run_id}.npz"),
            energies=energies,
            samples=samples,
            num_occurrences=occurrences,
            source_files=np.array(d["files"]),
            metadata_json=np.array(json.dumps(d["metadata"], default=str)),
        )


def main():
    if not os.path.isdir(RAW_DIR):
        raise FileNotFoundError(
            f"Could not find '{RAW_DIR}/'. Run this script from the project root."
        )

    grouped = load_grouped_data()

    if not grouped:
        print("No samples_*.npz files found.")
        return

    print("\nFound experiment runs:")
    for run_id, d in grouped.items():
        print(f"- {d['label']} / {run_id}: {len(d['files'])} batch files")

    df = summarize(grouped)

    summary_path = os.path.join(OUT_DIR, "weighted_reconstructed_summary.csv")
    df.to_csv(summary_path, index=False)

    plot_energy_histograms(grouped)
    plot_boltzmann_energy_test(grouped)
    save_combined_npz(grouped)

    print("\nSaved outputs to:", OUT_DIR)
    print("- weighted_reconstructed_summary.csv")
    print("- energy_hist_weighted_<label>_<run_id>.png")
    print("- boltzmann_logP_vs_E_<label>_<run_id>.png")
    print("- boltzmann_energy_distribution.csv")
    print("- combined_<label>_<run_id>.npz")

    print("\nSummary:")
    print(df)


if __name__ == "__main__":
    main()