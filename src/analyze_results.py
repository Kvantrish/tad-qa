import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PLOTS_DIR = "plots"
OUT_DIR = "analysis_outputs"

os.makedirs(OUT_DIR, exist_ok=True)


def find_npz_files():
    files = []
    for f in os.listdir(PLOTS_DIR):
        if f.startswith("samples_") and f.endswith(".npz"):
            files.append(f)
    return sorted(files)


def extract_label(filename):
    # samples_<label>_batchX.npz
    match = re.match(r"samples_(.+)_batch\d+\.npz", filename)
    if not match:
        return None
    return match.group(1)


def load_grouped_data():
    grouped = {}

    for fname in find_npz_files():
        label = extract_label(fname)
        if label is None:
            continue

        path = os.path.join(PLOTS_DIR, fname)
        data = np.load(path)

        energies = data["energies"]
        samples = data["samples"]

        if label not in grouped:
            grouped[label] = {
                "files": [],
                "energies": [],
                "samples": [],
            }

        grouped[label]["files"].append(fname)
        grouped[label]["energies"].append(energies)
        grouped[label]["samples"].append(samples)

    return grouped


def summarize(grouped):
    rows = []

    for label, d in grouped.items():
        energies = np.concatenate(d["energies"])
        samples = np.concatenate(d["samples"], axis=0)

        ground = energies.min()
        ground_prob = np.mean(np.isclose(energies, ground))

        rows.append({
            "label": label,
            "num_batches": len(d["files"]),
            "num_samples": len(energies),
            "num_variables": samples.shape[1],
            "E_min": energies.min(),
            "E_mean": energies.mean(),
            "E_std": energies.std(),
            "E_5_percentile": np.quantile(energies, 0.05),
            "E_median": np.quantile(energies, 0.50),
            "E_95_percentile": np.quantile(energies, 0.95),
            "ground_state_probability": ground_prob,
        })

    df = pd.DataFrame(rows).sort_values("label")
    return df


def plot_energy_histograms(grouped):
    for label, d in grouped.items():
        energies = np.concatenate(d["energies"])

        plt.figure(figsize=(8, 5))
        plt.hist(energies, bins=40)
        plt.xlabel("Energy")
        plt.ylabel("Count")
        plt.title(f"Energy distribution: {label}")
        plt.tight_layout()

        out = os.path.join(OUT_DIR, f"energy_hist_{label}.png")
        plt.savefig(out, dpi=200)
        plt.close()


def plot_summary_comparison(df):
    # Energy mean with std
    plt.figure(figsize=(10, 5))
    x = np.arange(len(df))
    plt.errorbar(x, df["E_mean"], yerr=df["E_std"], fmt="o", capsize=4)
    plt.xticks(x, df["label"], rotation=45, ha="right")
    plt.ylabel("Energy")
    plt.title("Mean energy ± standard deviation")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "summary_energy_mean_std.png"), dpi=200)
    plt.close()

    # Ground-state probability
    plt.figure(figsize=(10, 5))
    plt.bar(x, df["ground_state_probability"])
    plt.xticks(x, df["label"], rotation=45, ha="right")
    plt.ylabel("Ground-state probability")
    plt.title("Ground-state probability by experiment")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "summary_ground_probability.png"), dpi=200)
    plt.close()


def save_combined_npz(grouped):
    for label, d in grouped.items():
        energies = np.concatenate(d["energies"])
        samples = np.concatenate(d["samples"], axis=0)

        np.savez(
            os.path.join(OUT_DIR, f"combined_{label}.npz"),
            energies=energies,
            samples=samples,
            source_files=np.array(d["files"])
        )


def main():
    if not os.path.isdir(PLOTS_DIR):
        raise FileNotFoundError(
            f"Could not find '{PLOTS_DIR}/'. Run this script from the folder containing your plots folder."
        )

    grouped = load_grouped_data()

    if not grouped:
        print("No samples_*.npz files found.")
        return

    print("\nFound experiment labels:")
    for label, d in grouped.items():
        print(f"- {label}: {len(d['files'])} batch files")

    df = summarize(grouped)

    summary_path = os.path.join(OUT_DIR, "reconstructed_summary.csv")
    df.to_csv(summary_path, index=False)

    plot_energy_histograms(grouped)
    plot_summary_comparison(df)
    save_combined_npz(grouped)

    print("\nSaved outputs to:", OUT_DIR)
    print("- reconstructed_summary.csv")
    print("- energy_hist_<label>.png")
    print("- summary_energy_mean_std.png")
    print("- summary_ground_probability.png")
    print("- combined_<label>.npz")

    print("\nSummary:")
    print(df)


if __name__ == "__main__":
    main()