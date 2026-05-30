"""Run D-Wave sampling experiments for the chromatin Ising model.

This script assumes the project layout:
    data/model_parameters.npy
    src/build_hamiltonian.py
    experiments/raw/
    experiments/plots/
    experiments/logs/

Run from project root, for example:
    python src/run_experiment.py --preset smoke
    python src/run_experiment.py --preset timing
    python src/run_experiment.py --preset alpha-small
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt

try:
    from build_hamiltonian import build_hamiltonian, load_parameters, scale_hamiltonian
except ImportError:
    # Allows running from unusual locations if src is not already on path.
    sys.path.append(str(Path(__file__).resolve().parent))
    from build_hamiltonian import build_hamiltonian, load_parameters, scale_hamiltonian


def now_run_id(label: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_label = label.replace(".", "p").replace("/", "-").replace(" ", "_")
    return f"{stamp}_{safe_label}"


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "not_available"


def split_reads(total_reads: int, max_reads: int) -> list[int]:
    if total_reads <= 0:
        raise ValueError("total_reads must be positive")
    if max_reads <= 0:
        raise ValueError("max_reads must be positive")
    batches = []
    remaining = total_reads
    while remaining > 0:
        chunk = min(max_reads, remaining)
        batches.append(chunk)
        remaining -= chunk
    return batches


def estimate_max_reads_per_job(annealing_time: float, max_runtime_us: int = 1_000_000, safety_factor: float = 0.5, hard_cap: int = 200) -> int:
    """Conservative read cap based on the old script's logic.

    This is intentionally conservative. Later we can replace it with a measured timing model.
    """
    per_read_us = annealing_time + 200
    max_reads = int(safety_factor * (max_runtime_us // per_read_us))
    return max(1, min(max_reads, hard_cap))


def get_solver_name(sampler: Any) -> str:
    # EmbeddingComposite has child; DWaveSampler has solver.
    try:
        return str(sampler.child.solver.name)
    except Exception:
        try:
            return str(sampler.solver.name)
        except Exception:
            return "unknown_solver"


def save_batch_npz(path: Path, response: Any, metadata: dict[str, Any]) -> None:
    info_json = json.dumps(response.info, default=str)
    metadata_json = json.dumps(metadata, default=str)

    arrays = {
        "samples": response.record.sample,
        "energies": response.record.energy,
        "metadata_json": np.array(metadata_json),
        "response_info_json": np.array(info_json),
    }

    if "chain_break_fraction" in response.record.dtype.names:
        arrays["chain_break_fraction"] = response.record.chain_break_fraction

    np.savez(path, **arrays)


def summarize_condition(label: str, all_energies: list[np.ndarray], all_chainbreaks: list[np.ndarray], timing_records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    energies = np.concatenate(all_energies)
    e_min = float(np.min(energies))
    e_mean = float(np.mean(energies))
    e_std = float(np.std(energies))
    ground_prob = float(np.mean(np.isclose(energies, e_min)))

    cb = None
    if all_chainbreaks:
        cb = float(np.mean(np.concatenate(all_chainbreaks)))

    def timing_sum(key: str) -> float | None:
        vals = []
        for rec in timing_records:
            val = rec.get(key)
            if isinstance(val, (int, float)):
                vals.append(float(val))
        return float(sum(vals)) if vals else None

    return {
        "run_id": config["run_id"],
        "label": label,
        "experiment_group": config.get("experiment_group", ""),
        "notes": config.get("notes", ""),
        "embedding_id": config.get("embedding_id", ""),
        "date": config["date"],
        "git_commit": config["git_commit"],
        "solver": config["solver"],
        "alpha": config["alpha"],
        "num_reads_requested": config["num_reads"],
        "num_samples_returned": int(len(energies)),
        "annealing_time": config["annealing_time"],
        "chain_strength": config["chain_strength"],
        "num_batches": config["num_batches"],
        "reads_per_batch": config["reads_per_batch"],
        "E_min": e_min,
        "E_mean": e_mean,
        "E_std": e_std,
        "ground_state_probability": ground_prob,
        "chain_break_fraction": cb,
        "qpu_access_time_us_sum": timing_sum("qpu_access_time"),
        "qpu_programming_time_us_sum": timing_sum("qpu_programming_time"),
        "qpu_sampling_time_us_sum": timing_sum("qpu_sampling_time"),
        "qpu_anneal_time_per_sample_us": timing_records[0].get("qpu_anneal_time_per_sample") if timing_records else None,
        "qpu_readout_time_per_sample_us": timing_records[0].get("qpu_readout_time_per_sample") if timing_records else None,
    }


def append_csv_row(csv_path: Path, row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def plot_histogram(energies: np.ndarray, label: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.hist(energies, bins=30)
    plt.xlabel("Energy")
    plt.ylabel("Count")
    plt.title(label)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def make_preset_configs(preset: str) -> list[dict[str, Any]]:
    if preset == "smoke":
        return [{"label": "smoke_alpha_0p4", "alpha": 0.4, "num_reads": 10, "annealing_time": 20, "chain_strength": 1.5}]

    if preset == "timing":
        return [
            {"label": f"timing_reads_{reads}", "alpha": 0.4, "num_reads": reads, "annealing_time": 20, "chain_strength": 1.5}
            for reads in [10, 50, 100, 500, 1000, 2000]
        ]

    if preset == "alpha-small":
        return [
            {"label": f"alpha_{str(alpha).replace('.', 'p')}", "alpha": alpha, "num_reads": 500, "annealing_time": 20, "chain_strength": 1.5}
            for alpha in [0.2, 0.4, 0.6]
        ]

    if preset == "previous-full":
        configs = []
        for alpha in [0.2, 0.25, 0.3, 0.4, 0.6]:
            configs.append({"label": f"alpha_{alpha}", "alpha": alpha, "num_reads": 3000, "annealing_time": 20, "chain_strength": 1.5})
        for t in [5, 20, 50, 100, 500]:
            configs.append({"label": f"anneal_{t}", "alpha": 0.25, "num_reads": 3000, "annealing_time": t, "chain_strength": 1.5})
        for cs in [0.5, 1.0, 1.5, 2.0, 3.0]:
            configs.append({"label": f"chain_{cs}", "alpha": 0.25, "num_reads": 3000, "annealing_time": 20, "chain_strength": cs})
        return configs

    raise ValueError(f"Unknown preset: {preset}")


def run_condition(sampler: Any, h_base: dict[int, float], J_base: dict[tuple[int, int], float], cfg: dict[str, Any], raw_dir: Path, plots_dir: Path, logs_dir: Path, max_reads_per_job: int | None) -> dict[str, Any]:
    label = cfg["label"]
    run_id = now_run_id(label)
    h_scaled, J_scaled = scale_hamiltonian(h_base, J_base, float(cfg["alpha"]))

    if max_reads_per_job is None:
        max_reads = estimate_max_reads_per_job(float(cfg["annealing_time"]))
    else:
        max_reads = max_reads_per_job

    batches = split_reads(int(cfg["num_reads"]), max_reads)
    solver_name = get_solver_name(sampler)

    run_meta = {
        **cfg,
        "run_id": run_id,
        "date": datetime.now().isoformat(timespec="seconds"),
        "git_commit": get_git_commit(),
        "solver": solver_name,
        "num_batches": len(batches),
        "reads_per_batch": ";".join(str(x) for x in batches),
    }

    print(f"\n--- {label} ---")
    print(json.dumps(run_meta, indent=2, default=str))

    all_energies: list[np.ndarray] = []
    all_chainbreaks: list[np.ndarray] = []
    timing_records: list[dict[str, Any]] = []

    for batch_index, reads_this_batch in enumerate(batches):
        print(f"Batch {batch_index + 1}/{len(batches)}: {reads_this_batch} reads")

        response = sampler.sample_ising(
            h_scaled,
            J_scaled,
            num_reads=reads_this_batch,
            annealing_time=cfg["annealing_time"],
            chain_strength=cfg["chain_strength"],
            label=f"{run_id}_batch{batch_index}",
        )

        timing = dict(response.info.get("timing", {}))
        timing_records.append(timing)

        metadata = {**run_meta, "batch_index": batch_index, "reads_this_batch": reads_this_batch, "timing": timing}
        batch_path = raw_dir / f"samples_{label}_batch{batch_index}_{run_id}.npz"
        save_batch_npz(batch_path, response, metadata)

        all_energies.append(response.record.energy)
        if "chain_break_fraction" in response.record.dtype.names:
            all_chainbreaks.append(response.record.chain_break_fraction)

    energies = np.concatenate(all_energies)
    plot_histogram(energies, label, plots_dir / f"energy_{label}_{run_id}.png")

    summary = summarize_condition(label, all_energies, all_chainbreaks, timing_records, run_meta)
    append_csv_row(logs_dir / "experiment_log.csv", summary)

    print("Summary:")
    print(json.dumps(summary, indent=2, default=str))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run chromatin Ising sampling on D-Wave.")
    parser.add_argument("--params", default="data/model_parameters.npy")
    parser.add_argument(
    "--preset",
    default="smoke",
    choices=["smoke", "timing", "alpha-small", "previous-full", "single"]
    )
    parser.add_argument("--raw-dir", default="experiments/raw")
    parser.add_argument("--plots-dir", default="experiments/plots")
    parser.add_argument("--logs-dir", default="experiments/logs")
    parser.add_argument("--max-reads-per-job", type=int, default=None, help="Override automatic conservative batching")
    parser.add_argument("--dry-run", action="store_true", help="Build Hamiltonian and print configs without using QPU")
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--num-reads", type=int)
    parser.add_argument("--annealing-time", type=float)
    parser.add_argument("--chain-strength", type=float)
    parser.add_argument("--label", type=str)
    parser.add_argument("--experiment-group", type=str, default="general")
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--embedding-id", type=str, default="auto_embedding")
    args = parser.parse_args()

    params = load_parameters(args.params)
    h_base, J_base = build_hamiltonian(params)
    if args.preset == "single":
        if None in (
            args.alpha,
            args.num_reads,
            args.annealing_time,
            args.chain_strength,
            args.label,
        ):
            raise ValueError(
                "single preset requires: "
                "--alpha --num-reads --annealing-time "
                "--chain-strength --label"
            )

        configs = [{
            "label": args.label,
            "experiment_group": args.experiment_group,
            "notes": args.notes,
            "embedding_id": args.embedding_id,
            "alpha": args.alpha,
            "num_reads": args.num_reads,
            "annealing_time": args.annealing_time,
            "chain_strength": args.chain_strength,
        }]
        
    else:
        configs = make_preset_configs(args.preset)
        for cfg in configs:
            cfg["experiment_group"] = args.experiment_group
            cfg["notes"] = args.notes
            cfg["embedding_id"] = args.embedding_id

    print(f"Hamiltonian constructed: {len(h_base)} linear terms, {len(J_base)} quadratic terms")
    print(f"Preset {args.preset} contains {len(configs)} condition(s).")

    if args.dry_run:
        print("Dry run only. No QPU call will be made.")
        print(json.dumps(configs, indent=2))
        return

    # Import only when actually running QPU, so local dry-runs work without Ocean installed.
    from dwave.system import DWaveSampler, EmbeddingComposite

    sampler = EmbeddingComposite(DWaveSampler())

    raw_dir = Path(args.raw_dir)
    plots_dir = Path(args.plots_dir)
    logs_dir = Path(args.logs_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for cfg in configs:
        summaries.append(
            run_condition(
                sampler=sampler,
                h_base=h_base,
                J_base=J_base,
                cfg=cfg,
                raw_dir=raw_dir,
                plots_dir=plots_dir,
                logs_dir=logs_dir,
                max_reads_per_job=args.max_reads_per_job,
            )
        )

    print("\nFinished all conditions.")
    print(f"Main log: {logs_dir / 'experiment_log.csv'}")


if __name__ == "__main__":
    main()
