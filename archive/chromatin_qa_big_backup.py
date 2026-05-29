"""
Quantum Annealing for Chromatin Domain Sampling
===============================================

This script:
1. Loads learned parameters (h, J, K) representing epigenetic interactions.
2. Constructs an Ising Model Hamiltonian representing the chromatin fiber.
3. Embeds the problem onto a D-Wave Quantum Processor.
4. Samples low-energy configurations (Chromatin states).

Author: Tobias Kempe
Updated: 01.02.2026
"""

import os
import time
import math
import numpy as np
import networkx as nx
import minorminer
#import neal
from dwave.system import DWaveSampler, EmbeddingComposite
from collections import defaultdict
import matplotlib.pyplot as plt

# --- CONFIGURATION ---

import numpy as np

# Epigenetic Marks (from Paper Fig. 24)
MARKS = [
    'H3K9me3', 'H3K27me3', 'H4K20me1', 'H3K79me2', 
    'H3K36me3', 'H3K4me1', 'H3K4me2', 'H3K4me3', 
    'H3K9ac', 'H3K27ac', 'DNase', 'H2A.Z'
]

# Physical System Parameters
N_NUCLEOSOMES = 25      # Number of nucleosome sites (N)
L_INTERACTION = 5       # Interaction range (L)
N_MARKS = len(MARKS)    # Number of mark types (M)

def print_ising_stats(h, J, name="Ising"):
    # h: dict {i: hi}
    # J: dict {(i,j): Jij}
    h_vals = np.array(list(h.values()), dtype=float) if h else np.array([])
    J_vals = np.array(list(J.values()), dtype=float) if J else np.array([])

    def summary(arr, label):
        if arr.size == 0:
            print(f"{label}: (empty)")
            return
        qs = np.quantile(arr, [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0])
        print(f"{label}: n={arr.size}")
        print(f"  min/1%/5%/median/95%/99%/max = {qs}")
        print(f"  mean={arr.mean():.6g}, std={arr.std():.6g}")

    print(f"\n--- {name} coefficient stats ---")
    summary(h_vals, "h")
    summary(J_vals, "J")

    # Useful sanity indicators
    if h_vals.size and J_vals.size:
        print(f"  |h| max: {np.max(np.abs(h_vals)):.6g}")
        print(f"  |J| max: {np.max(np.abs(J_vals)):.6g}")
        print(f"  |J|/|h| max ratio: {(np.max(np.abs(J_vals)) / (np.max(np.abs(h_vals))+1e-12)):.6g}")

def load_parameters(filepath):
    """
    Loads and parses the raw parameter file.
    
    The file is expected to be a (M, 18) array where for each mark m:
    - col 0:      h (local field/bias)
    - col 1..12:  J (interaction with other marks)
    - col 13..17: K (interaction spatial distance 1..5)
    """
    try:
        raw_data = np.load(filepath, allow_pickle=True)
        print(f"Loaded parameters from {filepath} with shape {raw_data.shape}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Parameter file {filepath} not found. Please ensure it is in the same directory.")

    # Splitting the array based on the 1+12+5 structure
    h_raw = raw_data[:, 0]
    J_raw = raw_data[:, 1:13]
    K_raw = raw_data[:, 13:]

    # J is treated as a masked array in the original logic (using only lower triangle)
    # We apply a mask where index >= m to mimic 'np.triu' masking behavior from original code
    mask = np.triu(np.ones((N_MARKS, N_MARKS)), k=0)
    J_masked = np.ma.masked_array(J_raw, mask=mask)

    return {'h': h_raw, 'J': J_masked, 'K': K_raw}

def build_hamiltonian(params):
    """
    Constructs the linear (h) and quadratic (J) biases for the D-Wave sampler.
    Corresponds to the Hamiltonian definition in the paper.
    """
    print("Building Hamiltonian...")
    
    # Scaling factors observed in original code
    h_params = params['h'] / 2
    J_params = params['J'] / 4
    K_params = params['K'] / 4

    # 1. Calculate effective linear bias 'h' per mark
    # The original code adjusts 'h' based on the sum of interactions J
    h_adjusted = h_params.copy()
    for m in range(N_MARKS):
        # Sum of valid (unmasked) interactions involving m
        row_sum = np.sum(np.ma.compressed(params['J'][m, :]))
        col_sum = np.sum(np.ma.compressed(params['J'][:, m]))
        h_adjusted[m] += (row_sum + col_sum) / 4

    # D-Wave expects a dictionary for linear biases: h[qubit_index] -> bias
    h_dict = defaultdict(float) 
    for n in range(N_NUCLEOSOMES):
        for m in range(N_MARKS):
            qubit_idx = n * N_MARKS + m
            h_dict[qubit_idx] = h_adjusted[m]

    # 2. Calculate Quadratic Couplings 'J' (Interactions)
    J_dict = defaultdict(float)

    # A. Intra-nucleosome correlation (same site n, different marks m, k)
    for n in range(N_NUCLEOSOMES):
        for m in range(N_MARKS):
            for k in range(m): # Iterate lower triangle (k < m)
                u = n * N_MARKS + m
                v = n * N_MARKS + k
                # J_params is masked, so we access data directly or ensure we use valid indices
                val = J_params.data[m, k] 
                if val != 0:
                    J_dict[(u, v)] = val

    # B. Inter-nucleosome correlation (sites n and n+l, dependent on mark type)
    for n in range(N_NUCLEOSOMES):
        for m in range(N_MARKS):
            # Interaction range L (e.g., 1 to 5 neighbors)
            for l in range(1, L_INTERACTION + 1):
                u = n * N_MARKS + m
                # Periodic boundary condition or linear? Code used modulo: ((n+l)%N)
                v = ((n + l) % N_NUCLEOSOMES) * N_MARKS + m
                
                weight = K_params[m, l-1]
                
                if u < v:
                    J_dict[(u, v)] += weight
                else:
                    J_dict[(v, u)] += weight

    return h_dict, J_dict

def run_quantum_annealing(h, J, num_reads=2000, annealing_time=100):
    """
    Submits the problem to the D-Wave Quantum Annealer.
    Requires D-Wave Ocean SDK configuration (API token).
    """
    print(f"Submitting to D-Wave... (Reads: {num_reads})")
    
    try:
        # Use EmbeddingComposite to automatically map our logical graph to the QPU topology
        # 'Advantage_system' is the newer Pegasus graph, 'DW_2000Q' is Chimera.
        # We try to use the default solver available to the user's API token. 
        #sampler = neal.SimulatedAnnealingSampler()
        sampler = EmbeddingComposite(DWaveSampler())
        
        results = []

        base_num_reads = 5000
        base_anneal = 2000  # usually near max
        base_alpha = 0.25   # keep whatever you decided
        base_chain = 1.5    # or None if you want default

        # If you're scaling h,J elsewhere with alpha, do it BEFORE this section.
        # Example:
        # h_use = {k: base_alpha*v for k,v in h.items()}
        # J_use = {k: base_alpha*v for k,v in J.items()}
        h_use, J_use = h, J

        for rep in range(5):
            label = f"base_rep{rep}_a{base_alpha}_t{base_anneal}_r{base_num_reads}"
            results.append(
                run_condition(
                    sampler, h_use, J_use,
                    label=label,
                    num_reads=base_num_reads,
                    annealing_time=base_anneal,
                    chain_strength=base_chain,
                    num_spin_reversal_transforms=2,   # try 0/None if not supported
                    save_dir="plots",
                )
            )

        # ---------- 1) num_reads sweep ----------
        for r in [500, 2000, 5000, 10000]:
            label = f"reads_r{r}_a{base_alpha}_t{base_anneal}"
            results.append(
                run_condition(
                    sampler, h_use, J_use,
                    label=label,
                    num_reads=r,
                    annealing_time=base_anneal,
                    chain_strength=base_chain,
                    num_spin_reversal_transforms=2,
                    save_dir="plots",
                )
            )

        # ---------- 2) annealing_time sweep ----------
        for t in [5, 20, 50, 200, 500, 1000, 2000]:
            label = f"anneal_t{t}_a{base_alpha}_r{base_num_reads}"
            results.append(
                run_condition(
                    sampler, h_use, J_use,
                    label=label,
                    num_reads=base_num_reads,
                    annealing_time=t,
                    chain_strength=base_chain,
                    num_spin_reversal_transforms=2,
                    save_dir="plots",
                )
            )

        # ---------- 3) chain_strength sweep ----------
        for cs in [0.5, 1.0, 1.5, 2.0, 3.0]:
            label = f"chain_cs{cs}_a{base_alpha}_t{base_anneal}"
            results.append(
                run_condition(
                    sampler, h_use, J_use,
                    label=label,
                    num_reads=base_num_reads,
                    annealing_time=base_anneal,
                    chain_strength=cs,
                    num_spin_reversal_transforms=2,
                    save_dir="plots",
                )
            )

        # Save a CSV summary
        import csv
        with open("plots/summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)

        print("\nSaved plots to ./plots/ and summary to ./plots/summary.csv")
        
        print("USING DWAVE SAMPLER LINE 1")
        print("sampler is:", sampler)

        response = sampler.sample_ising(
            h,
            J,
            num_reads=num_reads,
            annealing_time=annealing_time,     # microseconds (sweep this)
            chain_strength=2.0,                # sweep this
            # num_spin_reversal_transforms=4,    # gauge averaging
            return_embedding=True,             # lets you inspect embedding
                label="Chromatin Sampling"
        )

        energies = response.record.energy
        print("Energy stats:")
        print("min:", np.min(energies))
        print("mean:", np.mean(energies))
        print("std:", np.std(energies))

        unique_states = len(response.record.sample)
        print("Number of returned samples:", unique_states)

        if "chain_break_fraction" in response.record.dtype.names:
            print("Mean chain break fraction:",
                  np.mean(response.record.chain_break_fraction))
        
        print("Sampling complete.")
        return response
        
    except ValueError as e:
        print("\nERROR: D-Wave Solver not found or API setup missing.")
        print("Ensure you have set up 'dwave config create' or set the DWAVE_API_TOKEN environment variable.")
        print(f"Details: {e}")
        return None

def plot_chromatin_state(state_matrix, title="Sampled Chromatin State"):
    """
    Plots a binary matrix where:
    - X-axis: Epigenetic Marks
    - Y-axis: Nucleosome Positions
    - Color: Active (1) vs Inactive (0/ -1)
    """
    n_nucleosomes, n_marks = state_matrix.shape
    
    plt.figure(figsize=(10, 8))
    
    # Create Heatmap
    # We map -1 to 0 for better visualization if spins are {-1, 1}
    display_matrix = np.where(state_matrix == -1, 0, state_matrix)
    
    plt.imshow(display_matrix, cmap='Blues', aspect='auto', interpolation='nearest')
    
    # Axis formatting
    plt.xticks(range(len(MARKS)), MARKS, rotation=45, ha='right')
    plt.xlabel("Epigenetic Marks")
    
    plt.yticks(range(0, n_nucleosomes, 5)) # Label every 5th nucleosome
    plt.ylabel("Nucleosome Position (Index)")
    
    plt.title(title)
    plt.colorbar(label="Occupancy (1=Present, 0=Absent)")
    plt.tight_layout()
    plt.savefig("mean_occupancy.png", dpi=200)
    plt.close()
    plt.pause(2)
    plt.close()
        
def summarize_sampleset(response, tag=""):
    import numpy as np

    E = response.record.energy

    print(f"\n=== {tag} ===")
    print("Energy: min/mean/std:", np.min(E), np.mean(E), np.std(E))
    print("Energy percentiles (5,50,95):", np.quantile(E, [0.05, 0.5, 0.95]))

    # Ground state probability
    ground = np.min(E)
    ground_prob = np.sum(np.isclose(E, ground)) / len(E)
    print("Ground state probability:", ground_prob)

    # Distinct low energies
    rounded = np.round(E, 6)
    uniq, counts = np.unique(rounded, return_counts=True)
    order = np.argsort(uniq)
    uniq, counts = uniq[order], counts[order]
    k = min(10, len(uniq))
    print("Lowest distinct energies (value:count):",
          ", ".join([f"{uniq[i]}:{counts[i]}" for i in range(k)]))

    # Chain break stats (if available)
    if "chain_break_fraction" in response.record.dtype.names:
        cb = response.record.chain_break_fraction
        print("Chain break fraction: mean/max:",
              float(np.mean(cb)), float(np.max(cb)))
        
def plot_energy_histogram(response, tag=""):
    import matplotlib.pyplot as plt
    import numpy as np

    E = response.record.energy

    plt.figure(figsize=(8, 5))
    plt.hist(E, bins=40, edgecolor='black')
    plt.xlabel("Energy")
    plt.ylabel("Count")
    plt.title(f"Energy Distribution ({tag})")
    plt.tight_layout()
    plt.show()

def compute_spin_correlations(response):
    import numpy as np
    import matplotlib.pyplot as plt

    samples = response.record.sample.astype(float)

    # Expect spins in {-1, +1}
    mean_spin = samples.mean(axis=0)

    # Correlation matrix
    C = (samples.T @ samples) / samples.shape[0] - np.outer(mean_spin, mean_spin)

    print("Correlation matrix shape:", C.shape)

    # Plot heatmap
    plt.figure(figsize=(6, 6))
    plt.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(label="Connected correlation ⟨s_i s_j⟩ - ⟨s_i⟩⟨s_j⟩")
    plt.title("Spin–Spin Correlation Matrix")
    plt.tight_layout()
    plt.savefig("correlation_matrix.png", dpi=200)
    plt.show()

    return C

def corr_vs_distance(samples_spins, N_NUCLEOSOMES, N_MARKS, outdir="plots",
                     alpha=None, annealing_time=None):
    """
    samples_spins: np.ndarray shape (num_samples, N_NUCLEOSOMES*N_MARKS) with values in {-1,+1}
    Assumes variable ordering: idx = nuc*N_MARKS + mark
    Computes connected correlation C(d) averaged over marks and all nuc pairs at separation d.
    Also computes per-mark curves.
    """
    os.makedirs(outdir, exist_ok=True)

    X = np.asarray(samples_spins, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"samples_spins must be 2D, got shape {X.shape}")

    nS, nV = X.shape
    expected = N_NUCLEOSOMES * N_MARKS
    if nV != expected:
        raise ValueError(f"Expected {expected} vars (=N_NUCLEOSOMES*N_MARKS) but got {nV}")

    # mean spin per variable
    mu = X.mean(axis=0)  # shape (nV,)

    # per-mark curves
    per_mark = np.zeros((N_MARKS, N_NUCLEOSOMES), dtype=float)
    per_mark_abs = np.zeros((N_MARKS, N_NUCLEOSOMES), dtype=float)

    for m in range(N_MARKS):
        # for each distance d
        for d in range(N_NUCLEOSOMES):
            vals = []
            for n in range(N_NUCLEOSOMES - d):
                i = n * N_MARKS + m
                j = (n + d) * N_MARKS + m
                cij = (X[:, i] * X[:, j]).mean() - (mu[i] * mu[j])  # connected corr
                vals.append(cij)
            per_mark[m, d] = float(np.mean(vals))
            per_mark_abs[m, d] = float(np.mean(np.abs(vals)))

    # average across marks
    C = per_mark.mean(axis=0)
    Cabs = per_mark_abs.mean(axis=0)

    # ---- plot (average across marks)
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(N_NUCLEOSOMES), C, marker="o", linewidth=1)
    plt.axhline(0.0, linewidth=1)
    title = "Connected corr vs distance (avg over marks)"
    if alpha is not None or annealing_time is not None:
        title += f" | alpha={alpha}, t={annealing_time}us"
    plt.title(title)
    plt.xlabel("Genomic distance (bin separation d)")
    plt.ylabel(r"$\langle s_i s_{i+d}\rangle - \langle s_i\rangle \langle s_{i+d}\rangle$")
    plt.tight_layout()

    alpha_str = "NA" if alpha is None else str(alpha).replace(".", "p")
    t_str = "NA" if annealing_time is None else str(annealing_time)
    f1 = os.path.join(outdir, f"corr_vs_dist_alpha{alpha_str}_t{t_str}us.png")
    plt.savefig(f1, dpi=200, bbox_inches="tight")
    plt.close()

    # ---- plot (avg absolute corr)
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(N_NUCLEOSOMES), Cabs, marker="o", linewidth=1)
    title = "Mean |connected corr| vs distance (avg over marks)"
    if alpha is not None or annealing_time is not None:
        title += f" | alpha={alpha}, t={annealing_time}us"
    plt.title(title)
    plt.xlabel("Genomic distance (bin separation d)")
    plt.ylabel(r"$\mathrm{mean}\;|\langle s_i s_{i+d}\rangle - \langle s_i\rangle \langle s_{i+d}\rangle|$")
    plt.tight_layout()

    f2 = os.path.join(outdir, f"abs_corr_vs_dist_alpha{alpha_str}_t{t_str}us.png")
    plt.savefig(f2, dpi=200, bbox_inches="tight")
    plt.close()

    # return arrays in case you want to print/inspect
    return C, Cabs, per_mark, per_mark_abs

def qpu_access_time_us(response):
    """Returns qpu_access_time (µs) if available, else None."""
    try:
        return response.info["timing"]["qpu_access_time"]
    except Exception:
        return None

import math

def split_reads_into_batches(total_reads, max_reads_per_job):
    """
    Splits total_reads into multiple batches that each
    have at most max_reads_per_job reads.
    """
    batches = []
    remaining = total_reads
    while remaining > 0:
        chunk = min(max_reads_per_job, remaining)
        batches.append(chunk)
        remaining -= chunk
    return batches

def run_condition(
    sampler,
    h, J,
    label,
    num_reads=5000,
    annealing_time=2000,
    chain_strength=None,
    num_spin_reversal_transforms=None,
    save_dir="plots",
    do_energy_hist=True,
):
    """
    One QPU job + minimal diagnostics.
    """
    os.makedirs(save_dir, exist_ok=True)

    kwargs = {
        "num_reads": int(num_reads),
        "annealing_time": float(annealing_time),
        "label": label,
    }
    if chain_strength is not None:
        kwargs["chain_strength"] = float(chain_strength)
    # Only add SRT if solver supports it
    if num_spin_reversal_transforms is not None:
        try:
            if "num_spin_reversal_transforms" in sampler.child.parameters:
                kwargs["num_spin_reversal_transforms"] = int(num_spin_reversal_transforms)
        except Exception:
            pass

        # --- Automatic runtime splitting ---

    MAX_RUNTIME_US = 1_000_000  # solver limit

    # Rough per-read estimate (anneal + ~60µs read/delay overhead)
    per_read_us = annealing_time + 80

    max_reads_per_job = int(0.9 * (MAX_RUNTIME_US // per_read_us))

    batches = split_reads_into_batches(num_reads, max_reads_per_job)

    all_energies = []
    all_chainbreak = []

    print(f"Splitting into {len(batches)} QPU jobs (max {max_reads_per_job} reads/job)")

    for i, batch_reads in enumerate(batches):
        print(f"Submitting batch {i+1}/{len(batches)} with {batch_reads} reads")

        kwargs_batch = kwargs.copy()
        kwargs_batch["num_reads"] = batch_reads

        response = sampler.sample_ising(h, J, **kwargs_batch)

        all_energies.append(response.record.energy)

        if "chain_break_fraction" in response.record.dtype.names:
            all_chainbreak.append(response.record.chain_break_fraction)

    # Concatenate all energies
    energies = np.concatenate(all_energies)

    cb_mean = None
    if all_chainbreak:
        cb_mean = float(np.mean(np.concatenate(all_chainbreak)))

    e_min, e_mean, e_std = float(np.min(energies)), float(np.mean(energies)), float(np.std(energies))

    # --- core stats ---
    energies = response.record.energy
    e_min, e_mean, e_std = float(np.min(energies)), float(np.mean(energies)), float(np.std(energies))
    t_us = qpu_access_time_us(response)

    # Chain break fraction (EmbeddingComposite typically provides it)
    cb_mean = None
    try:
        cb = response.record.chain_break_fraction
        cb_mean = float(np.mean(cb))
    except Exception:
        pass

    print(f"\n=== {label} ===")
    print(f"num_reads={num_reads}, anneal={annealing_time}us, chain_strength={chain_strength}, srt={num_spin_reversal_transforms}")
    print(f"Energy min/mean/std: {e_min:.6f}  {e_mean:.6f}  {e_std:.6f}")
    if cb_mean is not None:
        print(f"Mean chain break fraction: {cb_mean:.6g}")
    if t_us is not None:
        print(f"qpu_access_time: {t_us} us ({t_us/1e6:.3f} s)")

    # --- optional plot ---
    if do_energy_hist:
        plt.figure()
        plt.hist(energies, bins=30)
        plt.title(f"Energy hist: {label}")
        plt.xlabel("Energy")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"energy_{label}.png"), dpi=200)
        plt.close()

    return {
        "label": label,
        "num_reads": num_reads,
        "annealing_time": annealing_time,
        "chain_strength": chain_strength,
        "num_spin_reversal_transforms": num_spin_reversal_transforms,
        "E_min": e_min,
        "E_mean": e_mean,
        "E_std": e_std,
        "chainbreak_mean": cb_mean,
        "qpu_access_time_us": t_us,
    }


def main():
    # 1. Load Data
    param_file = 'model_parameters.npy' 
    
    # Check if file exists, if not, warn user (for this demo, we assume the user provides it)
    try:
        params = load_parameters(param_file)
    except Exception as e:
        print(f"Could not load parameters: {e}")
        return

    # 2. Build Model
    h, J = build_hamiltonian(params)
    print(f"Hamiltonian constructed: {len(h)} linear terms, {len(J)} quadratic terms.")

    # print_ising_stats(h, J, name="Chromatin model")

    alpha = 0.25
    annealing_time = 100
    h_s = {i: alpha*v for i, v in h.items()}
    J_s = {e: alpha*v for e, v in J.items()}

    # 3. Run Sampling
    num_reads = 2000
    response = run_quantum_annealing(h_s, J_s, num_reads=num_reads, annealing_time=annealing_time)

    summarize_sampleset(response, f"alpha={alpha}, anneal_time={annealing_time}")
    plot_energy_histogram(response, tag=f"alpha={alpha}, anneal_time={annealing_time}")
    C = compute_spin_correlations(response)

    # # 4. Process Results (Example)
    if response:
        #best_sample = response.first.sample

        # 1) Get samples: shape = (num_reads_returned, num_variables)
        samples = response.record.sample

        # --- Spin-spin corr vs genomic distance (uses existing samples; no extra QPU calls)
        # Ensure we have spins in {-1,+1} with shape (num_samples, N_NUCLEOSOMES*N_MARKS)
        samples_spins = samples  # if your 'samples' are already -1/+1

        # If your samples are 0/1, uncomment this:
        # samples_spins = 2*samples - 1

        C, Cabs, per_mark, per_mark_abs = corr_vs_distance(
            samples_spins,
            N_NUCLEOSOMES=N_NUCLEOSOMES,
            N_MARKS=N_MARKS,
            outdir="plots",
            alpha=alpha,
            annealing_time=annealing_time
        )
        print("Saved correlation-vs-distance plots to ./plots/")

        # 2) Compute mean spin per variable (works if spins are -1/+1 or 0/1)
        mean_per_var = samples.mean(axis=0)

        # 3) Determine whether samples are in {-1, +1} or {0, 1}
        #    We'll inspect a small subset of values safely
        unique_vals = np.unique(samples[:min(50, samples.shape[0]), :min(200, samples.shape[1])])
        # If values look like -1/+1, convert to occupancy in [0,1]
        if set(unique_vals.tolist()).issubset({-1, 1}):
            occ_per_var = (mean_per_var + 1.0) / 2.0
        else:
            # assume already 0/1-like
            occ_per_var = mean_per_var

        # 4) Ensure variable ordering is consistent with how you index variables (0..N-1)
        #    This is the safe way, even if response.variables is not sorted.
        vars_list = list(response.variables)
        occ_by_var = {v: occ for v, occ in zip(vars_list, occ_per_var)}

        N = N_NUCLEOSOMES * N_MARKS  # you already have these constants
        # Build a flat vector ordered by variable index 0..N-1
        flat = np.array([occ_by_var[i] for i in range(N)], dtype=float)

        # 5) Reshape into (nucleosomes, marks)
        state_matrix = flat.reshape(N_NUCLEOSOMES, N_MARKS)

        print("samples shape:", samples.shape)
        print("variable count:", len(response.variables))
        print("unique spin values (subset):", unique_vals[:10])

        # 6) Plot
        plt.figure(figsize=(10, 4))
        plt.imshow(state_matrix,
           aspect='auto',
           cmap='viridis',   # try: 'magma', 'plasma', 'inferno', 'cividis'
           vmin=0.0,
           vmax=1.0)
        plt.colorbar(label="Mean occupancy (probability present)")
        plt.xlabel("Histone marks")
        plt.ylabel("Nucleosomes / bins")
        plt.title("Mean occupancy over all QPU samples")
        plt.tight_layout()
        import os

        os.makedirs("plots", exist_ok=True)

        alpha_str = str(alpha).replace(".", "p")      # 0.25 -> "0p25"
        t_str = str(annealing_time)                      # e.g. 20
        fname = f"plots/spin_corr_alpha{alpha_str}_t{t_str}us.png"

        plt.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close()

if __name__ == "__main__":
    main()
