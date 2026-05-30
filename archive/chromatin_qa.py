import os
import numpy as np
import matplotlib.pyplot as plt
from dwave.system import DWaveSampler, EmbeddingComposite

# --- CONFIG ---
MARKS = [
    'H3K9me3','H3K27me3','H4K20me1','H3K79me2',
    'H3K36me3','H3K4me1','H3K4me2','H3K4me3',
    'H3K9ac','H3K27ac','DNase','H2A.Z'
]

N_NUCLEOSOMES = 25
N_MARKS = len(MARKS)
L_INTERACTION = 5


# ---------------- LOAD ----------------
def load_parameters(filepath):
    raw = np.load(filepath, allow_pickle=True)
    print(f"Loaded parameters from {filepath} with shape {raw.shape}")

    h = raw[:, 0]
    J = raw[:, 1:13]
    K = raw[:, 13:]

    mask = np.triu(np.ones((N_MARKS, N_MARKS)), k=0)
    J = np.ma.masked_array(J, mask=mask)

    return {"h": h, "J": J, "K": K}


# ---------------- HAMILTONIAN ----------------
def build_hamiltonian(params):
    print("Building Hamiltonian...")

    h_params = params["h"] / 2
    J_params = params["J"] / 4
    K_params = params["K"] / 4

    h_adj = h_params.copy()

    for m in range(N_MARKS):
        row_sum = np.sum(np.ma.compressed(params["J"][m, :]))
        col_sum = np.sum(np.ma.compressed(params["J"][:, m]))
        h_adj[m] += (row_sum + col_sum) / 4

    h_dict = {}
    for n in range(N_NUCLEOSOMES):
        for m in range(N_MARKS):
            idx = n * N_MARKS + m
            h_dict[idx] = h_adj[m]

    J_dict = {}

    for n in range(N_NUCLEOSOMES):
        for m in range(N_MARKS):
            for k in range(m):
                u = n * N_MARKS + m
                v = n * N_MARKS + k
                val = J_params.data[m, k]
                if val != 0:
                    J_dict[(u, v)] = val

    for n in range(N_NUCLEOSOMES):
        for m in range(N_MARKS):
            for l in range(1, L_INTERACTION + 1):
                u = n * N_MARKS + m
                v = ((n + l) % N_NUCLEOSOMES) * N_MARKS + m
                w = K_params[m, l - 1]

                if u < v:
                    J_dict[(u, v)] = J_dict.get((u, v), 0) + w
                else:
                    J_dict[(v, u)] = J_dict.get((v, u), 0) + w

    print(f"Hamiltonian constructed: {len(h_dict)} linear, {len(J_dict)} quadratic")
    return h_dict, J_dict


# ---------------- BATCHING ----------------
def split_reads(total_reads, max_reads):
    batches = []
    while total_reads > 0:
        chunk = min(max_reads, total_reads)
        batches.append(chunk)
        total_reads -= chunk
    return batches


# ---------------- RUN CONDITION ----------------
def run_condition(sampler, h, J, label, num_reads, annealing_time, chain_strength, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)

    MAX_RUNTIME = 1_000_000
    per_read = annealing_time + 200

    max_reads = int(0.5 * (MAX_RUNTIME // per_read))
    max_reads = min(max_reads, 200)

    batches = split_reads(num_reads, max_reads)

    print(f"\n--- {label} ---")
    print(f"Splitting into {len(batches)} jobs")

    all_E = []
    all_CB = []

    for i, r in enumerate(batches):
        print(f"Batch {i+1}/{len(batches)}: {r} reads")

        response = sampler.sample_ising(
            h, J,
            num_reads=r,
            annealing_time=annealing_time,
            chain_strength=chain_strength,
            label=label
        )

        np.savez(
            os.path.join(save_dir, f"samples_{label}_batch{i}.npz"),
            samples=response.record.sample,
            energies=response.record.energy
        )

        all_E.append(response.record.energy)

        if "chain_break_fraction" in response.record.dtype.names:
            all_CB.append(response.record.chain_break_fraction)

    energies = np.concatenate(all_E)

    e_min = float(np.min(energies))
    e_mean = float(np.mean(energies))
    e_std = float(np.std(energies))

    cb = None
    if all_CB:
        cb = float(np.mean(np.concatenate(all_CB)))

    print("Energy:", e_min, e_mean, e_std)
    print("Chain break:", cb)

    plt.figure()
    plt.hist(energies, bins=30)
    plt.title(label)
    plt.savefig(os.path.join(save_dir, f"energy_{label}.png"))
    plt.close()

    return {
        "label": label,
        "num_reads": num_reads,
        "annealing_time": annealing_time,
        "chain_strength": chain_strength,
        "E_min": e_min,
        "E_mean": e_mean,
        "E_std": e_std,
        "chainbreak": cb
    }


# ---------------- MAIN ----------------
def main():
    params = load_parameters("model_parameters.npy")
    h, J = build_hamiltonian(params)

    sampler = EmbeddingComposite(DWaveSampler())

    results = []

    configs = []

    # -------- ALPHA SWEEP --------
    for alpha in [0.2, 0.25, 0.3, 0.4, 0.6]:
        configs.append({
            "label": f"alpha_{alpha}",
            "alpha": alpha,
            "num_reads": 3000,
            "annealing_time": 20,
            "chain_strength": 1.5
        })

    # -------- ANNEAL TIME SWEEP --------
    for t in [5, 20, 50, 100, 500]:
        configs.append({
            "label": f"anneal_{t}",
            "alpha": 0.25,
            "num_reads": 3000,
            "annealing_time": t,
            "chain_strength": 1.5
        })

    # -------- CHAIN STRENGTH SWEEP --------
    for cs in [0.5, 1.0, 1.5, 2.0, 3.0]:
        configs.append({
            "label": f"chain_{cs}",
            "alpha": 0.25,
            "num_reads": 3000,
            "annealing_time": 20,
            "chain_strength": cs
        })

    # -------- RUN ALL --------
    for cfg in configs:
        alpha = cfg["alpha"]

        h_s = {k: alpha*v for k,v in h.items()}
        J_s = {k: alpha*v for k,v in J.items()}

        results.append(
            run_condition(
                sampler,
                h_s,
                J_s,
                cfg["label"],
                cfg["num_reads"],
                cfg["annealing_time"],
                cfg["chain_strength"]
            )
        )

    # -------- SAVE CSV --------
    import csv
    os.makedirs("plots", exist_ok=True)

    with open("plots/summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\nSaved everything to ./plots/")


if __name__ == "__main__":
    main()