"""Build the chromatin-inspired Ising Hamiltonian.

This script/module loads the learned parameter matrix and converts it into
D-Wave-compatible Ising dictionaries h and J.

Expected project layout, when run from the project root:
    data/model_parameters.npy
    src/build_hamiltonian.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

MARKS = [
    "H3K9me3", "H3K27me3", "H4K20me1", "H3K79me2",
    "H3K36me3", "H3K4me1", "H3K4me2", "H3K4me3",
    "H3K9ac", "H3K27ac", "DNase", "H2A.Z",
]

N_NUCLEOSOMES = 25
N_MARKS = len(MARKS)
L_INTERACTION = 5

IsingH = Dict[int, float]
IsingJ = Dict[Tuple[int, int], float]


def load_parameters(filepath: str | Path) -> dict:
    """Load the learned chromatin parameters.

    The expected array shape is (12, 18):
      - column 0: local fields h
      - columns 1:13: mark-mark interactions J
      - columns 13:: distance interactions K
    """
    filepath = Path(filepath)
    raw = np.load(filepath, allow_pickle=True)

    if raw.shape != (N_MARKS, 18):
        raise ValueError(
            f"Expected parameter array shape {(N_MARKS, 18)}, got {raw.shape} from {filepath}"
        )

    h = raw[:, 0]
    J = raw[:, 1:13]
    K = raw[:, 13:]

    if K.shape[1] != L_INTERACTION:
        raise ValueError(f"Expected K to have {L_INTERACTION} columns, got {K.shape[1]}")

    # Keep only lower-triangular mark-mark interactions, as in the original code.
    mask = np.triu(np.ones((N_MARKS, N_MARKS)), k=0)
    J = np.ma.masked_array(J, mask=mask)

    return {"h": h, "J": J, "K": K, "raw": raw}


def build_hamiltonian(params: dict) -> tuple[IsingH, IsingJ]:
    """Build Ising h and J dictionaries from the parameter dictionary.

    This preserves the original transformation:
      h -> h/2 plus adjustment from J terms
      J -> J/4
      K -> K/4
    """
    h_params = params["h"] / 2
    J_params = params["J"] / 4
    K_params = params["K"] / 4

    h_adj = h_params.copy()

    for m in range(N_MARKS):
        row_sum = np.sum(np.ma.compressed(params["J"][m, :]))
        col_sum = np.sum(np.ma.compressed(params["J"][:, m]))
        h_adj[m] += (row_sum + col_sum) / 4

    h_dict: IsingH = {}
    for n in range(N_NUCLEOSOMES):
        for m in range(N_MARKS):
            idx = n * N_MARKS + m
            h_dict[idx] = float(h_adj[m])

    J_dict: IsingJ = {}

    # Same-nucleosome mark-mark couplings.
    for n in range(N_NUCLEOSOMES):
        for m in range(N_MARKS):
            for k in range(m):
                u = n * N_MARKS + m
                v = n * N_MARKS + k
                val = float(J_params.data[m, k])
                if val != 0:
                    J_dict[(u, v)] = val

    # Same-mark couplings across nearby nucleosomes with periodic boundary.
    for n in range(N_NUCLEOSOMES):
        for m in range(N_MARKS):
            for l in range(1, L_INTERACTION + 1):
                u = n * N_MARKS + m
                v = ((n + l) % N_NUCLEOSOMES) * N_MARKS + m
                w = float(K_params[m, l - 1])

                key = (u, v) if u < v else (v, u)
                J_dict[key] = J_dict.get(key, 0.0) + w

    return h_dict, J_dict


def scale_hamiltonian(h: IsingH, J: IsingJ, alpha: float) -> tuple[IsingH, IsingJ]:
    """Scale all local fields and couplings by alpha."""
    return (
        {k: float(alpha * v) for k, v in h.items()},
        {k: float(alpha * v) for k, v in J.items()},
    )


def hamiltonian_summary(h: IsingH, J: IsingJ) -> dict:
    h_values = np.array(list(h.values()), dtype=float)
    j_values = np.array(list(J.values()), dtype=float)
    return {
        "num_linear_terms": int(len(h)),
        "num_quadratic_terms": int(len(J)),
        "h_min": float(h_values.min()),
        "h_max": float(h_values.max()),
        "h_mean": float(h_values.mean()),
        "J_min": float(j_values.min()),
        "J_max": float(j_values.max()),
        "J_mean": float(j_values.mean()),
        "num_variables": int(N_NUCLEOSOMES * N_MARKS),
        "n_nucleosomes": int(N_NUCLEOSOMES),
        "n_marks": int(N_MARKS),
        "l_interaction": int(L_INTERACTION),
    }


def save_hamiltonian_npz(h: IsingH, J: IsingJ, out_path: str | Path) -> None:
    """Save h and J in a simple NPZ format for inspection/reuse."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    h_keys = np.array(list(h.keys()), dtype=int)
    h_vals = np.array(list(h.values()), dtype=float)
    j_keys = np.array(list(J.keys()), dtype=int)
    j_vals = np.array(list(J.values()), dtype=float)

    np.savez(out_path, h_keys=h_keys, h_values=h_vals, J_keys=j_keys, J_values=j_vals)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and inspect the chromatin Ising Hamiltonian.")
    parser.add_argument("--params", default="data/model_parameters.npy", help="Path to model_parameters.npy")
    parser.add_argument("--out", default="experiments/configs/hamiltonian_base.npz", help="Output NPZ path")
    parser.add_argument("--summary", default="experiments/configs/hamiltonian_summary.json", help="Output JSON summary path")
    args = parser.parse_args()

    params = load_parameters(args.params)
    h, J = build_hamiltonian(params)
    summary = hamiltonian_summary(h, J)

    save_hamiltonian_npz(h, J, args.out)
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Loaded parameters from {args.params} with shape {params['raw'].shape}")
    print(f"Hamiltonian constructed: {len(h)} linear terms, {len(J)} quadratic terms")
    print(json.dumps(summary, indent=2))
    print(f"Saved Hamiltonian to {args.out}")
    print(f"Saved summary to {args.summary}")


if __name__ == "__main__":
    main()
