# Quantum Annealing for Chromatin Domain Sampling

This repository contains the implementation of the Quantum Annealing (QA) approach described in the paper:  
**"Intermediate State Formation of Topologically Associated Chromatin Domains using Quantum Annealing"**.

It uses D-Wave's quantum processors to sample chromatin states based on an Ising model derived from 1D epigenetic data.

## 📂 Repository Structure

* `chromatin_qa.py`: The main script. Loads parameters, constructs the Hamiltonian, and runs the quantum annealing process.
* `visualize_results.py`: Helper script to plot the resulting chromatin states as matrix heatmaps.
* `model_parameters.npy`: Pre-trained model parameters ($h, J, K$) representing interaction strengths between epigenetic marks (e.g., H3K9me3, H3K27me3).
* `requirements.txt`: Python dependencies.

## 🚀 Getting Started

### Prerequisites

1.  **Python 3.8+**
2.  **D-Wave Leap Account:** To run the code on a real Quantum Processing Unit (QPU), you need a D-Wave Leap account.
    * Sign up at [cloud.dwavesys.com](https://cloud.dwavesys.com/leap/).
    * Install and configure the CLI:
        ```bash
        pip install dwave-ocean-sdk
        dwave config create
        ```
    *(Note: The code can theoretically be adapted to use a classical simulated annealer if no QPU access is available.)*

### Installation

```bash
pip install -r requirements.txt
