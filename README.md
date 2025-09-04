# Feature Selection (Classical + Quantum-ready)

This project implements a **Mutual Information-based feature selection pipeline** with:

- Discretization of a real dataset (UCI Taiwanese Bankruptcy)
- Mutual information relevance/redundancy computation
- Low-rank Nyström approximation of the redundancy matrix
- Low-dimensional QPFS solved via quadprog (**classical path**)
- Multibody Formulationand submission to QCi Dirac (**quantum-ready path**)

---

## Repository Structure

### `discretization/`

- **`data_loader.py`**: Fetch UCI dataset and save as `bankruptcy_data.npz`.
- **`discretizer.py`**: KMeans discretization to `discretized_bankruptcy_kmeans.pt`.
- **`information_bin_S_P.py`**: Information loss per feature after discretization.

### `mutual_information/`

- **`mutual_info.py`**: Computes relevance (feature-target MI) and redundancy (feature-feature MI); writes `mutual_i_result.py`.
- **`manual_mi.py`**: PyTorch-based MI calculator (self-contained alternative).

### `dimention_reduction/`

- **`dimention_reduction.py`**: Nyström low-rank approximation; writes `low_dim_components.py`.
- **`low_dim_components.py`**: Generated eigen-like components for downstream solvers.

### `classical/`

- **`quadprog_solver.py`**: Solves low-dim QPFS using quadprog; prints top features.

### `quantum/`

- **`coefcient_indices.py`**: Builds Multibody polynomial terms from MI outputs; writes `multi_body_data.py`.
- **`dirac_loader.py`**: Submits Multibody to QCi Dirac, processes and prints selected features.

### Root

- **`requirements.txt`**: Pinned dependencies
