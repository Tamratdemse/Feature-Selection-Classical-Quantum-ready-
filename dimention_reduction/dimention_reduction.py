import numpy as np
from mutual_information.mutual_i_result import redundancy

n_features = redundancy.shape[0]
r =64# Target low dimension

print(f"✅ Loaded data. Reducing from {n_features}x{n_features} to {r}x{r}...")

# --- 1. STANDARD RANDOM SAMPLING ---
print("Selecting features using uniform random sampling...")
selected_indices = np.random.choice(n_features, size=r, replace=False)
selected_indices.sort()
print(f"Sampled feature indices: {selected_indices}")

# --- 2. PARTITION MATRIX & REGULARIZE ---
mask = np.zeros(n_features, dtype=bool)
mask[selected_indices] = True
A = redundancy[mask, :][:, mask]  # r x r block
C = redundancy[:, mask]           # n x r block

# Regularize A to ensure positive definiteness (crucial for inversion)
# Use smallest eigenvalue for scaling
eigenvals_temp, _ = np.linalg.eigh(A)
min_eig = np.min(eigenvals_temp)
epsilon = max(1e-8, 1e-6 * abs(min_eig))  # small perturbation
A_reg = A + epsilon * np.eye(r)
print(f"Dynamic epsilon for regularization: {epsilon:.6e}")

# --- 3. EIGEN DECOMPOSITION ---
eigenvals_A, eigenvecs_A = np.linalg.eigh(A_reg)
idx = eigenvals_A.argsort()[::-1]
eigenvals_A = eigenvals_A[idx]
eigenvecs_A = eigenvecs_A[:, idx]

# Clip negative eigenvalues for stability
eigenvals_A = np.clip(eigenvals_A, a_min=0, a_max=None)

# --- 4. NYSTRÖM APPROXIMATION ---
# Avoid division by zero for null eigenvalues
Sigma_inv_sqrt_diag = np.zeros_like(eigenvals_A)
nonzero_mask = eigenvals_A > 1e-12
Sigma_inv_sqrt_diag[nonzero_mask] = 1.0 / np.sqrt(eigenvals_A[nonzero_mask])
Sigma_inv_sqrt = np.diag(Sigma_inv_sqrt_diag)

U_hat = C @ eigenvecs_A @ Sigma_inv_sqrt
Lambda_hat = np.diag(eigenvals_A)  # Keep diagonal eigenvalues (no extra scaling)



# --- 5. QUALITY CHECKS ---
print("\n🔍 Running quality checks on the approximation...")
Q_approx_sampled = U_hat[mask, :] @ U_hat[mask, :].T
error_sampled = np.linalg.norm(A - Q_approx_sampled) / np.linalg.norm(A)
print(f"Relative error on sampled block: {error_sampled:.6e}")

Q_approx_full = U_hat @ U_hat.T
error_full = np.linalg.norm(redundancy - Q_approx_full, 'fro') / np.linalg.norm(redundancy, 'fro')
print(f"Relative error on full matrix: {error_full:.6f}")

print(f"Eigenvalues of A (post-clip): {eigenvals_A}")
print(f"Condition number of A: {eigenvals_A[0]/eigenvals_A[-1]:.2e}" if eigenvals_A[-1] > 0 else "Ill-conditioned")

# --- 6. SAVE FOR QUADPROG ---
content = f'''# low_dim_components.py
# Low-dimensional representation of the redundancy matrix via Nyström method
# Generated automatically. Approximation error: {error_full:.6f}

import numpy as np

r = {r}
U_hat = np.array({U_hat.tolist()})          # Approximate eigenvectors (n x r)
Lambda_hat = np.array({Lambda_hat.tolist()}) # Low-dim system matrix (r x r diagonal)
selected_indices = np.array({selected_indices.tolist()}) # Indices of sampled features
'''

with open("dimention_reduction\\low_dim_components.py", "w") as f:
    f.write(content)

print(f"\n✅ Low-dimensional components saved to 'dimention_reduction\\low_dim_components.py'")
print(f"Final approximation error: {error_full:.6f}")
