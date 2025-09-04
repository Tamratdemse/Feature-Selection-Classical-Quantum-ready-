# qpfs_solver.py
# Stable low-dimensional QPFS solver using quadprog

import numpy as np
from quadprog import solve_qp

# ===============================
# Step 1: Load Data
# ===============================
from mutual_information.mutual_i_result import relevance, redundancy
from dimention_reduction.low_dim_components import U_hat, Lambda_hat, r, selected_indices

total_sum = np.log2(3)
n_features = U_hat.shape[0]

print(f"✅ Loaded data. Ready to solve {r}-dimensional QP problem.")

# ===============================
# Step 2: Prepare Nyström Blocks
# ===============================
mask = np.zeros(n_features, dtype=bool)
mask[selected_indices] = True
A_reg = redundancy[mask, :][:, mask]  # Already regularized in low_dim_components
C = redundancy[:, mask]

# Safe inversion with small regularization
epsilon = 1e-8
A_inv = np.linalg.inv(A_reg + epsilon * np.eye(r))

# Efficient computation of tilde_q without building n x n matrix
ones_vec = np.ones(n_features)
temp = C.T @ ones_vec
tilde_q = (temp.T @ A_inv @ temp) / (n_features ** 2)

# tilde_f: mean relevance
tilde_f = np.mean(relevance)

# Alpha hyperparameter
alpha_hat = tilde_q / (tilde_q + tilde_f)
print(f"\n--- Hyperparameter α ---")
print(f"tilde_f = {tilde_f:.6f}, tilde_q = {tilde_q:.6f}, α = {alpha_hat:.6f}")

# ===============================
# Step 3: Project relevance vector (normalized)
# ===============================
# Normalize columns of U_hat for stability
col_norms = np.linalg.norm(U_hat, axis=0)
U_hat_normed = U_hat / np.maximum(col_norms, 1e-12)  # avoid division by zero

relevance_low = U_hat_normed.T @ relevance

# ===============================
# Step 4: Formulate QP matrices
# ===============================
G = (1 - alpha_hat) * Lambda_hat   # shape: (r, r)
a = alpha_hat * relevance_low          # shape: (r,)

# ===============================
# Step 5: Constraints
# ===============================
# Equality constraint: sum(x) = total_sum
# x = U_hat_normed @ y => sum_i x_i = sum_i sum_j U[i,j] y_j = sum_j sum_i U[i,j] y_j
c_vec = np.sum(U_hat_normed, axis=0)   # shape: (r,)
C_eq = c_vec.reshape(1, -1)            # (1, r)
b_eq = np.array([total_sum])

# Inequality constraints: x >= 0 => U_hat_normed @ y >= 0
C_ineq = U_hat_normed.T                 # shape: (r, n_features) for solve_qp
b_ineq = np.zeros(n_features)

# Combine equality and inequality constraints
C = np.hstack([C_eq.T, C_ineq])         # shape: (r, 1 + n_features)
b = np.concatenate([b_eq, b_ineq])     # shape: (1 + n_features,)

# ===============================
# Step 6: Solve QP
# ===============================
# solve_qp solves: min 1/2 y^T G y - a^T y  s.t. C^T y >= b
meq = 1  # first constraint is equality
y_solution, f, xu, iterations, lagr, iact = solve_qp(G, a, C, b, meq=meq)



# ===============================
# Step 7: Recover full solution x = U_hat y
# ===============================
x_solution = U_hat_normed @ y_solution

# ===============================
# Step 8: Get top 10 features
# ===============================
top_10_indices = np.argsort(x_solution)[-10:][::-1]  # Indices of top 10 weights
top_10_weights = x_solution[top_10_indices]

print(f"\n✅ QP solved in {iterations} iterations.")
print(f"Objective value: {f:.6f}")
print(f"Top 10 features and their weights:")
for idx, weight in zip(top_10_indices, top_10_weights):
    print(f"Feature {idx}: weight = {weight:.6f}")

print(f"Sum of all weights: {np.sum(x_solution):.6f}")
print(f"Minimum weight (should be >=0): {np.min(x_solution):.6f}")

