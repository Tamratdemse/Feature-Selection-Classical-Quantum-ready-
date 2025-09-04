# qpfs_solver_dynamic_alpha.py
import numpy as np
from quadprog import solve_qp
from mutual_information.mutual_i_result import relevance, redundancy

# =========================================================
# Step 1: Load Data
# =========================================================
Q = redundancy.copy()
F = relevance.copy()
n = len(F)

# =========================================================
# Step 2: Calculate alpha dynamically
# =========================================================
tilde_f = np.mean(F)
tilde_q = np.sum(Q) / (n ** 2)
alpha = tilde_q / (tilde_q + tilde_f)

print(f"✅ Calculated alpha dynamically: {alpha:.6f}")
print(f"tilde_f = {tilde_f:.6f}, tilde_q = {tilde_q:.6f}")

# =========================================================
# Step 3: Formulate QP
# =========================================================
G = (1 - alpha) * Q
a = alpha * F

# Constraints: sum(x) = log2(3), x >= 0
sum_constraint = np.log2(3)  # 1.584962500721156
C = np.vstack([np.ones(n), np.eye(n)])
b = np.hstack([sum_constraint, np.zeros(n)])

C_qp = C.T
meq = 1   # first constraint is equality

# Regularization for stability
G += 1e-8 * np.eye(n)

# =========================================================
# Step 4: Solve QP
# =========================================================
result = solve_qp(G, a, C_qp, b, meq)
if len(result) == 2:
    x_opt, f_val = result
else:
    x_opt, f_val, *_ = result

# =========================================================
# Step 5: Postprocess
# =========================================================
x_opt = np.maximum(x_opt, 0)

# Top 10 features
top_indices = np.argsort(-x_opt)[:20]
top_weights = x_opt[top_indices]

# =========================================================
# Step 6: Output
# =========================================================
print("\n✅ Top 10 features by weight:")
for rank, (idx, w) in enumerate(zip(top_indices, top_weights), 1):
    print(f"{rank:2d}. Feature {idx} -> weight = {w:.6f}")

print("\nSum of all weights:", np.sum(x_opt))
print("Objective value:", f_val)
