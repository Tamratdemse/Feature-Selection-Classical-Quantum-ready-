# prepare_qubo.py
# Objective: 0.5 * (1 - alpha) * x^T Q x - alpha * F^T x
# With symmetric pair merging: [i,j] and [j,i] combined

import numpy as np

# =========================================================
# Step 1: Load Pre-Computed Mutual Information Results
# =========================================================

from mutual_information.mutual_i_result import relevance, redundancy

n_features = len(relevance)
print(f"✅ Loaded relevance vector: shape ({n_features},)")
print(f"✅ Loaded redundancy matrix: shape ({n_features}, {n_features})")

# =========================================================
# Step 2: Define Alpha and Prepare QUBO Components
# =========================================================

alpha = 0.758745  # tradeoff parameter

poly_coefficients = []
poly_indices = []

print("🔧 Preparing polynomial coefficients and indices (merged interactions)...")

# Quadratic terms
for i in range(n_features):
    for j in range(i, n_features):  # upper triangle only
        if i == j:
            # diagonal term (no doubling)
            coeff = 0.5 * (1 - alpha) * redundancy[i, i]
        else:
            # off-diagonal: combine Q[i,j] + Q[j,i]
            coeff = 0.5 * (1 - alpha) * (redundancy[i, j] + redundancy[j, i])
        poly_coefficients.append(coeff)
        poly_indices.append([i+1, j+1])  # indices start at 1

# Linear terms
for i in range(n_features):
    coeff = -alpha * relevance[i]
    poly_coefficients.append(coeff)
    poly_indices.append([0, i+1])

# Convert to numpy arrays
poly_coefficients = np.array(poly_coefficients)
poly_indices = np.array(poly_indices)

print(f"✅ Generated {len(poly_coefficients)} polynomial terms")

# =========================================================
# Step 3: Save to a Python File
# =========================================================

content = f'''# qubo_data.py
# Auto-generated polynomial coefficients and indices for QUBO formulation
# Objective: 0.5 * (1 - alpha) * x^T Q x - alpha * F^T x
# Symmetric interactions merged

import numpy as np

n_features = {n_features}

poly_coefficients = np.array({poly_coefficients.tolist()})
poly_indices = np.array({poly_indices.tolist()})

alpha = {alpha}
'''

with open("quantum/multi_body_data.py", "w") as f:
    f.write(content)

print("✅ Polynomial data saved to 'quantum/multi_body_data.py'")
print(f"   - Number of quadratic terms: {n_features*(n_features+1)//2}")
print(f"   - Number of linear terms: {n_features}")
print(f"   - Total terms: {len(poly_coefficients)}")
