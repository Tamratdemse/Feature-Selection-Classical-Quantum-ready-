import torch
import numpy as np
from sklearn.metrics import mutual_info_score

# =========================================================
# Step 1: Compute mutual information
# =========================================================
data = torch.load("discretization/discretized_bankruptcy_kmeans.pt")
X = data["X"].numpy()
y = data["y"].numpy()

n_features = X.shape[1]

def mutual_info_discrete(x, y):
    """Compute mutual information in bits (log base 2)."""
    mi_nats = mutual_info_score(x, y)  # MI in nats
    mi_bits = mi_nats / np.log(2)      # convert to bits
    return mi_bits

# Relevance: MI(feature, target)
relevance = np.array([mutual_info_discrete(X[:, i], y) for i in range(n_features)])

# Redundancy: MI(feature_i, feature_j)
redundancy = np.zeros((n_features, n_features))
for i in range(n_features):
    for j in range(n_features):
        redundancy[i, j] = mutual_info_discrete(X[:, i], X[:, j])

# =========================================================
# Step 2: Write results into result.py
# =========================================================
content = f'''# result.py
# --- Auto-generated Mutual Information Results (bits) ---

import numpy as np

# Relevance: MI(feature, target) in bits
relevance = np.array({relevance.tolist()})

# Redundancy: MI(feature_i, feature_j) as n x n matrix in bits
redundancy = np.array({redundancy.tolist()})
'''

with open("mutual_information\\mutual_i_result.py", "w") as f:
    f.write(content)

print("✅ Mutual information results (log base 2) saved inside mutual_information\\mutual_i_result.py")
