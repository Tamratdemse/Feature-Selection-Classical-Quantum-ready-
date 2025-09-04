# preprocess_bankruptcy_discretize_only.py
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

# --- Load pre-saved data ---
data = np.load('discretization\\bankruptcy_data.npz')
X = data['X']
y = data['y']

# --- Discretization with K-Means ---
print("=" * 60)
print(" K-MEANS Discretization - Bankruptcy Dataset ")
print("=" * 60)

print(f"Dataset shape: {X.shape}")
print(f"Target distribution:\n{pd.Series(y).value_counts()}")

# Discretize numerical features
discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='kmeans')
X_discretized = discretizer.fit_transform(X)

# Show bin edges (optional)
print("\nBin edges for each feature:")
for i, edges in enumerate(discretizer.bin_edges_):
    print(f"Feature {i}:")
    for j in range(len(edges)-1):
        print(f"  Bin {j}: [{edges[j]:.6f}, {edges[j+1]:.6f})")

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_discretized, dtype=torch.long)
y_tensor = torch.tensor(y, dtype=torch.long)

# Save tensors
torch.save({"X": X_tensor, "y": y_tensor}, 
           "discretization\\discretized_bankruptcy_kmeans.pt")

print(f"\nSaved: X shape {X_tensor.shape}, y shape {y_tensor.shape}")
