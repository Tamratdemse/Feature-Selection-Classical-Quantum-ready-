from ucimlrepo import fetch_ucirepo
import numpy as np

# --- Fetch dataset ---
taiwanese_bankruptcy_prediction = fetch_ucirepo(id=572)
data = taiwanese_bankruptcy_prediction

# Extract features and target
X = data.data.features.to_numpy()
y = data.data.targets.to_numpy()
y = np.reshape(y, (6819,))

# --- Remove constant-value features ---
# Find columns where all values are the same
non_constant_cols = np.var(X, axis=0) > 0
X_filtered = X[:, non_constant_cols]

print(f"Removed {X.shape[1] - X_filtered.shape[1]} constant features.")
print(f"Remaining features: {X_filtered.shape[1]}")

# Save filtered dataset
np.savez('discretization\\bankruptcy_data.npz', X=X_filtered, y=y)
