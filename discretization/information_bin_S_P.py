import numpy as np
import torch

# --- Load original data ---
data_npz = np.load('discretization\\bankruptcy_data.npz')
X_orig = data_npz['X']   # shape: [n_samples, n_features]

# --- Load discretized data ---
data_pt = torch.load('discretization\\discretized_bankruptcy_kmeans.pt')
X_disc = data_pt['X'].numpy()  # shape: [n_samples, n_features]

n_samples, n_features = X_orig.shape
info_loss = np.zeros(n_features)

# --- Compute conditional entropy for each feature (vectorized) ---
for i in range(n_features):
    x = X_orig[:, i]
    xd = X_disc[:, i]

    # Build joint counts directly with histogram2d
    x_bins = np.unique(x)
    xd_bins = np.unique(xd)
    joint_counts, _, _ = np.histogram2d(
        x, xd, 
        bins=[len(x_bins), len(xd_bins)],
        range=[[x_bins.min(), x_bins.max()+1],
               [xd_bins.min(), xd_bins.max()+1]]
    )

    # Normalize to joint probability
    p_joint = joint_counts / joint_counts.sum()

    # Marginal probability P(Xd)
    p_xd = p_joint.sum(axis=0, keepdims=True)   # shape (1, n_xd)

    # Compute conditional entropy safely
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.divide(p_joint, p_xd, where=(p_joint > 0))
        H_cond = -np.sum(p_joint[p_joint > 0] * np.log2(ratio[p_joint > 0]))

    info_loss[i] = H_cond

# --- Print average information loss across all features ---
average_info_loss = info_loss.mean()
print(f"Average information loss H(X|Xd) = {average_info_loss:.4f} bits")

