import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import rel_entr
from scipy.sparse import load_npz

# Global font settings
plt.rcParams.update({
    "font.size": 14,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "legend.frameon": False
})

def kl_divergence(p, q, epsilon=1e-10):
    """KL(P || Q) with epsilon for stability."""
    p = np.asarray(p) + epsilon
    q = np.asarray(q) + epsilon
    return np.sum(rel_entr(p, q))

def jeffreys_divergence(p, q):
    """Jeffreys divergence: symmetric KL"""
    return kl_divergence(p, q) + kl_divergence(q, p)

def get_feature_distribution(npz_file):
    """Get normalized per-feature distribution."""
    X = load_npz(npz_file).toarray()
    feature_sums = X.sum(axis=0)
    total = feature_sums.sum()
    return feature_sums / total if total > 0 else np.zeros_like(feature_sums)

def compute_divergence_matrices(data_dir, years):
    distributions = {}
    for year in years:
        path = os.path.join(data_dir, f"{year}_X.npz")
        if os.path.exists(path):
            distributions[year] = get_feature_distribution(path)

    year_list = list(distributions.keys())
    n = len(year_list)
    kl_matrix = np.zeros((n, n))
    jeffreys_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            p = distributions[year_list[i]]
            q = distributions[year_list[j]]
            kl_matrix[i, j] = kl_divergence(p, q)
            jeffreys_matrix[i, j] = jeffreys_divergence(p, q)

    return kl_matrix, jeffreys_matrix, year_list

def plot_heatmap(matrix, labels, title, output_path, cmap='viridis'):
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, cmap=cmap)
    plt.colorbar(label=title)
    plt.xticks(np.arange(len(labels)), labels, rotation=45)
    plt.yticks(np.arange(len(labels)), labels)
    # plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# Settings
data_dir = "/home/shared-datasets/Feature_extraction/npz_yearwise_Final_rerun_0.001"
years = [y for y in range(2013, 2026) if y != 2015]
kl_output = "results/kl_divergence_featurewise_heatmap.png"
jeffreys_output = "results/jeffreys_divergence_featurewise_heatmap_combined.png"

# Compute both matrices
kl_matrix, jeffreys_matrix, year_labels = compute_divergence_matrices(data_dir, years)

# Save heatmaps
plot_heatmap(kl_matrix, year_labels, "KL Divergence (Per-Feature)", kl_output)
plot_heatmap(jeffreys_matrix, year_labels, "Jeffreys Divergence (Per-Feature)", jeffreys_output)
print(f"Heatmaps saved to {kl_output} and {jeffreys_output}")
