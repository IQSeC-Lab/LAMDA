import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.sparse import load_npz
import logging

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

def plot_yearwise_tsne_grid_same_scale(split_dir='npz_yearwise_Final',
                                       output_file='tsne_grid_all_years.png',
                                       save_npz_dir='tsne_embeddings'):
    logging.basicConfig(level=logging.INFO)
    os.makedirs(save_npz_dir, exist_ok=True)
    years = [y for y in range(2013, 2026) if y != 2015]

    all_embeddings = []
    all_labels = []

    for year in years:
        try:
            logging.info(f"Processing year {year}")
            X_path = os.path.join(split_dir, f"{year}_X.npz")
            meta_path = os.path.join(split_dir, f"{year}_meta.npz")

            # Check if both files exist
            if not os.path.exists(X_path) or not os.path.exists(meta_path):
                logging.warning(f"Missing data for year {year}: skipping...")
                continue

            # Load data
            X = load_npz(X_path).toarray()
            y = np.load(meta_path)["y"]

            # Check for cached t-SNE result
            tsne_result_path = os.path.join(save_npz_dir, f"{year}_tsne.npz")
            if os.path.exists(tsne_result_path):
                logging.info(f"Loading saved t-SNE for year {year}")
                tsne_data = np.load(tsne_result_path)
                X_embedded = tsne_data["embedding"]
                y = tsne_data["labels"]
            else:
                logging.info(f"Computing t-SNE for year {year}")
                tsne = TSNE(n_components=2, random_state=42, init='pca')
                X_embedded = tsne.fit_transform(X)
                np.savez_compressed(tsne_result_path, embedding=X_embedded, labels=y)

            all_embeddings.append((year, X_embedded))
            all_labels.append(y)
        except Exception as e:
            logging.warning(f"Skipping year {year} due to error: {e}")

    # Determine global axis limits
    all_coords = np.vstack([emb for _, emb in all_embeddings])
    x_min, x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
    y_min, y_max = all_coords[:, 1].min(), all_coords[:, 1].max()

    # Plot grid
    n_years = len(all_embeddings)
    n_cols = 4
    n_rows = (n_years + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    axes = axes.flatten()

    for i, ((year, embedding), y) in enumerate(zip(all_embeddings, all_labels)):
        benign_mask = (y == 0)
        malware_mask = (y == 1)
        axes[i].scatter(embedding[benign_mask, 0], embedding[benign_mask, 1], c='blue', s=5, alpha=0.5, label='Benign')
        axes[i].scatter(embedding[malware_mask, 0], embedding[malware_mask, 1], c='red', s=5, alpha=0.5, label='Malware')
        axes[i].set_title(f"{year}", fontsize=12, fontweight='bold')
        axes[i].set_xlim(x_min, x_max)
        axes[i].set_ylim(y_min, y_max)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_aspect('equal')

    # Turn off unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Add a global legend (outside the plot)
    handles = [
        plt.Line2D([], [], marker='o', color='blue', linestyle='', label='Benign'),
        plt.Line2D([], [], marker='o', color='red', linestyle='', label='Malware')
    ]
    fig.legend(handles=handles, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.05))

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved t-SNE grid plot to: {output_file}")
    print(f"Embeddings saved in: {save_npz_dir}")

# === Entry point ===
if __name__ == "__main__":
    plot_yearwise_tsne_grid_same_scale(
        split_dir='/home/shared-datasets/Feature_extraction/npz_yearwise_Final_rerun_0.001',  # Replace with your actual directory
        output_file='tsne_malware_dataset_grid_trim.png',
        save_npz_dir='tsne_embeddings'
    )
