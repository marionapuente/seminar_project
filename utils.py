# UTILITY FUNCTIONS FILE

# Import packages
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Function to compute RDMs

def compute_rdm(vectors, labels=None, metric='euclidean', show_heatmap=True, save_path=None, title="RDM"):
    
    # Compute RDM
    dist_array = pdist(vectors, metric=metric)
    dist_matrix = pd.DataFrame(squareform(dist_array), index=labels, columns=labels)
    rdm_vector = dist_matrix.where(np.triu(np.ones(dist_matrix.shape), k=1).astype(bool)).stack().values

    # Optional heatmap
    if show_heatmap or save_path:
        plt.figure(figsize=(10, 8))
        sns.heatmap(dist_matrix, xticklabels=True, yticklabels=True, cmap="viridis", annot=False, square=True, cbar_kws={"shrink": 0.75})
        plt.title(title)
        fontsize = 6 if len(labels) > 25 else 10
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.tight_layout()

        # Save if path is given
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"Heatmap saved to {save_path}")

        if show_heatmap:
            plt.show()
        else:
            plt.close()

    return dist_matrix, rdm_vector, labels