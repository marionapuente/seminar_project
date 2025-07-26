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
    
    # Compute RDM depending on the metric
    if metric == 'cosine':
        dist_array = 1 - np.dot(vectors, vectors.T) / (
            np.linalg.norm(vectors, axis=1, keepdims=True) *
            np.linalg.norm(vectors, axis=1, keepdims=True).T
        )
        # Convert to condensed form for compatibility with squareform-like behavior
        dist_matrix = pd.DataFrame(dist_array, index=labels, columns=labels)
    elif metric == 'euclidean':
        dist_array = pdist(vectors, metric=metric)
        dist_matrix = pd.DataFrame(squareform(dist_array), index=labels, columns=labels)
    elif metric == 'correlation':
        dist_array = pdist(vectors, metric='correlation')  # returns 1 - P correlation
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
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, "rdm.png"), dpi=300)
            dist_matrix.to_csv(os.path.join(save_path, "matrix.csv"))
            np.save(os.path.join(save_path, "vector.npy"), rdm_vector)

        if show_heatmap:
            plt.show()
        else:
            plt.close()

    return dist_matrix, rdm_vector, labels