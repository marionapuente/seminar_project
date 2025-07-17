# UTILITY FUNCTIONS FILE

# Import packages
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform

# Function to compute RDMs
def compute_rdm(vectors, labels=None, metric='euclidean'):

    if labels is None:
        labels = [f"Item{i}" for i in range(len(vectors))]

    dist_array = pdist(vectors, metric=metric)
    dist_matrix = pd.DataFrame(squareform(dist_array), index=labels, columns=labels)
    rdm_vector = dist_matrix.where(np.triu(np.ones(dist_matrix.shape), k=1).astype(bool)).stack().values

    return dist_matrix, rdm_vector, labels