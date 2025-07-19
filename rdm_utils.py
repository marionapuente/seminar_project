import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


def compute_rdm(class_df, electrode_columns):
    """
    Compute the Representational Dissimilarity Matrix (RDM) and vector from class-averaged electrode responses.
    """
    class_vectors = class_df.groupby("Class")[electrode_columns].mean().sort_index()
    labels = class_vectors.index.tolist()

    # Distance matrix (squareform)
    dist_array = pdist(class_vectors.values, metric='euclidean')
    dist_matrix = pd.DataFrame(squareform(dist_array), index=labels, columns=labels)

    # RDM vector (upper triangle)
    rdm_vector = dist_matrix.where(np.triu(np.ones(dist_matrix.shape), k=1).astype(bool)).stack().values

    return dist_matrix, rdm_vector, labels


def compute_all_rdms(animal_df, region_electrode_map):
    """
    Compute RDMs for all regions using the given animal dataframe and electrode mapping.
    """
    rdms = {}
    for region, cols in region_electrode_map.items():
        if cols:
            print(f"Computing RDM for region: {region} (n={len(cols)} electrodes)")
            matrix, vector, labels = compute_rdm(animal_df, cols)
            rdms[region] = {
                "distance_matrix": matrix,
                "rdm_vector": vector,
                "labels": labels
            }
        else:
            print(f"No electrodes found for region: {region}")
    return rdms


def save_rdms(rdms, output_dir):
    """
    Save RDM matrices, vectors, and labels to the specified directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    for region, data in rdms.items():
        # Save distance matrix (CSV)
        data["distance_matrix"].to_csv(os.path.join(output_dir, f"{region}_distance_matrix.csv"))

        # Save RDM vector (NumPy)
        np.save(os.path.join(output_dir, f"{region}_rdm_vector.npy"), data["rdm_vector"])

        # Save labels (TXT)
        with open(os.path.join(output_dir, f"{region}_labels.txt"), "w") as f:
            for label in data["labels"]:
                f.write(f"{label}\n")

        print(f"Saved RDM and labels for region: {region}")