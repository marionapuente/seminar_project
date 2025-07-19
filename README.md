# seminar_project

DESCRIPTION:

Datasets are not attached because they are too big, instead instructions to get the raw data and the preprocessed datasets are provided here.

DOWNLOAD DATA:

Vlada explain how to download MUA data and Concept_to_category_linking.csv

Download THINGS dataset: https://things-initiative.org/projects/things-images/

CONTENT:

'prepare_MUA_dataset.py' is designed to process multi-unit activity (MUA) data from macaque monkey experiments. It follows these steps:

- Extract and structure neural responses (MUA) from .mat HDF5 files.
- Label each electrode response by its cortical region (V1, V4, IT).
- Associate each stimulus image with object categories.
- Save the final enriched dataset as a CSV file for further analysis.


### RDM Outputs
Precomputed Representational Dissimilarity Matrices (RDMs) are stored in `rdm_utils_output/`. To regenerate, see Step 4 in `main.py`.
