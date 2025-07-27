# Linking Primate Visual Cortex to Deep Neural Representations

This project is made by Vladlena Samusev and Mariona Puente Quera for the course 'Data Science Applications in Neuroscience' at Bar Ilan University (Israel).

ABSTRACT:

[...]

DOWNLOAD DATA:

Datasets are not attached because they are too big, instead instructions to get the raw data and the preprocessed datasets are provided here.

[VLADA EXPLAIN HOW TO DOWNLOAD MUA DATASET AND CONCEPT TO CATEGORY LINKING]

Download THINGS dataset: https://things-initiative.org/projects/things-images/

CONTENT:

'prepare_MUA_dataset.py' is designed to process multi-unit activity (MUA) data from macaque monkey experiments. It follows these steps:

- Extract and structure neural responses (MUA) from .mat HDF5 files.
- Label each electrode response by its cortical region (V1, V4, IT).
- Associate each stimulus image with object categories.
- Save the final enriched dataset as a CSV file for further analysis.

'get_embeddings.py' is designed to get image embeddings of the downloaded THINGS dataset from specified pretrained model (options: CLIP-ViT and VGG16). It follows these steps:

- Defines model and preprocess.
- Creates a list of image paths and their labels.
- Encodes every image via specified model and normalizes the resulting embedding.
- Saves embeddings and corresponding image labels.

'config.py' allows the user to specify certain configurations such as different directories' paths, the monkey, the pretrained model (to get the embeddings in 'get_embeddings.py') and the metric used to compute the RDMs (options: 'cosine', 'euclidean', 'correlation').

'utils.py' contains the function to compute, display and save RDMs that is repeatedly used in 'main.py'.

'main.py' is the main file with the main pipeline. It starts from preprocessed MUA dataset obtained in 'prepare_MUA_dataset.py' and embeddings computed in 'get_embeddings.py', as well as it needs the file 'Concept_to_category_linking.csv' located in the appropriate directory. It also needs the above specified set configurations. It follows three main steps:

- STEP 1: Compute RDMs from MUA responses --> it loads and prepares the MUA data necessary to compute RDMs per category and per animal, as well as using the whole visual cortex or the specified regions (V1, V4 and IT). It computes the RDMs using the function from 'utils.py', saving the RDM matrix, vector and heatmap.png.
- STEP 2: Compute RDMs from pretrained models --> it loads, prepares data, computes and saves RDMs in the same way first for CLIP-ViT model and then VGG16.
- STEP 3: Compare RDMs --> it compares neural RDMs vs RDMs from the pretrained artificial models using Spearman correlation. It compares the RDMs per category and per animal separately (category RDMs are compared only among other category RDMs and the same for animal RDMs). It computes, displays and saves a table with the results of these correlations.

'get_max_RDMs.py' is designed to get a relevant summary of the information contained in all the RDMs computed. It follows these steps:

- Loads the RDM matrices computed in 'main.py' and lists them.
- Gets maximum values and assures 0 minimum (maximum will then correspond to range), as well as it gets the category/animal with the highest dissimilarity value on average.
- Generates, displays and saves a table with this information.

Directories with results:

In 'RDMs' folder we get the RDMs' heatmaps, vectors and matrices per each of the neural and model RDMs and per each of the metrics (output from 'main.py').

In 'tables' folder we get the tables of the RDM comparison per metric (output from 'main.py'), as well as the table with the summary of the relevant information from each of the RDMs (output from 'get_max_RDMs.py').
