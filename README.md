# Linking Primate Visual Cortex to Deep Neural Representations

This project is made by Vladlena Samusev and Mariona Puente Quera for the course 'Data Science Applications in Neuroscience' at Bar Ilan University (Israel).

ABSTRACT:

Understanding how the brain represents visual information is a central challenge in cognitive neuroscience. Recent deep neural networks (DNNs) trained on large-scale image datasets enable direct comparison between biological and artificial vision systems. Here, we examine the correspondence between multi-unit activity (MUA) from the macaque visual cortex (V1, V4, IT) and image embeddings from CLIP-ViT and VGG16. Using a subset of the THINGS database, with emphasis on the “animal” category, we compute Representational Dissimilarity Matrices (RDMs) using cosine and correlation metrics. CLIP showed the strongest alignment with neural representations, particularly in IT. Euclidean RDMs were also computed and are available in the accompanying repository but are excluded here due to interpretability limitations.

DOWNLOAD DATA:

Datasets are not attached because they are too big, instead instructions to get the raw data and the preprocessed datasets are provided here.

DOWNLOAD DATA:

How to Access the TVSD Multi-Unit Activity (MUA) Data

The MUA recordings used in this study are sourced from the THINGS Ventral Stream Spiking Dataset (TVSD) published by Papale et al. (2025), containing large-scale electrophysiological data from macaque visual cortex in response to THINGS images.

Direct Download (Recommended for Modeling/Tuning)
For normalized, averaged MUA data used in model comparison, download:

Monkey N (used in this study):

https://gin.g-node.org/paolo_papale/TVSD/raw/master/monkeyN/THINGS_normMUA.mat

This file includes:
-test_MUA: Averaged and normalized neural responses to each test stimulus.
-test_MUA_reps: Per-repetition responses (used for trial averaging).
-tb: Time points relative to stimulus onset.
-train_MUA: Responses to training images (not used here).

Additional metadata (e.g., mapping to ROIs) and full repository available at: https://gin.g-node.org/paolo_papale/TVSD

Concept-to-category mapping was obtained from the official THINGS dataset repository, where it is available as Concept_to_category_linking.csv under the Files section (https://osf.io/5a7z6). This file provides a mapping of 1,854 object concepts to 27 high-level semantic categories (e.g., “animal”, “tool”, “furniture”), based on WordNet hierarchy and manual curation as described in Hebart et al. (2019).

Download THINGS dataset: https://things-initiative.org/projects/things-images/

CONTENT:

'prepare_MUA_dataset.py' is designed to process multi-unit activity (MUA) data from macaque monkey experiments. It follows these steps:

- Extract and structure neural responses (MUA) from .mat HDF5 files.
- Label each electrode response by its cortical region (V1, V4, IT).
- Associate each stimulus image with object categories.
- Save the final enriched dataset as a CSV file for further analysis.
