# MAIN FILE

# Import packages
import pandas as pd

from config import data_base_path, monkey_id, project_base_path, excluded_classes
from utils import compute_rdm

### COMPUTE RDMs FROM MUA RESPONSES ###

# Load MUA data
csv_path = f'{data_base_path}/datasets/{monkey_id}_MUA_responses.csv'
df = pd.read_csv(csv_path)
df = df[df['category_label'].notna()].reset_index(drop=True)  # Keep only rows with non-missing category label

# Extract electrode columns and electrode columns per region
electrode_cols = [col for col in df.columns if 'Electrode_' in col and 'Region' not in col]
V1_electrode_cols = [col for col in electrode_cols if df.loc[df.index[0], f'{col}_Region'] == 'V1']
V4_electrode_cols = [col for col in electrode_cols if df.loc[df.index[0], f'{col}_Region'] == 'V4']
IT_electrode_cols = [col for col in electrode_cols if df.loc[df.index[0], f'{col}_Region'] == 'IT']

# Prepare data for RDM computation
# RDM of all brain per category
category_vectors = df.groupby('category_label')[electrode_cols].mean()
category_labels = category_vectors.index.values
# RDM per region per category
region_vectors = {
    'V1': df.groupby('category_label')[V1_electrode_cols].mean(),
    'V4': df.groupby('category_label')[V4_electrode_cols].mean(),
    'IT': df.groupby('category_label')[IT_electrode_cols].mean()
}
# RDM of all brain per animal
animal_df = df[
    (df['category_label'] == 'animal') &
    (~df['Class'].isin(excluded_classes))
].reset_index(drop=True)
animal_vectors = animal_df.groupby('Class')[electrode_cols].mean()
animal_labels = animal_vectors.index.values
# RDM per region per animal
region_animal_vectors = {
    'V1': animal_df.groupby('Class')[V1_electrode_cols].mean(),
    'V4': animal_df.groupby('Class')[V4_electrode_cols].mean(),
    'IT': animal_df.groupby('Class')[IT_electrode_cols].mean()
}

# Compute, display and save RDMs
# RDM of all brain per category
rdm_brain_category, rdm_vector_brain_category, labels_brain_category = compute_rdm(category_vectors.values,
                                                                                   labels=category_labels,
                                                                                   metric='euclidean',
                                                                                   save_path=f'{project_base_path}/RDMs/{monkey_id}_rdm_brain_category.png',
                                                                                   title="RDM - per category (all brain)")
# RDM per region per category
rdm_V1_category, rdm_vector_V1_category, labels_V1_category = compute_rdm(region_vectors['V1'].values,
                                                                             labels=category_labels,
                                                                             metric='euclidean',
                                                                             save_path=f'{project_base_path}/RDMs/{monkey_id}_rdm_V1_category.png',
                                                                             title="RDM - per category (V1)")
rdm_V4_category, rdm_vector_V4_category, labels_V4_category = compute_rdm(region_vectors['V4'].values,
                                                                             labels=category_labels,
                                                                             metric='euclidean',
                                                                             save_path=f'{project_base_path}/RDMs/{monkey_id}_rdm_V4_category.png',
                                                                             title="RDM - per category (V4)")
rdm_IT_category, rdm_vector_IT_category, labels_IT_category = compute_rdm(region_vectors['IT'].values,
                                                                             labels=category_labels,
                                                                             metric='euclidean',
                                                                             save_path=f'{project_base_path}/RDMs/{monkey_id}_rdm_IT_category.png',
                                                                             title="RDM - per category (IT)")
# RDM of all brain per animal
rdm_brain_animal, rdm_vector_brain_animal, labels_brain_animal = compute_rdm(animal_vectors.values,
                                                                                   labels=animal_labels,
                                                                                   metric='euclidean',
                                                                                   save_path=f'{project_base_path}/RDMs/{monkey_id}_rdm_brain_animal.png',
                                                                                   title="RDM - per animal (all brain)")
# RDM per region per animal
rdm_V1_animal, rdm_vector_V1_animal, labels_V1_animal = compute_rdm(region_animal_vectors['V1'].values,
                                                                             labels=animal_labels,
                                                                             metric='euclidean',
                                                                             save_path=f'{project_base_path}/RDMs/{monkey_id}_rdm_V1_animal.png',
                                                                             title="RDM - per animal (V1)")
rdm_V4_animal, rdm_vector_V4_animal, labels_V4_animal = compute_rdm(region_animal_vectors['V4'].values,
                                                                             labels=animal_labels,
                                                                             metric='euclidean',
                                                                             save_path=f'{project_base_path}/RDMs/{monkey_id}_rdm_V4_animal.png',
                                                                             title="RDM - per animal (V4)")
rdm_IT_animal, rdm_vector_IT_animal, labels_IT_animal = compute_rdm(region_animal_vectors['IT'].values,
                                                                             labels=animal_labels,
                                                                             metric='euclidean',
                                                                             save_path=f'{project_base_path}/RDMs/{monkey_id}_rdm_IT_animal.png',
                                                                             title="RDM - per animal (IT)")

### COMPUTE RDMs FROM PRETRAINED MODELS ###

# Load CLIP-ViT embeddings
#df = pd.read_csv('clip_embeddings_with_labels.csv')