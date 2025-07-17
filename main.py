# MAIN FILE

# Import packages
import pandas as pd
import numpy as np

from config import data_base_path, monkey_id, excluded_classes
from utils import compute_rdm

### COMPUTE RDMs FROM MUA RESPONSES ###

# Load MUA data
csv_path = f'{data_base_path}/datasets/{monkey_id}_MUA_responses.csv'
df = pd.read_csv(csv_path)

# Extract electrode columns
electrode_cols = [col for col in df.columns if 'Electrode_' in col and 'Region' not in col]


### ARE WE ACUTALLY USING THIS?
# Compute mean response across all electrodes for each stimulus
df['Mean_Response_All_Electrodes'] = df[electrode_cols].mean(axis=1)

# Compute mean response per region (V1, V4, IT)
region_means = {}
for region in ['V1', 'V4', 'IT']:
    region_cols = [col for col in electrode_cols if df.loc[df.index[0], f'{col}_Region'] == region]
    if region_cols:
        df[f'Mean_Response_{region}'] = df[region_cols].mean(axis=1)
    else:
        df[f'Mean_Response_{region}'] = np.nan  # Assign NaN if no columns exist
### ARE WE ACTUALLY USING THIS?


# Keep only rows with non-missing category label
df = df[df['category_label'].notna()].reset_index(drop=True)

# Average MUA response per category and compute RDM of all brain per category
category_vectors = df.groupby('category_label')[electrode_cols].mean()
rdm_brain_category, rdm_vector_brain_category, labels_brain_category = compute_rdm(category_vectors.values, metric='euclidean')
print(rdm_brain_category)
print(rdm_vector_brain_category)
print(labels_brain_category)

# Average MUA response per brain region per category and compute RDM of brain regions per category
category_vectors = df.groupby('category_label')[region_cols].mean()
region_electrode_map = {
    region: [col for col in df.columns if col.startswith('Electrode_') and not col.endswith('_Region') and df[f'{col}_Region'].iloc[0] == region]
    for region in ['V1', 'V4', 'IT']
}
for region, cols in region_electrode_map.items():
    if cols:
        compute_rdm(category_vectors.values, metric='euclidean')
    else:
        print(f'No electrodes found for region: {region}')

#
animal_df = df[
    (df['category_label'] == 'animal') &
    (~df['Class'].isin(excluded_classes))
].reset_index(drop=True)
#compute_rdm(animal_df[electrode_cols].values, metric='euclidean')

# now per region


### COMPUTE RDMs FROM PRETRAINED MODELS ###

# Load CLIP-ViT embeddings
df = pd.read_csv('clip_embeddings_with_labels.csv')