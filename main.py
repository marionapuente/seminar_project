# MAIN FILE

# Import packages
import pandas as pd
import random
import os
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

from config import data_base_path, monkey_id, project_base_path,  metric
from utils import compute_rdm

############### STEP 1 ################
### COMPUTE RDMs FROM MUA RESPONSES ###
#######################################

# Load MUA data
df1 = pd.read_csv(f'{data_base_path}/datasets/{monkey_id}_MUA_responses.csv')
df1 = df1[df1['category_label'].notna()].reset_index(drop=True)  # Keep only rows with non-missing category label

# Extract electrode columns and electrode columns per region
electrode_cols = [col for col in df1.columns if 'Electrode_' in col and 'Region' not in col]
V1_electrode_cols = [col for col in electrode_cols if df1.loc[df1.index[0], f'{col}_Region'] == 'V1']
V4_electrode_cols = [col for col in electrode_cols if df1.loc[df1.index[0], f'{col}_Region'] == 'V4']
IT_electrode_cols = [col for col in electrode_cols if df1.loc[df1.index[0], f'{col}_Region'] == 'IT']

# Prepare data for RDM computation
# RDM of all brain per category
category_vectors = df1.groupby('category_label')[electrode_cols].mean()
category_labels = category_vectors.index.values
# RDM per region per category
region_vectors = {
    'V1': df1.groupby('category_label')[V1_electrode_cols].mean(),
    'V4': df1.groupby('category_label')[V4_electrode_cols].mean(),
    'IT': df1.groupby('category_label')[IT_electrode_cols].mean()
}
# RDM of all brain per animal
animal_df1 = df1[df1['category_label'] == 'animal'].reset_index(drop=True)
random.seed(42)
animals_to_keep = random.sample(animal_df1['Class'].unique().tolist(), 23)
animal_df1 = animal_df1[animal_df1['Class'].isin(animals_to_keep)]
animal_vectors = animal_df1.groupby('Class')[electrode_cols].mean()
animal_labels = animal_vectors.index.values

# RDM per region per animal
region_animal_vectors = {
    'V1': animal_df1.groupby('Class')[V1_electrode_cols].mean(),
    'V4': animal_df1.groupby('Class')[V4_electrode_cols].mean(),
    'IT': animal_df1.groupby('Class')[IT_electrode_cols].mean()
}

# Compute, display and save RDMs
# RDM of all brain per category
rdm_vector_brain_category = compute_rdm(category_vectors.values,
                                        labels=category_labels,
                                        metric=metric,
                                        save_path=os.path.join(project_base_path, "RDMs", metric, f"{monkey_id}_brain_category"),
                                        title="RDM - per category (all brain)")
# RDM per region per category
rdm_vector_V1_category = compute_rdm(region_vectors['V1'].values,
                                     labels=category_labels,
                                     metric=metric,
                                     save_path=os.path.join(project_base_path, "RDMs", metric, f"{monkey_id}_V1_category"),
                                     title="RDM - per category (V1)")
rdm_vector_V4_category = compute_rdm(region_vectors['V4'].values,
                                     labels=category_labels,
                                     metric=metric,
                                     save_path=os.path.join(project_base_path, "RDMs", metric, f"{monkey_id}_V4_category"),
                                     title="RDM - per category (V4)")
rdm_vector_IT_category = compute_rdm(region_vectors['IT'].values,
                                     labels=category_labels,
                                     metric=metric,
                                     save_path=os.path.join(project_base_path, "RDMs", metric, f"{monkey_id}_IT_category"),
                                     title="RDM - per category (IT)")
# RDM of all brain per animal
rdm_vector_brain_animal = compute_rdm(animal_vectors.values,
                                      labels=animal_labels,
                                      metric=metric,
                                      save_path=os.path.join(project_base_path, "RDMs", metric, f"{monkey_id}_brain_animal"),
                                      title="RDM - per animal (all brain)")
# RDM per region per animal
rdm_vector_V1_animal = compute_rdm(region_animal_vectors['V1'].values,
                                   labels=animal_labels,
                                   metric=metric,
                                   save_path=os.path.join(project_base_path, "RDMs", metric, f"{monkey_id}_V1_animal"),
                                   title="RDM - per animal (V1)")
rdm_vector_V4_animal = compute_rdm(region_animal_vectors['V4'].values,
                                   labels=animal_labels,
                                   metric=metric,
                                   save_path=os.path.join(project_base_path, "RDMs", metric, f"{monkey_id}_V4_animal"),
                                   title="RDM - per animal (V4)")
rdm_vector_IT_animal = compute_rdm(region_animal_vectors['IT'].values,
                                   labels=animal_labels,
                                   metric=metric,
                                   save_path=os.path.join(project_base_path, "RDMs", metric, f"{monkey_id}_IT_animal"),
                                   title="RDM - per animal (IT)")

################# STEP 2 ##################
### COMPUTE RDMs FROM PRETRAINED MODELS ###
###########################################

## CLIP-ViT Embeddings

# Load CLIP-ViT embeddings
df2 = pd.read_csv(os.path.join(data_base_path, "datasets", "clip_embeddings_with_labels.csv"))

# Add category labels
category_df = pd.read_csv(os.path.join(data_base_path, "raw_datasets", "Concept_to_category_linking.csv"))
category_df = category_df.drop(columns=[col for col in category_df.columns if col not in ['concept', 'category_label']])
category_df = category_df.rename(columns={'concept': 'label'})
df2 = df2.merge(category_df, on='label', how='left')
df2 = df2.dropna(subset=['label', 'category_label'])

# Prepare embeddings for RDM computation
# RDM per category
clip_cols = df2.select_dtypes(include='number').columns
category_clip = df2.groupby('category_label')[clip_cols].mean()
category_labels = category_clip.index.values
# RDM per animal
animal_df2 = df2[df2['category_label'] == 'animal'].reset_index(drop=True)
animal_df2 = animal_df2[animal_df2['label'].isin(animals_to_keep)]
animal_clip = animal_df2.groupby('label')[clip_cols].mean()
animal_labels = animal_clip.index.values

# Compute, display and save RDMs
# RDM per category
rdm_vector_clip_category = compute_rdm(category_clip.values,
                                       labels=category_labels,
                                       metric=metric,
                                       save_path=os.path.join(project_base_path, "RDMs", metric, "clip_category"),
                                       title="RDM - per category (CLIP-ViT)")
# RDM per animal
rdm_vector_clip_animal = compute_rdm(animal_clip.values,
                                     labels=animal_labels,
                                     metric=metric,
                                     save_path=os.path.join(project_base_path, "RDMs", metric, "clip_animal"),
                                     title="RDM - per animal (CLIP-ViT)")

## VGG16 Embeddings

# Load VGG16 embeddings
df3 = pd.read_csv(os.path.join(data_base_path, "datasets", "vgg16_embeddings_with_labels.csv"))

# Add category labels
df3 = df3.merge(category_df, on='label', how='left')
df3 = df3.dropna(subset=['label', 'category_label'])

# Prepare embeddings for RDM computation
# RDM per category
vgg_cols = df3.select_dtypes(include='number').columns
category_vgg = df3.groupby('category_label')[vgg_cols].mean()
category_labels = category_vgg.index.values
# RDM per animal
animal_df3 = df3[df3['category_label'] == 'animal'].reset_index(drop=True)
animal_df3 = animal_df3[animal_df3['label'].isin(animals_to_keep)]
animal_vgg = animal_df3.groupby('label')[vgg_cols].mean()
animal_labels = animal_vgg.index.values

# Compute, display and save RDMs
# RDM per category
rdm_vector_vgg_category = compute_rdm(category_vgg.values,
                                      labels=category_labels,
                                      metric=metric,
                                      save_path=os.path.join(project_base_path, "RDMs", metric, "vgg_category"),
                                      title="RDM - per category (VGG16)")
# RDM per animal
rdm_vector_vgg_animal = compute_rdm(animal_vgg.values,
                                    labels=animal_labels,
                                    metric=metric,
                                    save_path=os.path.join(project_base_path, "RDMs", metric, "vgg_animal"),
                                    title="RDM - per animal (VGG16)")

###### STEP 3 ######
### COMPARE RDMs ###
####################

# Compare RDMs using Spearman correlation
rho11, pval11 = spearmanr(rdm_vector_clip_category, rdm_vector_brain_category)
rho12, pval12 = spearmanr(rdm_vector_clip_animal, rdm_vector_brain_animal)
rho13, pval13 = spearmanr(rdm_vector_vgg_category, rdm_vector_brain_category)
rho14, pval14 = spearmanr(rdm_vector_vgg_animal, rdm_vector_brain_animal)
rho21, pval21 = spearmanr(rdm_vector_clip_category, rdm_vector_V1_category)
rho22, pval22 = spearmanr(rdm_vector_clip_animal, rdm_vector_V1_animal)
rho23, pval23 = spearmanr(rdm_vector_vgg_category, rdm_vector_V1_category)
rho24, pval24 = spearmanr(rdm_vector_vgg_animal, rdm_vector_V1_animal)
rho31, pval31 = spearmanr(rdm_vector_clip_category, rdm_vector_V4_category)
rho32, pval32 = spearmanr(rdm_vector_clip_animal, rdm_vector_V4_animal)
rho33, pval33 = spearmanr(rdm_vector_vgg_category, rdm_vector_V4_category)
rho34, pval34 = spearmanr(rdm_vector_vgg_animal, rdm_vector_V4_animal)
rho41, pval41 = spearmanr(rdm_vector_clip_category, rdm_vector_IT_category)
rho42, pval42 = spearmanr(rdm_vector_clip_animal, rdm_vector_IT_animal)
rho43, pval43 = spearmanr(rdm_vector_vgg_category, rdm_vector_IT_category)
rho44, pval44 = spearmanr(rdm_vector_vgg_animal, rdm_vector_IT_animal)

# Build table
correlation_table = pd.DataFrame(
    data=[
        [f"{rho11:.2f}\n(p={pval11:.3f})", f"{rho21:.2f}\n(p={pval21:.3f})", f"{rho31:.2f}\n(p={pval31:.3f})", f"{rho41:.2f}\n(p={pval41:.3f})"],
        [f"{rho12:.2f}\n(p={pval12:.3f})", f"{rho22:.2f}\n(p={pval22:.3f})", f"{rho32:.2f}\n(p={pval32:.3f})", f"{rho42:.2f}\n(p={pval42:.3f})"],
        [f"{rho13:.2f}\n(p={pval13:.3f})", f"{rho23:.2f}\n(p={pval23:.3f})", f"{rho33:.2f}\n(p={pval33:.3f})", f"{rho43:.2f}\n(p={pval43:.3f})"],
        [f"{rho14:.2f}\n(p={pval14:.3f})", f"{rho24:.2f}\n(p={pval24:.3f})", f"{rho34:.2f}\n(p={pval34:.3f})", f"{rho44:.2f}\n(p={pval44:.3f})"],
    ],
    index=['clip-category', 'clip-animal', 'vgg-category', 'vgg-animal'],
    columns=['MUA brain', 'MUA V1', 'MUA V4', 'MUA IT']
)

# Add row and column averages

# Extract rho values only
rho_values = correlation_table.map(lambda x: float(x.split('\n')[0]))
# Compute means
row_means = rho_values.mean(axis=1).round(2)
col_means = rho_values.mean(axis=0).round(2)
overall_mean = rho_values.values.mean().round(2)
# Format means to match table format
row_means_str = row_means.apply(lambda x: f"{x:.2f}")
col_means_str = col_means.apply(lambda x: f"{x:.2f}")
# Append row and column means
correlation_table.loc['average'] = col_means_str
correlation_table['average'] = list(row_means_str) + [f"{overall_mean:.2f}"]

# Display and save table

os.makedirs(f"correlation_table/{metric}", exist_ok=True)
correlation_table.to_csv(f"correlation_table/{metric}/rdm_spearman_table.csv")

plt.figure(figsize=(8, 4))
ax = sns.heatmap(
    data=correlation_table.map(lambda x: float(str(x).split('\n')[0]) if isinstance(x, str) else x),
    annot=correlation_table,
    fmt='',
    cmap='Reds',
    vmin=0,
    vmax=1,
    cbar=True,
    linewidths=0.5,
    linecolor='gray',
    annot_kws={"size": 8}
)

n_rows, n_cols = correlation_table.shape  # Get the position of the "average" row/column

# Thicker black lines
ax.hlines(n_rows - 1, *ax.get_xlim(), colors='black', linewidth=1.5)
ax.vlines(n_cols - 1, *ax.get_ylim(), colors='black', linewidth=1.5)

plt.title(f"Spearman Correlation of RDM computed using '{metric}'", fontsize=12)
plt.tight_layout()
plt.savefig(f"correlation_table/{metric}/rdm_spearman_heatmap.png", dpi=300)
plt.show()