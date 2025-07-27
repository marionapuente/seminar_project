# Import packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load RDM matrices
# Cosine RDMs
cosine_MonkeyN_brain_category = pd.read_csv('RDMs/cosine/MonkeyN_brain_category/matrix.csv', header=0, index_col=0)
cosine_MonkeyN_brain_animal = pd.read_csv('RDMs/cosine/MonkeyN_brain_animal/matrix.csv', header=0, index_col=0)
cosine_MonkeyN_V1_category = pd.read_csv('RDMs/cosine/MonkeyN_V1_category/matrix.csv', header=0, index_col=0)
cosine_MonkeyN_V1_animal = pd.read_csv('RDMs/cosine/MonkeyN_V1_animal/matrix.csv', header=0, index_col=0)
cosine_MonkeyN_V4_category = pd.read_csv('RDMs/cosine/MonkeyN_V4_category/matrix.csv', header=0, index_col=0)
cosine_MonkeyN_V4_animal = pd.read_csv('RDMs/cosine/MonkeyN_V4_animal/matrix.csv', header=0, index_col=0)
cosine_MonkeyN_IT_category = pd.read_csv('RDMs/cosine/MonkeyN_IT_category/matrix.csv', header=0, index_col=0)
cosine_MonkeyN_IT_animal = pd.read_csv('RDMs/cosine/MonkeyN_IT_animal/matrix.csv', header=0, index_col=0)
cosine_clip_category = pd.read_csv('RDMs/cosine/clip_category/matrix.csv', header=0, index_col=0)
cosine_clip_animal = pd.read_csv('RDMs/cosine/clip_animal/matrix.csv', header=0, index_col=0)
cosine_vgg_category = pd.read_csv('RDMs/cosine/vgg_category/matrix.csv', header=0, index_col=0)
cosine_vgg_animal = pd.read_csv('RDMs/cosine/vgg_animal/matrix.csv', header=0, index_col=0)
# Euclidean RDMs
eucl_MonkeyN_brain_category = pd.read_csv('RDMs/euclidean/MonkeyN_brain_category/matrix.csv', header=0, index_col=0)
eucl_MonkeyN_brain_animal = pd.read_csv('RDMs/euclidean/MonkeyN_brain_animal/matrix.csv', header=0, index_col=0)
eucl_MonkeyN_V1_category = pd.read_csv('RDMs/euclidean/MonkeyN_V1_category/matrix.csv', header=0, index_col=0)
eucl_MonkeyN_V1_animal = pd.read_csv('RDMs/euclidean/MonkeyN_V1_animal/matrix.csv', header=0, index_col=0)
eucl_MonkeyN_V4_category = pd.read_csv('RDMs/euclidean/MonkeyN_V4_category/matrix.csv', header=0, index_col=0)
eucl_MonkeyN_V4_animal = pd.read_csv('RDMs/euclidean/MonkeyN_V4_animal/matrix.csv', header=0, index_col=0)
eucl_MonkeyN_IT_category = pd.read_csv('RDMs/euclidean/MonkeyN_IT_category/matrix.csv', header=0, index_col=0)
eucl_MonkeyN_IT_animal = pd.read_csv('RDMs/euclidean/MonkeyN_IT_animal/matrix.csv', header=0, index_col=0)
eucl_clip_category = pd.read_csv('RDMs/euclidean/clip_category/matrix.csv', header=0, index_col=0)
eucl_clip_animal = pd.read_csv('RDMs/euclidean/clip_animal/matrix.csv', header=0, index_col=0)
eucl_vgg_category = pd.read_csv('RDMs/euclidean/vgg_category/matrix.csv', header=0, index_col=0)
eucl_vgg_animal = pd.read_csv('RDMs/euclidean/vgg_animal/matrix.csv', header=0, index_col=0)
# Correlation RDMs
corr_MonkeyN_brain_category = pd.read_csv('RDMs/correlation/MonkeyN_brain_category/matrix.csv', header=0, index_col=0)
corr_MonkeyN_brain_animal = pd.read_csv('RDMs/correlation/MonkeyN_brain_animal/matrix.csv', header=0, index_col=0)
corr_MonkeyN_V1_category = pd.read_csv('RDMs/correlation/MonkeyN_V1_category/matrix.csv', header=0, index_col=0)
corr_MonkeyN_V1_animal = pd.read_csv('RDMs/correlation/MonkeyN_V1_animal/matrix.csv', header=0, index_col=0)
corr_MonkeyN_V4_category = pd.read_csv('RDMs/correlation/MonkeyN_V4_category/matrix.csv', header=0, index_col=0)
corr_MonkeyN_V4_animal = pd.read_csv('RDMs/correlation/MonkeyN_V4_animal/matrix.csv', header=0, index_col=0)
corr_MonkeyN_IT_category = pd.read_csv('RDMs/correlation/MonkeyN_IT_category/matrix.csv', header=0, index_col=0)
corr_MonkeyN_IT_animal = pd.read_csv('RDMs/correlation/MonkeyN_IT_animal/matrix.csv', header=0, index_col=0)
corr_clip_category = pd.read_csv('RDMs/correlation/clip_category/matrix.csv', header=0, index_col=0)
corr_clip_animal = pd.read_csv('RDMs/correlation/clip_animal/matrix.csv', header=0, index_col=0)
corr_vgg_category = pd.read_csv('RDMs/correlation/vgg_category/matrix.csv', header=0, index_col=0)
corr_vgg_animal = pd.read_csv('RDMs/correlation/vgg_animal/matrix.csv', header=0, index_col=0)

# List of RDM matrices
rdm_matrices = {
    'cosine_brain_category': cosine_MonkeyN_brain_category, 'cosine_brain_animal': cosine_MonkeyN_brain_animal,
    'cosine_V1_category': cosine_MonkeyN_V1_category, 'cosine_V1_animal': cosine_MonkeyN_V1_animal,
    'cosine_V4_category': cosine_MonkeyN_V4_category, 'cosine_V4_animal': cosine_MonkeyN_V4_animal,
    'cosine_IT_category': cosine_MonkeyN_IT_category, 'cosine_IT_animal': cosine_MonkeyN_IT_animal,
    'cosine_clip_category': cosine_clip_category, 'cosine_clip_animal': cosine_clip_animal,
    'cosine_vgg_category': cosine_vgg_category, 'cosine_vgg_animal': cosine_vgg_animal,
    'eucl_brain_category': eucl_MonkeyN_brain_category, 'eucl_brain_animal': eucl_MonkeyN_brain_animal,
    'eucl_V1_category': eucl_MonkeyN_V1_category, 'eucl_V1_animal': eucl_MonkeyN_V1_animal,
    'eucl_V4_category': eucl_MonkeyN_V4_category, 'eucl_V4_animal': eucl_MonkeyN_V4_animal,
    'eucl_IT_category': eucl_MonkeyN_IT_category, 'eucl_IT_animal': eucl_MonkeyN_IT_animal,
    'eucl_clip_category': eucl_clip_category, 'eucl_clip_animal': eucl_clip_animal,
    'eucl_vgg_category': eucl_vgg_category, 'eucl_vgg_animal': eucl_vgg_animal,
    'corr_brain_category': corr_MonkeyN_brain_category, 'corr_brain_animal': corr_MonkeyN_brain_animal,
    'corr_V1_category': corr_MonkeyN_V1_category, 'corr_V1_animal': corr_MonkeyN_V1_animal,
    'corr_V4_category': corr_MonkeyN_V4_category, 'corr_V4_animal': corr_MonkeyN_V4_animal,
    'corr_IT_category': corr_MonkeyN_IT_category, 'corr_IT_animal': corr_MonkeyN_IT_animal,
    'corr_clip_category': corr_clip_category, 'corr_clip_animal': corr_clip_animal,
    'corr_vgg_category': corr_vgg_category, 'corr_vgg_animal': corr_vgg_animal
}

# Get max values and max average row/column

def get_max(RDM_matrix):
    numeric = RDM_matrix.apply(pd.to_numeric, errors='coerce')  # Ensure float values
    min_val = numeric.min().min()
    assert np.isclose(round(min_val, 4), 0.0000), f"min_val is {min_val}, not 0.0000 to 4 decimals"
    max_val = numeric.max().max()
    RDM_matrix['row_means'] = numeric.mean(axis=1)
    max_avg_row_index = RDM_matrix['row_means'].idxmax()

    return max_val, max_avg_row_index

max_table = []
for name, rdm in rdm_matrices.items():
    max_val, max_avg_row_index = get_max(rdm)
    max_table.append([name, max_val, max_avg_row_index])

max_summary = pd.DataFrame(max_table, columns=["RDM name", "RDM maximum value", "Most dissimilar category/animal"])  # For better display
max_summary[["RDM maximum value"]] = max_summary[["RDM maximum value"]].round(3)  # Round values

# Display and save table
fig, ax = plt.subplots(figsize=(12, len(max_summary)*0.5))
ax.axis('off')
tbl = ax.table(cellText=max_summary.to_numpy().tolist(),
               colLabels=max_summary.columns.tolist(),
               cellLoc='center',
               loc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(0.9, 2.7)
for key, cell in tbl.get_celld().items():
    if key[0] == 0:
        cell.get_text().set_fontweight('bold')
plt.savefig("tables/summary_table.png", dpi=300, bbox_inches='tight')  # Save
plt.show()
