# Import packages
import h5py
import pandas as pd
import numpy as np
import os
import re

from config import base_path

monkeys = {
    "MonkeyN": {"V1": (1, 512), "V4": (513, 768), "IT": (769, 1024)},
    "MonkeyF": {"V1": (1, 512), "IT": (513, 832), "V4": (833, 1024)}
}

def get_region(electrode, region_map):
    for region, (start, end) in region_map.items():
        if start <= electrode <= end:
            return region

def extract_string(f, dataset):
    if isinstance(dataset, h5py.Reference):
        dataset = f[dataset]
    if isinstance(dataset, h5py.Dataset):
        dataset = dataset[()]
    if isinstance(dataset, np.ndarray) and dataset.dtype == object:
        dataset = dataset[0]
        if isinstance(dataset, h5py.Reference):
            dataset = f[dataset][()]
    if isinstance(dataset, np.ndarray) and dataset.dtype == np.uint16:
        try:
            return ''.join(chr(int(c)) for c in dataset.flatten() if int(c) > 0)
        except Exception as e:
            print(f"ERROR: Failed to decode dataset {dataset} - {e}")
            return "UNKNOWN"
    if isinstance(dataset, bytes):
        return dataset.decode('utf-8')
    elif isinstance(dataset, np.ndarray):
        if dataset.dtype.kind in {'S', 'U'}:
            return dataset.astype(str)[0]
    return str(dataset)

def extract_object_name(stimulus_name):
    match = re.search(r'([^/]+)(?:_\d+[a-z]?)?\.jpg$', stimulus_name)
    return match.group(1) if match else stimulus_name

def load_category_data():
    filename = os.path.join(base_path, "raw_datasets", "Concept_to_category_linking.csv")
    print(f"Loading category data from {filename}...")
    category_df = pd.read_csv(filename)
    category_df["concept"] = category_df["concept"].str.lower().str.strip()
    print(f"Loaded category data: {len(category_df)} rows.")
    return category_df

def process_monkey(monkey, regions, category_df):
    print(f"\nProcessing {monkey}...")

    things_imgs_path = os.path.join(base_path, "raw_datasets", monkey, "things_imgs.mat")
    norm_mua_path = os.path.join(base_path, "raw_datasets", monkey, "THINGS_normMUA.mat")

    if not os.path.exists(things_imgs_path) or not os.path.exists(norm_mua_path):
        print(f"Error: One or both files for {monkey} are missing.")
        return

    with h5py.File(things_imgs_path, 'r') as f_things, h5py.File(norm_mua_path, 'r') as f_mua:
        train_paths = [extract_string(f_things, f_things['train_imgs']['things_path'][i]) for i in range(len(f_things['train_imgs']['things_path']))]
        train_classes = [extract_string(f_things, f_things['train_imgs']['class'][i]) for i in range(len(f_things['train_imgs']['class']))]

        test_paths = [extract_string(f_things, f_things['test_imgs']['things_path'][i]) for i in range(len(f_things['test_imgs']['things_path']))]
        test_classes = [extract_string(f_things, f_things['test_imgs']['class'][i]) for i in range(len(f_things['test_imgs']['class']))]

        train_MUA = f_mua['train_MUA'][()].T
        test_MUA = f_mua['test_MUA'][()].T

    num_electrodes, num_train_stimuli = train_MUA.shape
    combined_data = []

    def process_data(paths, classes, mua_data, trial_type):
        for i in range(mua_data.shape[1]):
            stim_name = paths[i].lower().strip().replace("\\", "/")
            stim_class = classes[i].lower().strip()

            row = {'Stimulus_Name': stim_name, 'Class': stim_class, 'Trial_Type': trial_type}

            for j in range(num_electrodes):
                region = get_region(j + 1, regions)
                row[f'Electrode_{j+1}_Region'] = region
                row[f'Electrode_{j+1}'] = mua_data[j, i]

            combined_data.append(row)

    process_data(train_paths, train_classes, train_MUA, "train")

    test_df = pd.DataFrame({
        'Stimulus_Name': [p.lower().strip().replace("\\", "/") for p in test_paths],
        'Class': [c.lower().strip() for c in test_classes]
    })

    for j in range(num_electrodes):
        test_df[f'Electrode_{j+1}'] = test_MUA[j, :]

    grouped_test_df = test_df.groupby(['Stimulus_Name', 'Class']).mean().reset_index()

    for _, row in grouped_test_df.iterrows():
        out_row = {
            'Stimulus_Name': row['Stimulus_Name'],
            'Class': row['Class'],
            'Trial_Type': 'test'
        }
        for j in range(num_electrodes):
            region = get_region(j + 1, regions)
            out_row[f'Electrode_{j+1}_Region'] = region
            out_row[f'Electrode_{j+1}'] = row[f'Electrode_{j+1}']
        combined_data.append(out_row)

    combined_df = pd.DataFrame(combined_data)

    combined_df['Image_File'] = combined_df['Stimulus_Name'].apply(lambda x: x.split("/")[-1].lower().strip())
    combined_df['Base_Object'] = combined_df['Image_File'].apply(lambda x: re.sub(r'_\d+[a-z]?\.jpg$', '', x))

    unmatched_category = combined_df[~combined_df['Base_Object'].isin(category_df['concept'])]
    print(f"\nUnmatched Category Labels: {len(unmatched_category)}")
    if len(unmatched_category) > 0:
        print("\nFirst 5 Unmatched Category Labels:")
        print(unmatched_category[['Stimulus_Name', 'Base_Object']].head())

    print("Merging with category labels...")
    merged_df = pd.merge(combined_df, category_df[['concept', 'category_label']], left_on='Base_Object', right_on='concept', how='left')
    merged_df = merged_df.drop(columns=['concept', 'Image_File', 'Base_Object'])

    csv_path = os.path.join(base_path, "datasets", f"{monkey}_MUA_responses.csv")
    merged_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path} | Total Rows: {len(merged_df)}")

    return csv_path

# Load category data only
category_df = load_category_data()

# Process both monkeys without memorability
for monkey, region_map in monkeys.items():
    process_monkey(monkey, region_map, category_df)

print("Processing complete!")