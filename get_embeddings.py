# Import packages
import os
import clip
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
import pandas as pd

from config import image_path

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image_paths = []
labels = []

# Walk through label folders
for label in os.listdir(image_path):
    label_folder = os.path.join(image_path, label)
    if not os.path.isdir(label_folder):
        continue
    for fname in os.listdir(label_folder):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            full_path = os.path.join(label_folder, fname)
            image_paths.append(full_path)
            labels.append(label)  # folder name is the label

# Sort to keep consistent order
image_paths, labels = zip(*sorted(zip(image_paths, labels)))

embeddings = []

for path in tqdm(image_paths):
    try:
        image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(image)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        embeddings.append(embedding.cpu().numpy())
    except Exception as e:
        print(f"Skipping {path}: {e}")

# Stack and save
embeddings = np.vstack(embeddings)
np.save("clip_embeddings.npy", embeddings)

# Save labels and filenames
df = pd.DataFrame(embeddings)
df['filename'] = [os.path.basename(p) for p in image_paths]
df['label'] = labels
df.to_csv("clip_embeddings_with_labels.csv", index=False)