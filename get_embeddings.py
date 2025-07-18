# Import packages
import os
import clip
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import pandas as pd

from config import image_path, pretrained_model, data_base_path

device = "cuda" if torch.cuda.is_available() else "cpu"
if pretrained_model == "ViT-B/32":
    model, preprocess = clip.load(pretrained_model, device=device)
elif pretrained_model == "VGG16":
    model = models.vgg16(pretrained=True).to(device)
    model.classifier = torch.nn.Sequential(*list(model.classifier.children())[:-1])  # keep up to second-last layer
    model.eval()
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

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
image_paths, labels = zip(*sorted(zip(image_paths, labels)))  # type: ignore

embeddings = []

for path in tqdm(image_paths):
    try:
        image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            if pretrained_model == "ViT-B/32":
                embedding = model.encode_image(image)
            elif pretrained_model == "VGG16":
                embedding = model(image)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        embeddings.append(embedding.cpu().numpy())
    except Exception as e:
        print(f"Skipping {path}: {e}")

# Stack and save
embeddings = np.vstack(embeddings)  # type: ignore
if pretrained_model == "ViT-B/32":
    np.save(os.path.join(data_base_path, "datasets", "clip_embeddings.npy"), embeddings)
elif pretrained_model == "VGG16":
    np.save(os.path.join(data_base_path, "datasets", "vgg16_embeddings.npy"), embeddings)

# Save labels and filenames
df = pd.DataFrame(embeddings)
df['filename'] = [os.path.basename(p) for p in image_paths]
df['label'] = labels
if pretrained_model == "ViT-B/32":
    df.to_csv(os.path.join(data_base_path, "datasets", "clip_embeddings_with_labels.csv"), index=False)
elif pretrained_model == "VGG16":
    df.to_csv(os.path.join(data_base_path, "datasets", "vgg16_embeddings_with_labels.csv"), index=False)