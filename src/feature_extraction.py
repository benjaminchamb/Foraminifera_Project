from src.dataset import ForamsDataset
from src.utils import *
from src.models.MAE import MAE_3D_Lightning

from torch.utils.data import DataLoader
import os
import torchvision.transforms as transforms
from torchvision.models.video import r3d_18
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import warnings
# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")    


def extract_features(loader: DataLoader, model):
    model.to(device).eval()

    feature_list = []
    label_list   = []

    with torch.no_grad():
        for volumes, y in tqdm(loader, desc="Extracting features"):
            volumes = volumes.float().to(device)

            # --- forward ---
            z = model.extract_features(volumes)   # (B, D_e)
            feature_list.append(z.cpu())          # keep on CPU

            # --- collect labels ---
            label_list.append(y)                  # still a torch Tensor on CPU

    features = torch.cat(feature_list).numpy()    # (N, D_e)
    labels   = torch.cat(label_list).numpy()      # (N,) or (N, …)

    return features, labels



labelled_path = os.path.join(DATA_PATH, 'volumes/labelled')
unlabelled_path = os.path.join(DATA_PATH, 'volumes/unlabelled')
labels_csv_path = os.path.join(DATA_PATH, 'labelled.csv')

# Convert to tensor and add a channel dimension
volume_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.unsqueeze(0)),  # Add a channel dimension
])

# Load the labelled data
dataset = ForamsDataset(
    csv_labels_path=labels_csv_path, labelled_data_path=labelled_path, 
    volume_transforms=volume_transforms, 
    unlabeled_data_path=unlabelled_path,
    max_num_samples=None
)

print(f"Number of samples in dataset: {len(dataset)}")

# Create a dataloader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# 2. Extract features using ResNet3D
model_pl = load_checkpoint(
    MAE_3D_Lightning, 
    os.path.join(TRAINED_MODELS_PATH, "epoch=99-step=184300.ckpt"), 
    device)

volume_features, labels = extract_features(dataloader, model_pl)
print(f"Extracted features shape: {volume_features.shape}")
print(f"Labels shape: {labels.shape}")

# Save the features and labels
np.save(os.path.join(FEATURES_PATH, 'volume_features.npy'), volume_features)
np.save(os.path.join(FEATURES_PATH, 'labels.npy'), labels)

print(f"Features and labels saved to {FEATURES_PATH}")
print(os.listdir(FEATURES_PATH))