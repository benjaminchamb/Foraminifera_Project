import torch
import os
import torchvision.transforms as transforms

from src.models.MAE import MAE_3D_Lightning
from src.dataset import ForamsDataset
from src.utils import *


if __name__ == "__main__":
    # Load the model state dictionary
    # model_path = os.path.join(TRAINED_MODELS_DIR, "mae.pth")
    # # Load the model
    # model = load_model(model_path, MAE_3D_Lightning())
    # print(f"Model loaded from {model_path}")

    # Load the model checkpoint
    model = load_checkpoint(
        MAE_3D_Lightning, 
        "foramifiera-self-supervised/e3v2l2ga/checkpoints/epoch=99-step=184300.ckpt")

    
    # 2. Load the dataset
    # Convert to tensor and add a channel dimension
    volume_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.unsqueeze(0)),  # Add a channel dimension
    ])
    
    # Load the dataset
    dataset = ForamsDataset(
        csv_labels_path=os.path.join(DATA_PATH, "labelled.csv"), 
        labelled_data_path=os.path.join(DATA_PATH, "volumes", "volumes", "labelled"),
        unlabeled_data_path=os.path.join(DATA_PATH, "volumes", "volumes", "unlabelled"),
        volume_transforms=volume_transforms,
        max_num_samples=20
    )
    # Create a dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=False
    )
    
    # 3. Get a batch of data
    for i, batch in enumerate(dataloader):
        mask_ratio = i / len(dataloader) 
        volumes, labels = batch
        # Move to GPU
        volumes = volumes.float().to("cuda")
        labels = labels.float().to("cuda")
        predictions, masks = model.inference(volumes, mask_ratio=mask_ratio)
        model.plot_inference(volumes, predictions, masks, save_path=os.path.join(FIGURES_DIR, f"inference{mask_ratio:.2f}.png"), cmap=None)
