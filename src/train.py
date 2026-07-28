from src.models.MAE import MAE_3D_Lightning
from src.models.resnet_3d import Resnet3D
from src.dataset import ForamsDataset
from src.utils import *

import torchvision.transforms as transforms
import pytorch_lightning as pl
import torch
import os
import wandb
from pytorch_lightning.loggers import WandbLogger
import sys

if not os.path.exists(TRAINED_MODELS_DIR):
    os.makedirs(TRAINED_MODELS_DIR)
    print(f"Created directory: {TRAINED_MODELS_DIR}")

DATA_PATH = "/zhome/a2/c/213547/group_Anhinga/forams_classification/data"

TRAIN_SPLIT = 0.8
NUM_SAMPLES = None
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 6

LEARNING_RATE = 1e-4
BATCH_SIZE = 8


def train(model_pl, train_dataloader, val_dataloader=None, model_name='network'):

    accelerator = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"accelerator: {accelerator}")

    early_stopping = pl.callbacks.EarlyStopping(
        monitor='val_loss' if val_dataloader is not None else 'train_loss',
        mode='min',
        patience=EARLY_STOPPING_PATIENCE
    )

    # Initialize wandb
    wandb_logger = WandbLogger(
        project="foramifiera-self-supervised",
        name=f"mae_{wandb.util.generate_id()}",  # Create a unique run name
        log_model=True  # This will log your model checkpoints
    )
    # Optional: Log hyperparameters
    wandb_logger.log_hyperparams({
        'learning_rate': LEARNING_RATE,
        'batch_size': BATCH_SIZE,
        'epochs': NUM_EPOCHS,
        'num_samples': NUM_SAMPLES,
    })
    
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        callbacks=[early_stopping],
        accelerator=accelerator,
        logger=wandb_logger
    )
    
    # Train the model
    trainer.fit(model_pl, train_dataloader, val_dataloader)

    
    # Test the model
    save_path = f"{TRAINED_MODELS_DIR}/{model_name}.pth"

    # Save the model
    print(f"Saving the model to {save_path}")
    torch.save(model_pl.model.state_dict(), save_path)

    # Finish the wandb run
    wandb.finish()
    
    return model_pl



if __name__ == "__main__":
    
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
        max_num_samples=NUM_SAMPLES
    )

    # Split into train, validation and test sets
    train_size = int(TRAIN_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")

    # Use the sampler only for the training loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize the model
    model_pl = Resnet3D(num_classes=2, pretrained=False, learning_rate=LEARNING_RATE)


    train(model_pl, train_loader, val_loader, model_name='resnet_normal')