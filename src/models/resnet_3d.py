import pytorch_lightning as pl
import torch
import torchmetrics
from torchvision.models.video import r3d_18  # 3D ResNet-18
import torch.nn.functional as F


class Resnet3D(pl.LightningModule):

    def __init__(self, num_classes, pretrained=True, learning_rate=1e-4):
        super().__init__()
        
        self.num_classes = num_classes
        self.model = r3d_18(pretrained=pretrained)
        # Add dropout layer
        self.model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(in_features=self.model.fc.in_features, out_features=num_classes)
        )

        # Use focal loss for imbalanced classes
        self.criterion = torch.nn.CrossEntropyLoss()
        self.learning_rate = learning_rate


    def log_metrics(self, targets, outputs, type='train', batch_size=None):
        
        # Calculate the loss
        loss = self.criterion(outputs, targets)

        # Get the predictions of the model
        preds = torch.argmax(outputs, dim=1)
        metric_dict = {
            f'{type}_loss': loss,
            f'{type}_accuracy': torchmetrics.functional.accuracy(preds, targets, task='multiclass', average='macro', num_classes=self.num_classes),
            f'{type}_precision': torchmetrics.functional.precision(preds, targets, task='multiclass', average='macro', num_classes=self.num_classes),
            f'{type}_recall': torchmetrics.functional.recall(preds, targets, task='multiclass', average='macro', num_classes=self.num_classes),
            f'{type}_f1': torchmetrics.functional.f1(preds, targets, task='multiclass', average='macro', num_classes=self.num_classes),
        }
        # Log the metrics
        self.log_dict(
            metric_dict, 
            on_epoch=True, on_step=False,
            prog_bar=True, logger=True,
            batch_size=batch_size
        )
        return metric_dict

    def training_step(self, batch, batch_idx):
        images, targets, ids = batch
        outputs = self.model(images)
        
        metrics = self.log_metrics(targets, outputs, type='train', batch_size=len(batch[0]))
        return metrics['train_loss']

    def validation_step(self, batch, batch_idx):
        images, targets, ids = batch
        outputs = self.model(images)
        
        metrics = self.log_metrics(targets, outputs, type='val', batch_size=len(batch[0]))
        return metrics['val_loss']

    def test_step(self, batch, batch_idx):
        images, targets, ids = batch
        outputs = self.model(images)
        
        metrics = self.log_metrics(targets, outputs, type='test', batch_size=len(batch[0]))
        return metrics['test_loss']

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        return optimizer

    def inference(self, image_tensor):
        """
        Perform inference on a single image tensor.
        :param image_tensor (torch.Tensor): The input image tensor of shape (C, T, H, W).
        :return torch.Tensor: The predicted class probabilities.
        """
        self.eval()
        with torch.no_grad():
            output = self.model(image_tensor.unsqueeze(0))
            probabilities = F.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
            
        return predicted_class, probabilities
    
