import os
import pandas as pd
from tifffile import imread
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torch


class ForamsDataset:
    
    def __init__(self, 
                csv_labels_path=None, 
                labelled_data_path=None, unlabeled_data_path=None, 
                volume_transforms=None,
                max_num_samples=None):
        
        self.volume_transforms = volume_transforms
        self.max_num_samples = max_num_samples
        
        self.data = []
        if csv_labels_path and labelled_data_path:
            self.load_labelled_data(csv_labels_path, labelled_data_path)
        if unlabeled_data_path:
            self.load_unlabelled_data(unlabeled_data_path)
    
    def load_labelled_data(self, csv_labels_path, labelled_data_path):
        """
        Load labelled data from a CSV file and a directory of images.
        The CSV file should contain the following columns:
        - id: the id of the volume
        - label: the label of the volume
        :param csv_labels_path: path to the CSV file
        :param labelled_data_path: path to the directory of images
        :return: None
        """
        
        # Read the CSV file
        labels_df = pd.read_csv(csv_labels_path)
        
        # Load the images
        for volume_filename in tqdm(os.listdir(labelled_data_path), desc='Loading labelled data'):
            volume_path = os.path.join(labelled_data_path, volume_filename)
            # Read the volume
            volume = imread(volume_path)
            
            volume_id = volume_filename.split('_')[2]
            # Get the labels for the volume
            volume_labels = labels_df[labels_df['id'].apply(lambda x: x.split('_')[1]) == volume_id]
            # Get the labels for the volume
            label = volume_labels['label'].values[0]
            
            self.data.append({
                'volume': volume,
                'label': label
            })
            if self.max_num_samples and len(self.data) >= self.max_num_samples:
                break
            
    def load_unlabelled_data(self, unlabeled_data_path):
        """
        Load unlabelled data from a directory of images.
        :param unlabeled_data_path: path to the directory of images
        :return: None
        """
        # Load the images
        for volume_filename in tqdm(os.listdir(unlabeled_data_path), desc='Loading unlabelled data'):
            volume_path = os.path.join(unlabeled_data_path, volume_filename)
            # Read the volume
            volume = imread(volume_path)
            
            self.data.append({
                'volume': volume,
                'label': -1
            })
            if self.max_num_samples and len(self.data) >= self.max_num_samples:
                break
    
    def __len__(self):
        """
        Return the number of labelled images.
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Return the labelled image and its label.
        """
        volume = self.data[idx]['volume']
        volume = volume.astype(np.float32) / 255.0
        label = self.data[idx]['label']
        
        # Apply the transformations to the volume
        if self.volume_transforms:
            volume = self.volume_transforms(volume)
        
        # Convert the label to a tensor
        label = torch.tensor(label, dtype=torch.long)
        
        return volume, label
    
    
    def plot_sample_slices(self, idx, idx_x=None, idx_y=None, idx_z=None, cmap='gray', save_path=None):
        """
        Plot the slices of the volume at the given indices.
        :param idx: index of the volume
        :param idx_x: index of the x slice
        :param idx_y: index of the y slice
        :param idx_z: index of the z slice
        :param cmap: colormap to use
        :param save_path: path to save the plot
        :return: None
        """
         
        volume, label = self[idx]
        
        if idx_x is None:
            idx_x = volume.shape[2] // 2
        if idx_y is None:
            idx_y = volume.shape[1] // 2
        if idx_z is None:
            idx_z = volume.shape[0] // 2
            
        fig, axes = plt.subplots(1, 3, figsize=(10, 5))
        
        if idx_x is not None:
            axes[0].imshow(volume[:, :, idx_x], cmap=cmap)
            axes[0].set_title(f'Slice at X={idx_x}')
            axes[0].axis('off')
            
        if idx_y is not None:
            axes[1].imshow(volume[:, idx_y, :], cmap=cmap)
            axes[1].set_title(f'Slice at Y={idx_y}')
            axes[1].axis('off')
            
        if idx_z is not None:
            axes[2].imshow(volume[idx_z, :, :], cmap=cmap)
            axes[2].set_title(f'Slice at Z={idx_z}')
            axes[2].axis('off')
        
        plt.suptitle(f'Label: {label}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()
    

if __name__ == "__main__":
    pass
            
            

            