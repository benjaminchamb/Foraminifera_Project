import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os


TRAINED_MODELS_PATH = "/dtu/3d-imaging-center/courses/02510/groups/group_Anhinga/forams_classification/trained_models/"
DATA_PATH = "/dtu/3d-imaging-center/courses/02510/data/Foraminifera/kaggle_data/"
# FIGURES_DIR = "figures/"
FEATURES_PATH = "/dtu/3d-imaging-center/courses/02510/groups/group_Anhinga/forams_classification/features/"

print()


def plot_volume(volume, opacity=0.5, view='orthogonal'):
    # Create coordinate arrays
    x_coords = np.arange(volume.shape[0])
    y_coords = np.arange(volume.shape[1])
    z_coords = np.arange(volume.shape[2])
    
    # Define slice positions based on view type
    if view == 'orthogonal':
        # Middle points for all three planes
        x_slice = volume.shape[0]//2
        y_slice = volume.shape[1]//2
        z_slice = volume.shape[2]//2
    elif view == 'end':
        # End slices
        x_slice = volume.shape[0]-1
        y_slice = volume.shape[1]-1
        z_slice = volume.shape[2]-1
    else:
        raise ValueError("Invalid view type. Choose from 'orthogonal' or 'end'.")
            
    # Create meshgrid for each plane
    X, Y = np.meshgrid(x_coords, y_coords)
    Y_z, Z_y = np.meshgrid(y_coords, z_coords)
    X_z, Z_x = np.meshgrid(x_coords, z_coords)

    # Create figure
    fig = go.Figure()

    # XY-plane (constant z) - Axial
    fig.add_trace(go.Surface(
        x=X,
        y=Y,
        z=np.ones(X.shape) * z_slice,
        surfacecolor=volume[:, :, z_slice].T,
        colorscale='Viridis',
        opacity=opacity,
        showscale=True,
        name='Axial Slice'
    ))
    
    # YZ-plane (constant x) - Sagittal
    fig.add_trace(go.Surface(
        x=np.ones(Y_z.shape) * x_slice,
        y=Y_z,
        z=Z_y,
        surfacecolor=volume[x_slice, :, :].T,
        colorscale='Viridis',
        opacity=opacity,
        showscale=False,
        name='Sagittal Slice'
    ))
    
    # XZ-plane (constant y) - Coronal
    fig.add_trace(go.Surface(
        x=X_z,
        y=np.ones(X_z.shape) * y_slice,
        z=Z_x,
        surfacecolor=volume[:, y_slice, :].T,
        colorscale='Viridis',
        opacity=opacity,
        showscale=False,
        name='Coronal Slice'
    ))

    # Update layout
    view_name = view.capitalize()
    fig.update_layout(
        title=f'3D Volume Visualization - {view_name} View',
        width=800,
        height=800,
        scene=dict(
            aspectmode='data',
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z')
        )
    )

    fig.show()
    
def plot_slices(volume, idx_x=None, idx_y=None, idx_z=None, cmap='gray', save_path=None):
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
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.tight_layout()
        plt.show()
        
    
def plot_histogram(volume):
    # Flatten the volume to get all pixel values
    pixel_values = volume.flatten()
    
    # Create a histogram
    plt.figure(figsize=(10, 5))
    plt.hist(pixel_values, bins=50, color='deepskyblue', edgecolor='black')
    plt.title('Histogram of Pixel Intensities')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.show()
    
    
def load_model(state_dict_path, model_pl, device):
    """
    Load the pytorch lightning model from the given path.
    :param model_path: path to the model file
    :return: the loaded model
    """
    
    # Check if the model path exists
    if not os.path.exists(state_dict_path):
        raise FileNotFoundError(f"Model file not found: {state_dict_path}")
    
    # Load the weights
    state_dict = torch.load(state_dict_path, map_location=device)
    # If you saved the inner nn.Module:
    model_pl.model.load_state_dict(state_dict)
    
    model_pl.eval()
    return model_pl

def load_checkpoint(model_class, checkpoint_path, device):
    """
    Load the checkpoint from the given path.
    :param checkpoint_path: path to the checkpoint file
    :return: the loaded checkpoint
    """

    # correct: call on the class
    model = model_class.load_from_checkpoint(
        checkpoint_path,
        map_location=device      # or "cpu", or a torch.device
    )

    model.eval()                # inference mode
    model.to(device)            # move to GPU if you like
    return model