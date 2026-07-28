import pytorch_lightning as pl
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
from functools import partial
import random
from src.utils import *
import torch.nn as nn
from timm.models.vision_transformer import Block
from skimage.io import imread
import sys


# --- Configuration for 3D ---
# Image size: (VOLUME_DEPTH, VOLUME_HEIGHT, VOLUME_WIDTH)
VOLUME_DEPTH = 128 # Depth of the volume
VOLUME_HEIGHT = 128 # Height of the volume
VOLUME_WIDTH = 128 # Width of the volume

PATCH_SIZE = 16   # Cubic patches (PATCH_SIZE x PATCH_SIZE x PATCH_SIZE)
NUM_CHANNELS = 1  # Number of channels in the input volume (3 for RGB, 1 for grayscale)

# Calculate number of patches in 3D
NUM_PATCHES_D = VOLUME_DEPTH // PATCH_SIZE
NUM_PATCHES_H = VOLUME_HEIGHT // PATCH_SIZE
NUM_PATCHES_W = VOLUME_WIDTH // PATCH_SIZE
NUM_PATCHES = NUM_PATCHES_D * NUM_PATCHES_H * NUM_PATCHES_W


class PatchEmbed(nn.Module):
    """ 
    3D Image to Patch Embedding
    This module takes a 3D image and divides it into patches.
    It uses a 3D convolutional layer to project the patches into a higher-dimensional space.
    
    Code from: https://github.com/LangDaniel/MAEMI/blob/main/training/mae3d/util/helpers.py
    3D Masked Autoencoders with Application to Anomaly Detection in Non-Contrast Enhanced Breast MRI
    """
    def __init__(self, img_size: tuple, patch_size: tuple, in_channels=1, embed_dim=768):
        """
        :param img_size: Size of the input image (depth, height, width)
        :param patch_size: Size of the patches (depth, height, width)
        :param in_channels: Number of input channels (e.g., 1 for grayscale, 3 for RGB)
        :param embed_dim: Dimension of the embedding space
        """
        super().__init__()
        assert len(img_size) == 3 and len(patch_size) == 3, f'''
            dimension != 3: img_size={img_size} patch_size={patch_size}'''

        self.img_size = np.array(img_size)
        self.patch_size = np.array(patch_size)

        assert not (self.img_size % self.patch_size).any(), f'''
            image size module patch size error'''

        # Compute the number of patches in each dimension
        # (depth, height, width)
        self.dim_num_patches = (self.img_size // self.patch_size).astype(int)
        self.num_patches = np.prod(self.dim_num_patches)

        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x): 
        """
        x: (B, C, H, W, D) -- (batch size, channels, height, width, depth)
        x: (B, L, D) -- (batch size, number of patches, embedding dimension)
        """
        x = self.proj(x)                        # (B,embed,D',H',W')
        x = x.flatten(2).transpose(1, 2)        # (B, L, embed)
        return x
# ----------------------------------------------------------------------------------------



class MaskedAutoencoderViT_3D(nn.Module):
    """ 
    Masked Autoencoder with VisionTransformer backbone
    from: https://github.com/LangDaniel/MAEMI/blob/main/training/mae3d/models_mae.py
    3D Masked Autoencoders with Application to Anomaly Detection in Non-Contrast Enhanced Breast MRI
    """
    def __init__(
        self, image_size, patch_size, 
        in_channels, embed_dim=256*3, depth=24, num_heads=16,
        decoder_embed_dim=128*3, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        
        super().__init__()

        # Check sizes and calculate patch numbers 
        self.image_size = np.array(image_size)
        self.patch_size = np.array(patch_size)
        assert not (self.image_size % self.patch_size).any(), \
            f'''img_size modulo patch_size error''' 
        self.num_patches = (self.image_size // self.patch_size).astype(int)
        total_num_patches = np.prod(self.num_patches)
        self.in_channels = in_channels

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(
            image_size, patch_size, 
            in_channels, embed_dim)

        self.cls_token = nn.Parameter(
            data=torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            data=torch.zeros(1, total_num_patches + 1, embed_dim), 
            requires_grad=True)

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, 
                mlp_ratio=mlp_ratio, 
                qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)])
        self.norm = norm_layer(embed_dim)

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(
            in_features=embed_dim, out_features=decoder_embed_dim, 
            bias=True)

        self.mask_token = nn.Parameter(
            data=torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            data=torch.zeros(1, total_num_patches + 1, decoder_embed_dim), 
            requires_grad=True)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(
                dim=decoder_embed_dim, num_heads=decoder_num_heads, 
                mlp_ratio=mlp_ratio, 
                qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, np.prod(patch_size) * in_channels, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    def initialize_weights(self):
        """
        Initialize weights for the model
        """
        torch.nn.init.normal_(self.pos_embed, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed, std=.02)

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Initialize weights for nn.Linear and nn.LayerNorm
        :param m: module
        """
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """ 
        imgs: (N, C, H, W, D)
        x: (N, L, patch_size**3 * C)
        """
        p = self.patch_size # = (p_h, p_w, p_d)
        c = self.in_channels 
        h, w, d = self.num_patches 
    
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p[0], w, p[1], d, p[2])) 
        x = torch.einsum('nchpwqdo->nhwdpqoc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w * d, np.prod(p) * c)) 
        return x
        
    def unpatchify(self, x): 
        """ 
        x: (N, L, patch_size**3 * C)
        imgs: (N, C, H, W, D)
        """
        p = self.patch_size
        h, w, d = self.num_patches 
        assert h * w * d == x.shape[1], f'''
            size error: ({h}, {w}, {d}) vs {x.shape[1]}'''
    
        x = x.reshape(shape=(x.shape[0], h, w, d, p[0], p[1], p[2], -1))
        x = torch.einsum('nhwdpqoc->nchpwqdo', x)
        imgs = x.reshape(shape=(x.shape[0], -1, h * p[0], w * p[1], d * p[2])) 
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        """
        Decode the latent representation to reconstruct the original image.
        :param x: [N, L, D], sequence
        :param ids_restore: [N, L], index to restore the original order
        :return: [N, L, p*p*3], prediction
        """
        
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        Compute the loss between the original image and the predicted image.
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        """
        Forward pass of the model.
        :param imgs: [N, C, H, W, D], input image
        :param mask_ratio: ratio of patches to be masked (the bigger the ratio, the more patches are masked)
        :return: loss, pred, mask
        """
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*p*c]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

    @torch.no_grad()
    def encode(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Deterministic encoder forward pass **without** any masking or shuffling.

        Args
        ----
        imgs : (B, C, D, H, W)  input volume.

        Returns
        -------
        z : (B, 1+L, D_e)       latent sequence; z[:,0] is the CLS token.
        """
        # patch -> embed
        x = self.patch_embed(imgs)                          # (B, L, D_e)

        # add positional enc. (no CLS slot)
        x = x + self.pos_embed[:, 1:, :]

        # prepend CLS token
        cls_tok = self.cls_token + self.pos_embed[:, :1, :] # (1, 1, D_e)
        x = torch.cat([cls_tok.expand(x.size(0), -1, -1),   # (B, 1, D_e)
                       x], dim=1)                           # (B, 1+L, D_e)

        # transformer encoder
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x                                            # (B, 1+L, D_e)


class MAE_3D_Lightning(pl.LightningModule):
    
    def __init__(self, learning_rate=1e-4, weight_decay=1e-4):
        """
        Initialize the MAE model.
        :param learning_rate: Learning rate for the optimizer.
        :param weight_decay: Weight decay for the optimizer.
        """
        super().__init__()
        
        self.model = MaskedAutoencoderViT_3D(
            image_size=(VOLUME_DEPTH, VOLUME_HEIGHT, VOLUME_WIDTH),
            patch_size=(PATCH_SIZE, PATCH_SIZE, PATCH_SIZE),
            in_channels=NUM_CHANNELS,
            embed_dim=768, depth=12, num_heads=12,
            decoder_embed_dim=384, decoder_depth=4, decoder_num_heads=16,
            mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
    @torch.no_grad()
    def extract_features(self, volumes: torch.Tensor,
                         pool: str = "cls") -> torch.Tensor:
        """
        Args
        ----
        volumes : (B, C, D, H, W)
        pool    : "cls" | "mean"  how to obtain a single feature vector.

        Returns
        -------
        feats   : (B, D_e) if pooled, else (B, 1+L, D_e)
        """
        z = self.model.encode(volumes)          # (B, 1+L, D_e)
        if pool == "cls":
            return z[:, 0]                      # (B, D_e)
        elif pool == "mean":
            return z.mean(1)                    # (B, D_e)
        else:
            return z                            # full sequence
    
    def _shared_step(self, batch, batch_idx, step_name):
        """
        Helper function for common logic in training, validation, and test steps.
        """
        pixel_values, labels = batch
       
        # Pass pixel values to the model. It handles masking and reconstruction internally.
        # The model outputs include the loss calculated *only* on the masked patches.
        loss, predictions, mask = self.model(pixel_values)

        # Log the loss
        self.log(
            f'{step_name}_loss', loss, 
            on_step=(step_name=='train'), on_epoch=True, 
            prog_bar=True, logger=True, sync_dist=True)

        return loss
    
    def training_step(self, batch, batch_idx):
        """
        Performs a single training step.
        """
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        """
        Performs a single validation step.
        """
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        """
        Performs a single test step.
        """
        return self._shared_step(batch, batch_idx, "test")
    
    def configure_optimizers(self):
        """
        Configures the optimizer and potentially a learning rate scheduler.
        """
        # Use AdamW optimizer, common for Transformer models
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        return optimizer
    
    @torch.no_grad()
    def inference(self, volumes, mask_ratio=0.75):
        """
        Performs inference on a single image.
        """
        self.model.eval()
        
        loss, predictions, masks = self.model(volumes, mask_ratio=mask_ratio)
        return predictions, masks
    
    def plot_inference(self, volumes, reconstructed_patches, masks, save_path=None, cmap='gray'):
        """
        Plots the original image, reconstructed patches, and the mask.
        :param image: Original image tensor of shape (num_channels, height, width).
        :param reconstructed_patches: Reconstructed patches tensor of shape (num_patches, patch_dim).
        :param mask: Mask tensor of shape (num_patches) indicating which patches are masked (1) and which are not (0).
        :param save_path: Path to save the plot. If None, the plot will be shown.
        :return: None
        """
        
        # Unpatchify the reconstructed patches to get the full image size
        reconstructed_volumes = self.model.unpatchify(reconstructed_patches)
        
        # Convert tensors to numpy arrays for plotting
        volumes_np = volumes.cpu().numpy()
        reconstructed_volumes_np = reconstructed_volumes.cpu().numpy()
        
        masks_np = masks.cpu().numpy()
        
        num_volumes = volumes_np.shape[0]
        fig, axes = plt.subplots(num_volumes, 3, figsize=(10, 5 * num_volumes))
        if num_volumes == 1:
            axes = np.expand_dims(axes, axis=0)
            
        for i in range(num_volumes):
            volume = volumes_np[i, 0]  # Get the first channel
            reconstructed_volume = reconstructed_volumes_np[i, 0]  # Get the first channel
            mask = masks_np[i]
            # Reshape the mask to the original volume shape
            mask = mask.reshape(self.model.num_patches[0], self.model.num_patches[1], self.model.num_patches[2])
            
            slice_idx = volume.shape[0] // 2
            mask_idx = mask.shape[0] // 2
            
            axes[i, 0].imshow(volume[slice_idx], cmap=cmap)
            axes[i, 0].set_title('Original Volume')
            axes[i, 0].axis('off')
            axes[i, 1].imshow(reconstructed_volume[slice_idx], cmap=cmap)
            axes[i, 1].set_title('Reconstructed Volume')
            axes[i, 1].axis('off')
            axes[i, 2].imshow(mask[mask_idx], cmap='gray')
            axes[i, 2].set_title('Mask')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        
        
if __name__ == "__main__":
    
    DATA_PATH = 'data/'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    print("Using device:", device)
    
    # Create the model
    model_pl = MAE_3D_Lightning()
    model_pl.to(device)
    print("Model created\n")
    
    print(f"\tNumber of patches: {NUM_PATCHES}")
    print(f"\tPatch size: {PATCH_SIZE}x{PATCH_SIZE}x{PATCH_SIZE}")
    
    # Read a random 3D volume
    volumes_path = os.path.join(DATA_PATH, 'volumes', 'volumes', 'labelled')
    random_volume_path = os.path.join(volumes_path, random.choice(os.listdir(volumes_path)))
    
    # Read the volume
    volume = imread(random_volume_path)
    # Convert to tensor
    volume_tensor = torch.tensor(volume, dtype=torch.float32)
    # Add a channel dimension
    volume_tensor = volume_tensor.unsqueeze(0)
    # Add a batch dimension
    volume_tensor = volume_tensor.unsqueeze(0)
    # Reorder to (B,C,H,W,D) because PatchEmbed assumes that order
    # volume_tensor = volume_tensor.permute(0, 1, 3, 4, 2).contiguous() # (1, 1, H, W, D)
    # Move the volume to the same device as the model
    volume_tensor = volume_tensor.to(device)
    
    # Pass the volume through the model
    model_pl.training_step(volume_tensor, 0)
    print("Volume passed through the model\n")
    
    # Test the inference method
    predictions, mask = model_pl.inference(volume_tensor)
    print(f"Predictions shape: {predictions.shape}")
    # Plot the original volume and the reconstructed volume
    model_pl.plot_inference(volume_tensor, predictions, mask, save_path='inference.png')
    
   