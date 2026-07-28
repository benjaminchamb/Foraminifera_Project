# Foraminifera Semi-Supervised Classification

A project on classifying planktonic foraminifera from 3D µCT volumes, tackling a highly imbalanced, mostly-unlabeled classification problem with 3D CNNs, a self-supervised Masked Autoencoder (MAE), and pseudo-label propagation.

Built for the Forams Classification 2025 Kaggle challenge (DTU course project).

**Authors**: Maja Gojska, Benjamin Chambaudet, Grzegorz Pawlak

## Problem

Planktonic foraminifera ("forams") are tiny marine organisms whose calcium carbonate shells (tests) are preserved in sea sediment for millions of years. Identifying which species are present in a sediment sample gives insight into past ocean/climate conditions. µCT scanning lets thousands of forams be scanned at once — but manually labeling each one is slow, so this project targets species classification with minimal labeled data.
- **18,426** volumetric scans total, each 128×128×128 voxels
- Only **210** are labeled (15 samples per class, 14 known species + 1 "unknown" class for broken fragments/noise)
- **18,216** volumes are unlabeled, the core challenge is making good use of them
- Evaluation metric: macro F1-score

See *kaggle.md* for the full data/challenge description (file naming conventions, folder layout, submission format).

## Approach

### 1.Supervised baselines
Two baseline classifiers were trained directly on the small labeled set:
| Model | Data format | Notes |
|--------|-------------|-------|
| ResNet3D-18 | Full 3D volume | 3D-specific augmentations: random flips, 3D rotations, Gaussian noise |
| ResNet2D-18 | 45 stacked 2D slices (15 per axis) | 2D augmentations: horizontal/vertical flips, rotations, random resized crop, Gaussian blur |

Trained with cross-entropy loss, Adam, lr = 1e-4.

### Results (test set F1):
| Model | Augmentation | F1 |
|--------|-------------|-------|
| ResNet2D | Yes | 0.458 |
| ResNet2D | No | 0.456 |
| ResNet3D | Yes | 0.232 |
| ResNet3D | No | 0.216 |

ResNet2D on slice representations was noticeably more stable and accurate than full-volume ResNet3D, which overfit on the tiny labeled set.

### 2. Self-supervised representation learning

A 3D Masked Autoencoder (MAE) was trained on the entire dataset (labeled + unlabeled) to learn general-purpose representations, encoding each 128³ volume into a 768-dimensional feature vector without needing labels.

### 3. Semi-supervised pseudo-labeling

Using the MAE's learned features:

1. Train a logistic regression on the 210 labeled feature vectors
2. Propagate labels to unlabeled samples via feature similarity
3. Keep only predictions with confidence > 0.9 as pseudo-labels
4. Retrain logistic regression on the expanded (real + pseudo) label set
5. Repeat while F1 keeps improving

### Results:

| Model | F1 |
|-------------|-------------------------|
| Logistic regression on MAE features (labeled only) | 0.639 |
| + pseudo-labels | **0.704** |

This outperformed both supervised CNN baselines, showing the value of the unlabeled data once a good feature space is learned.

## Key findings

- ResNet3D benefited from 3D augmentations but had unstable validation loss (overfitting on so few labels).
- ResNet2D, despite discarding most of the volume, generalized better than ResNet3D.
- The MAE encoder learned compact, informative features from raw 3D volumes without supervision; two of the fourteen classes formed clearly separable clusters in feature space, while others remained mixed.
- Pseudo-labeling on top of MAE features gave the best overall result.

## Repository Structure

.
├── src/
│   ├── __init__.py
│   ├── dataset.py               # ForamsDataset: loads labeled and unlabeled .tif volumes
│   ├── train.py                 # Training loop (PyTorch Lightning + Weights & Biases logging)
│   ├── evaluate.py              # Loads a checkpoint and evaluates the MAE model
│   ├── feature_extraction.py    # Extracts MAE encoder features from all volumes
│   ├── feature_investigator.py  
│   ├── utils.py                 # Utility functions: paths, volume visualization (3D, slices, histograms), checkpoint loading
│   ├── eda                
│   └── models/
│       ├── __init__.py
│       ├── MAE.py               # MAE_3D_Lightning: masked autoencoder and feature extraction
│       └── resnet_3d.py         # ResNet3D: PyTorch Lightning wrapper around torchvision's r3d_18
├── Report/
│   └── Poster_foraminifera.pdf      # Project poster summarizing the methods and results
├── .gitignore
└── kaggle.md                    # Challenge and dataset description

## Requirements
pip install torch torchvision pytorch-lightning torchmetrics tifffile pandas numpy matplotlib plotly tqdm wandb scikit-learn
- A CUDA-capable GPU is strongly recommended (3D CNNs + MAE on 128³ volumes are memory-heavy).
- Training logs and checkpoints are tracked via Weights & Biases (wandb.init in train.py) — you'll need a wandb account or should switch to a different logger.

## Future Work

- Explore other 3D architectures (e.g. 3D DenseNet) for better parameter efficiency
- Additional augmentation, focal loss, or class-balanced sampling to address class imbalance
- Alternative self-supervised objectives for more discriminative features across all classes
- Non-linear classifiers (e.g. a small MLP) on top of MAE features instead of logistic regression

## Reference
F. Ramzan, M. U. Khan, A. Rehmat, S. Iqbal, T. Saba, A. Rehman, and Z. Mehmood. "A deep learning approach for automated diagnosis and multi-class classification of Alzheimer's disease stages using resting-state fMRI and residual neural networks." Journal of Medical Systems, 44, 2019.

