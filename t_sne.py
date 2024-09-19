import torch
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid

import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
import os, sys
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from absl import app, flags
import warnings

def tsne_visualization_by_index(index):
    # Define the folder path
    save_dir = 'tsne_visualization'
    
    # Create the folder if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Load features and labels for the given index
    features = np.load(f'features_index_{index}.npy')
    labels = np.load(f'labels_index_{index}.npy')
    
    # Flatten features if necessary, depending on the shape
    features = features.reshape(features.shape[0], -1)  # Flatten to (n_samples, n_features)
    
    # Fit and transform with t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    features_tsne = tsne.fit_transform(features)
    
    # Plotting
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=labels, cmap='tab20', alpha=0.6)
    plt.colorbar(scatter, ticks=range(len(set(labels))))  # Adjust ticks based on the number of unique labels
    plt.title(f"t-SNE of Features for Index {index}")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    
    # Save the plot in the tsne_visualization folder
    plt.savefig(os.path.join(save_dir, f'tsne_index_{index}.png'))  # Save figure to file
    plt.close()  # Close the figure to free memory

def tsne_visualization_conditions(conditions_path, labels_path):
    # Define the folder path
    save_dir = 'tsne_visualization'
    
    # Create the folder if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Load condition vectors and labels
    conditions = np.load(conditions_path)
    labels = np.load(labels_path)
    
    # Flatten condition vectors if necessary, depending on the shape
    conditions = conditions.reshape(conditions.shape[0], -1)  # Flatten to (n_samples, n_features)
    
    # Fit and transform with t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    conditions_tsne = tsne.fit_transform(conditions)
    
    # Plotting
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(conditions_tsne[:, 0], conditions_tsne[:, 1], c=labels, cmap='tab20', alpha=0.6)
    plt.colorbar(scatter, ticks=range(len(set(labels))))  # Adjust ticks based on the number of unique labels
    plt.title("t-SNE of Condition Vectors")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    
    # Save the plot in the tsne_visualization folder
    plt.savefig(os.path.join(save_dir, 'tsne_conditions.png'))  # Save figure to file
    plt.close()  # Close the figure to free memory



for index in range(10):
    tsne_visualization_by_index(index)

# Perform t-SNE on condition vectors
tsne_visualization_conditions('condition_vectors.npy', 'condition_labels.npy')
