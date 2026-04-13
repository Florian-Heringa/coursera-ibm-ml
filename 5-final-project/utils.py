"""
Utility functions for VAE Doodle Generator
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
from pathlib import Path


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    print(f"[INFO] Random seed set to {seed}")


def get_device():
    """Get device (GPU or CPU)"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Using device: {device}")
    return device


class ImageCache:
    """Simple cache for rendered images to avoid recomputation"""
    
    def __init__(self, cache_dir="./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get(self, key):
        """Get cached image"""
        cache_path = self.cache_dir / f"{key}.npy"
        if cache_path.exists():
            return np.load(cache_path)
        return None
    
    def put(self, key, image):
        """Cache image"""
        cache_path = self.cache_dir / f"{key}.npy"
        np.save(cache_path, image)
    
    def exists(self, key):
        """Check if key is cached"""
        cache_path = self.cache_dir / f"{key}.npy"
        return cache_path.exists()


def visualize_reconstructions(original_images, reconstructed_images, num_samples=8):
    """
    Visualize original and reconstructed images side by side
    
    Args:
        original_images: [batch_size, 1, H, W] tensor
        reconstructed_images: [batch_size, 1, H, W] tensor
        num_samples: number of samples to display
    """
    num_samples = min(num_samples, len(original_images))
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(6, 3*num_samples))
    
    for i in range(num_samples):
        # Original
        orig = original_images[i, 0].detach().cpu().numpy()
        axes[i, 0].imshow(orig, cmap='gray')
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')
        
        # Reconstructed
        recon = reconstructed_images[i, 0].detach().cpu().numpy()
        axes[i, 1].imshow(recon, cmap='gray')
        axes[i, 1].set_title('Reconstructed')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    return fig


def visualize_latent_space(latent_vectors, labels=None):
    """
    Visualize 2D latent space
    
    Args:
        latent_vectors: [num_samples, 2] array
        labels: [num_samples] array of class labels (optional)
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    if labels is not None:
        scatter = ax.scatter(latent_vectors[:, 0], latent_vectors[:, 1], 
                           c=labels, cmap='tab20', alpha=0.6, s=50)
        plt.colorbar(scatter, ax=ax, label='Class')
    else:
        ax.scatter(latent_vectors[:, 0], latent_vectors[:, 1], alpha=0.6, s=50)
    
    ax.set_xlabel('Latent Dim 1')
    ax.set_ylabel('Latent Dim 2')
    ax.set_title('2D Latent Space')
    ax.grid(True, alpha=0.3)
    
    return fig


def encode_dataset(model, dataloader, device='cpu'):
    """
    Encode all images in dataset to latent space
    
    Args:
        model: VAE model
        dataloader: DataLoader of images
        device: device to use
        
    Returns:
        latent_vectors: [num_samples, latent_dim] array
        original_images: [num_samples, 1, H, W] tensor
    """
    all_latents = []
    all_images = []
    
    model.eval()
    with torch.no_grad():
        for batch_images in dataloader:
            batch_images = batch_images.to(device)
            all_images.append(batch_images)
            
            # Flatten and encode
            batch_flat = batch_images.reshape(batch_images.size(0), -1)
            mu = model.encode(batch_flat)
            all_latents.append(mu.cpu())
    
    latent_vectors = torch.cat(all_latents, dim=0).numpy()
    original_images = torch.cat(all_images, dim=0)
    
    return latent_vectors, original_images


def reduce_to_2d(latent_vectors, method='pca'):
    """
    Reduce latent vectors to 2D for visualization
    
    Args:
        latent_vectors: [num_samples, latent_dim] array
        method: 'pca' or 'tsne'
        
    Returns:
        reduced_vectors: [num_samples, 2] array
    """
    if latent_vectors.shape[1] == 2:
        return latent_vectors
    
    if method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
        return reducer.fit_transform(latent_vectors)
    
    elif method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42)
        return reducer.fit_transform(latent_vectors)
    
    else:
        raise ValueError(f"Unknown reduction method: {method}")


def create_image_grid(images: List, num_cols=5):
    """
    Create a grid of images for visualization
    
    Args:
        images: list of PIL Images
        num_cols: number of columns in grid
        
    Returns:
        grid_image: PIL Image
    """
    from PIL import Image
    
    num_images = len(images)
    num_rows = (num_images + num_cols - 1) // num_cols
    
    # Assume all images are same size
    img_width, img_height = images[0].size
    grid_width = num_cols * img_width
    grid_height = num_rows * img_height
    
    grid = Image.new('L', (grid_width, grid_height), color=255)
    
    for idx, img in enumerate(images):
        row = idx // num_cols
        col = idx % num_cols
        x = col * img_width
        y = row * img_height
        grid.paste(img, (x, y))
    
    return grid


# TODO: Additional utilities as needed
# - Metrics computation (MS-SSIM, FID, etc.)
# - Checkpoint management
# - Configuration loading
# - Logging setup
