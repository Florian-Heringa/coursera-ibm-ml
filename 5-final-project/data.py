"""
QuickDraw Dataset and DataLoader setup
Adapted from course final-project implementation with caching and balanced sampling.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from pathlib import Path
from PIL import Image, ImageDraw


class QuickDrawDataset(Dataset):
    """
    PyTorch Dataset for Google QuickDraw doodles
    
    Loads stroke data from .npz files, renders to 128×128 grayscale images,
    and caches for efficiency. Uses balanced sampling per label.
    
    Expected format: numpy arrays of shape [num_samples, max_strokes, points, 3]
    (strokes stored as cumulative deltas with (dx, dy, pen_up) coordinates)
    """
    
    def __init__(self, labels, data_path, split="train", counts=None, max_length=None, cache_dir=None):
        """
        Args:
            labels (list): List of class names (e.g., ['airplane', 'apple', 'doodle'])
            data_path (str): Path to directory containing .npz files
            split (str): 'train', 'valid', or 'test'
            counts (dict): Override sample counts per label (if None, uses full dataset)
            max_length (int): Max stroke length to process (optional)
            cache_dir (str): Directory to cache rendered images (auto-created if None)
        """
        self.labels = labels
        self.data_path = Path(data_path)
        self.split = split
        self.max_length = max_length
        self.label_files = [self.data_path / f"{label}.npz" for label in labels]
        
        # Create cache directory for rendered images
        if cache_dir is None:
            cache_dir = self.data_path.parent / f"cache_{split}"
        self.cache_dir = Path(cache_dir)
        self._cache_dir_created = False
        
        # Get sample counts per label
        if counts is None:
            self.counts = {}
            for fpath, label in zip(self.label_files, labels):
                try:
                    with np.load(fpath, encoding="latin1", allow_pickle=True) as f:
                        self.counts[label] = len(f[split])
                except Exception as e:
                    print(f"Warning: Could not load {label} ({fpath}): {e}")
                    self.counts[label] = 0
        else:
            self.counts = counts
    
    def _get_cache_path(self, label_idx, sample_idx):
        """Get cache file path for a specific sample"""
        return self.cache_dir / f"label_{label_idx}_sample_{sample_idx}.npy"
    
    def _render_image(self, stroke):
        """
        Render stroke data (cumulative deltas) to 128×128 grayscale image [0, 1]
        
        Args:
            stroke: [num_points, 3] array with (dx, dy, pen_up) entries
            
        Returns:
            img_array: [128, 128] float32 array in [0, 1]
        """
        if self.max_length is not None and len(stroke) > self.max_length:
            stroke = stroke[:self.max_length]
        
        # Convert cumulative deltas to absolute coordinates
        xy = np.cumsum(stroke[:, :2], axis=0).astype(np.float32)
        pen_up = stroke[:, 2].astype(bool)
        
        # Normalize to canvas
        canvas_size, pad = 128, 12
        x_min, y_min = xy.min(axis=0)
        x_max, y_max = xy.max(axis=0)
        span = max(x_max - x_min, y_max - y_min, 1.0)  # Avoid division by zero
        scale = (canvas_size - 2 * pad) / span
        
        # Scale and center points on canvas
        pts = np.empty_like(xy)
        pts[:, 0] = (xy[:, 0] - x_min) * scale + pad
        pts[:, 1] = (xy[:, 1] - y_min) * scale + pad
        
        # Create PIL image and draw strokes
        pil_img = Image.new("L", (canvas_size, canvas_size), 255)  # White background
        draw = ImageDraw.Draw(pil_img)
        
        # Draw each stroke segment (between pen_up events)
        start = 0
        for end in np.where(pen_up)[0]:
            seg = pts[start:end + 1]
            if len(seg) > 1:
                # Draw line with width=3 pixels, black (0) on white (255)
                draw.line([tuple(p) for p in seg], fill=0, width=3)
            start = end + 1
        
        # Draw final segment if exists
        if start < len(pts) - 1:
            seg = pts[start:]
            if len(seg) > 1:
                draw.line([tuple(p) for p in seg], fill=0, width=3)
        
        # Convert to float [0, 1]
        return np.array(pil_img, dtype=np.float32) / 255.0
    
    def __len__(self):
        return sum(self.counts.values())
    
    def __getitem__(self, index):
        """
        Get item by composite index (label_idx, sample_idx)
        
        Args:
            index (tuple): (label_idx, sample_idx)
            
        Returns:
            image_tensor: [1, 128, 128] in [0, 1]
            label_idx: integer class index
        """
        label_idx, sample_idx = index
        
        # Ensure cache directory exists
        if not self._cache_dir_created:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                self._cache_dir_created = True
            except Exception:
                pass  # Dir may already exist from another process
        
        cache_path = self._get_cache_path(label_idx, sample_idx)
        
        # Try to load from cache first
        if cache_path.exists():
            img_array = np.load(cache_path)
        else:
            # Load from npz and render
            with np.load(self.label_files[label_idx], encoding="latin1", allow_pickle=True) as f:
                stroke = f[self.split][sample_idx]
            
            img_array = self._render_image(stroke)
            
            # Save to cache for future use
            try:
                np.save(cache_path, img_array)
            except Exception:
                pass  # File may have been saved by another worker
        
        # Return as tensor [1, 128, 128] (1 channel grayscale)
        image_tensor = torch.from_numpy(img_array).unsqueeze(0)
        return image_tensor, label_idx


class DifferentLabelBatchSampler(Sampler):
    """
    Custom batch sampler that ensures each batch contains samples from different labels.
    Useful for balanced training when dataset has multiple classes.
    
    Example: batch_size=4, 4 different random labels selected per batch.
    """
    
    def __init__(self, label_counts, batch_size, steps_per_epoch, seed=42):
        """
        Args:
            label_counts (dict): {label_name: count} or can use label indices
            batch_size (int): Must be <= number of available labels
            steps_per_epoch (int): Number of batches per epoch
            seed (int): Random seed for reproducibility
        """
        self.label_counts = label_counts
        self.counts_list = list(label_counts.values())
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.valid_labels = [i for i, c in enumerate(self.counts_list) if c > 0]
        
        if batch_size > len(self.valid_labels):
            raise ValueError(
                f"batch_size={batch_size} > number of available labels={len(self.valid_labels)}"
            )
        
        self.g = torch.Generator()
        self.g.manual_seed(seed)
    
    def __len__(self):
        return self.steps_per_epoch
    
    def __iter__(self):
        """Yield batches of (label_idx, sample_idx) tuples"""
        for _ in range(self.steps_per_epoch):
            # Randomly select batch_size different labels
            label_perm = torch.randperm(len(self.valid_labels), generator=self.g)[:self.batch_size]
            chosen_labels = [self.valid_labels[i] for i in label_perm.tolist()]
            
            # For each chosen label, select a random sample
            batch = []
            for label_idx in chosen_labels:
                sample_idx = int(
                    torch.randint(self.counts_list[label_idx], (1,), generator=self.g).item()
                )
                batch.append((label_idx, sample_idx))
            
            yield batch


def collate_fn(batch):
    """
    Collate function to combine batch items into tensors
    
    Args:
        batch: List of (image_tensor, label_idx) tuples
        
    Returns:
        images: [batch_size, 1, 128, 128] stacked tensor
        labels: [batch_size] label indices
    """
    images, labels = zip(*batch)
    return torch.stack(images, dim=0), torch.tensor(labels, dtype=torch.long)


def get_data_loaders(
    data_dir="C:\\datasets\\google-quickdraw\\sketches",
    labels=None,
    batch_size=32,
    num_workers=0,
    seed=42,
    train_counts=None,
    val_counts=None,
    test_counts=None,
):
    """
    Create train, validation, and test dataloaders for QuickDraw dataset
    
    Args:
        data_dir (str): Path to directory containing .npz files
        labels (list): List of class names. If None, auto-discover from directory
        batch_size (int): Batch size
        num_workers (int): Number of data loading workers
        seed (int): Random seed for reproducibility
        train_counts, val_counts, test_counts (dict): Override sample counts per split
        
    Returns:
        train_loader, val_loader, test_loader: DataLoaders
        class_names: List of class names used
    """
    data_path = Path(data_dir)
    
    # Auto-discover classes if not provided
    if labels is None:
        labels = sorted([f.stem for f in data_path.glob("*.npz")])
    
    print(f"[INFO] Found {len(labels)} classes: {labels[:5]}{'...' if len(labels) > 5 else ''}")
    
    # Create datasets
    train_dataset = QuickDrawDataset(
        labels, data_dir, split="train", counts=train_counts
    )
    val_dataset = QuickDrawDataset(
        labels, data_dir, split="valid", counts=val_counts
    )
    test_dataset = QuickDrawDataset(
        labels, data_dir, split="test", counts=test_counts
    )
    
    # Calculate steps per epoch
    train_steps = len(train_dataset) // batch_size
    val_steps = len(val_dataset) // batch_size
    test_steps = len(test_dataset) // batch_size
    
    print(f"[INFO] Train: {len(train_dataset)} samples ({train_steps} steps)")
    print(f"[INFO] Val: {len(val_dataset)} samples ({val_steps} steps)")
    print(f"[INFO] Test: {len(test_dataset)} samples ({test_steps} steps)")
    
    # Create batch samplers
    train_sampler = DifferentLabelBatchSampler(
        train_dataset.counts, batch_size, train_steps, seed=seed
    )
    val_sampler = DifferentLabelBatchSampler(
        val_dataset.counts, batch_size, val_steps, seed=seed + 1
    )
    test_sampler = DifferentLabelBatchSampler(
        test_dataset.counts, batch_size, test_steps, seed=seed + 2
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=test_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    
    return train_loader, val_loader, test_loader, labels


def create_dummy_loader(batch_size=32, num_batches=10, image_size=128):
    """
    Create a dummy dataloader with random data for testing
    
    Args:
        batch_size (int): Batch size
        num_batches (int): Number of batches to generate
        image_size (int): Image size (square, e.g., 128)
        
    Yields:
        images: [batch_size, 1, image_size, image_size]
        labels: [batch_size]
    """
    for _ in range(num_batches):
        images = torch.rand(batch_size, 1, image_size, image_size)
        labels = torch.randint(0, 345, (batch_size,))  # QuickDraw has ~345 classes
        yield images, labels
