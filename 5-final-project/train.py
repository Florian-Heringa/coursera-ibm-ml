"""
Training script for VAE on QuickDraw dataset
"""

import os
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import VAE
from data import get_data_loaders, create_dummy_loader


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    print(f"[INFO] Random seed set to {seed}")


class VAETrainer:
    """Trainer class for VAE model"""
    
    def __init__(self, model, device='cpu', checkpoint_dir='./checkpoints'):
        """
        Args:
            model: VAE model instance
            device: 'cpu' or 'cuda'
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        self.train_losses = []
        self.val_losses = []
        self.train_recon_losses = []
        self.train_kl_losses = []
        
    def train_epoch(self, train_loader):
        """
        Train for one epoch
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            avg_loss, avg_recon_loss, avg_kl_loss
        """
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        num_batches = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            # Handle both (images, labels) tuple and just images
            if isinstance(batch, (tuple, list)):
                batch_images = batch[0]
            else:
                batch_images = batch
                
            # Move batch to device (keep as [batch_size, 1, 128, 128], no flattening)
            batch_images = batch_images.to(self.device)
            
            # Forward pass through VAE
            x_recon, mu, log_var, z = self.model(batch_images)
            
            # Compute loss
            total_batch_loss, recon_loss, kl_loss = self.model.compute_loss(
                batch_images, x_recon, mu, log_var
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            total_batch_loss.backward()
            clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += total_batch_loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        avg_kl_loss = total_kl_loss / num_batches
        
        return avg_loss, avg_recon_loss, avg_kl_loss
    
    def validate(self, val_loader):
        """
        Validate model on validation set
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            avg_loss
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                # Handle both (images, labels) tuple and just images
                if isinstance(batch, (tuple, list)):
                    batch_images = batch[0]
                else:
                    batch_images = batch
                    
                batch_images = batch_images.to(self.device)
                
                # Forward pass (no flattening needed)
                x_recon, mu, log_var, z = self.model(batch_images)
                total_batch_loss, _, _ = self.model.compute_loss(
                    batch_images, x_recon, mu, log_var
                )
                
                total_loss += total_batch_loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self, train_loader, val_loader, num_epochs=50, early_stopping_patience=10):
        """
        Full training loop
        
        Args:
            train_loader: DataLoader for training
            val_loader: DataLoader for validation
            num_epochs: Number of epochs
            early_stopping_patience: Patience for early stopping
        """
        best_val_loss = float('inf')
        patience_count = 0
        
        for epoch in range(num_epochs):
            print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
            
            # Train
            train_loss, train_recon, train_kl = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_recon_losses.append(train_recon)
            self.train_kl_losses.append(train_kl)
            
            # Validate
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            print(f"Train Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, KL: {train_kl:.4f})")
            print(f"Val Loss: {val_loss:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_count = 0
                self.save_checkpoint(epoch, tag="best")
                print("[INFO] Best model saved!")
            else:
                patience_count += 1
            
            # Early stopping
            if patience_count >= early_stopping_patience:
                print(f"[INFO] Early stopping after {epoch+1} epochs")
                break
        
        # Save final model
        self.save_checkpoint(num_epochs - 1, tag="final")
        
    def save_checkpoint(self, epoch, tag="checkpoint"):
        """Save model checkpoint"""
        ckpt_path = self.checkpoint_dir / f"vae_{tag}_epoch{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
        }, ckpt_path)
        print(f"[INFO] Checkpoint saved to {ckpt_path}")
    
    def load_checkpoint(self, ckpt_path):
        """Load model from checkpoint"""
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['optimizer_state'])
        print(f"[INFO] Checkpoint loaded from {ckpt_path}")
    
    def plot_losses(self, save_path="./training_curves.png"):
        """Plot training curves"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Total loss
        axes[0].plot(self.train_losses, label='Train', marker='o')
        axes[0].plot(self.val_losses, label='Val', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Total Loss')
        axes[0].legend()
        axes[0].grid()
        
        # Reconstruction vs KL
        axes[1].plot(self.train_recon_losses, label='Reconstruction', marker='o')
        axes[1].plot(self.train_kl_losses, label='KL', marker='s')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Loss Components (Train)')
        axes[1].legend()
        axes[1].grid()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"[INFO] Training curves saved to {save_path}")


def main():
    """Main training script"""
    # Setup
    set_seed(42)
    
    # Hyperparameters
    batch_size = 16  # Reduced to fit in 30-class subset (batch_size <= num_classes)
    num_epochs = 50
    latent_dim = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"[INFO] Device: {device}")
    
    # QuickDraw dataset configuration
    quickdraw_path = Path(r"C:\datasets\google-quickdraw\sketches")
    
    # Check if QuickDraw data is available
    if not quickdraw_path.exists():
        print(f"[WARNING] QuickDraw data not found at {quickdraw_path}")
        print("[INFO] Using dummy dataloaders for testing instead")
        use_real_data = False
    else:
        use_real_data = True
    
    # Create model (convolutional VAE for 128×128 images)
    model = VAE(
        image_channels=1,
        latent_dim=latent_dim,
        kl_weight=1.0/(128*128)  # Scale KL by image size (follows course pattern)
    )
    print(f"[INFO] Model created with architecture:")
    print(f"       - Input: [batch, 1, 128, 128]")
    print(f"       - Latent dim: {latent_dim}")
    print(f"       - Encoder: Conv2D (32->64->128 filters with stride=2)")
    print(f"       - Decoder: ConvTranspose (128->64->32->1 with stride=2)")
    print(f"       - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = VAETrainer(model, device=device)
    
    # Setup dataloaders
    if use_real_data:
        print("[INFO] Loading QuickDraw dataset...")
        try:
            # Use a subset of classes for faster training (adjust count as needed)
            # Full dataset: 345 classes × ~70k samples = ~24M samples
            # Subset: 25 classes × ~5k samples = ~125k samples (much faster for testing)
            selected_classes = [
                'airplane', 'apple', 'banana', 'bird', 'blueberry',
                'book', 'bus', 'butterfly', 'cactus', 'cake',
                'camera', 'car', 'cat', 'clock', 'cloud',
                'cup', 'dog', 'door', 'eye', 'fish',
                'flower', 'frog', 'hand', 'house', 'key'
            ]
            
            # Define sample counts per split for faster iteration
            train_counts = {cls: 5000 for cls in selected_classes}
            val_counts = {cls: 1000 for cls in selected_classes}
            test_counts = {cls: 1000 for cls in selected_classes}
            
            train_loader, val_loader, test_loader, class_names = get_data_loaders(
                data_dir=str(quickdraw_path),
                labels=selected_classes,
                batch_size=batch_size,
                num_workers=0,  # Set to 0 to avoid multiprocessing issues on Windows
                seed=42,
                train_counts=train_counts,
                val_counts=val_counts,
                test_counts=test_counts,
            )
            print(f"[INFO] Loaded {len(class_names)} classes for training (subset mode)")
        except Exception as e:
            print(f"[ERROR] Failed to load QuickDraw data: {e}")
            print("[INFO] Falling back to dummy dataloaders...")
            use_real_data = False
    
    if not use_real_data:
        print("[INFO] Creating dummy dataloaders for testing...")
        train_loader = create_dummy_loader(batch_size=batch_size, num_batches=10)
        val_loader = create_dummy_loader(batch_size=batch_size, num_batches=5)
        test_loader = create_dummy_loader(batch_size=batch_size, num_batches=5)
    
    # Train
    print("\n[INFO] Starting training...")
    trainer.train(train_loader, val_loader, num_epochs=num_epochs, early_stopping_patience=10)
    
    # Plot results
    trainer.plot_losses()
    print("[INFO] Training complete!")
    
    # Optional: Test on test set
    if use_real_data:
        print("\n[INFO] Evaluating on test set...")
        test_loss = trainer.validate(test_loader)
        print(f"[INFO] Test loss: {test_loss:.4f}")


if __name__ == '__main__':
    main()
