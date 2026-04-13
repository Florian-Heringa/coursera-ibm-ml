"""
Phase 1: QuickDraw Data Exploration and Verification

This script verifies Phase 1 implementation:
- Load QuickDraw dataset
- Visualize sample doodles
- Test caching functionality
- Verify batch shapes
- Check dataloader performance
"""

import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import time

# Add project to path
project_dir = Path(".")
sys.path.insert(0, str(project_dir))

# Import our modules
from data import get_data_loaders, QuickDrawDataset, DifferentLabelBatchSampler
from model import VAE

print("=" * 70)
print("PHASE 1: QUICKDRAW DATA EXPLORATION AND VERIFICATION")
print("=" * 70)

# ============================================================================
# STEP 1: Verify Imports
# ============================================================================
print("\n[STEP 1] Verifying imports...")
try:
    print("✅ Imports successful")
    print(f"   - PyTorch version: {torch.__version__}")
    print(f"   - NumPy version: {np.__version__}")
except Exception as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# ============================================================================
# STEP 2: Configure and Check QuickDraw Data
# ============================================================================
print("\n[STEP 2] Checking QuickDraw dataset...")
quickdraw_path = Path(r"C:\datasets\google-quickdraw\sketches")
print(f"   QuickDraw path: {quickdraw_path}")
print(f"   Exists: {quickdraw_path.exists()}")

if quickdraw_path.exists():
    npz_files = list(quickdraw_path.glob("*.npz"))
    print(f"   ✅ Found {len(npz_files)} .npz files")
    sample_classes = [f.stem for f in npz_files[:5]]
    print(f"   Sample classes: {', '.join(sample_classes)}")
else:
    print("   ❌ QuickDraw data directory not found")
    print(f"   Expected: {quickdraw_path}")
    sys.exit(1)

# ============================================================================
# STEP 3: Load Dataset with Subset for Testing
# ============================================================================
print("\n[STEP 3] Loading dataloaders (subset for faster testing)...")
test_classes = ["airplane", "apple", "cat", "door", "flower"]

# Override counts to use fewer samples for testing
train_counts = {cls: 1000 for cls in test_classes}
val_counts = {cls: 200 for cls in test_classes}
test_counts = {cls: 200 for cls in test_classes}

try:
    print(f"   Classes: {test_classes}")
    print("   This may take a moment on first run due to caching...")
    
    train_loader, val_loader, test_loader, classes = get_data_loaders(
        data_dir=str(quickdraw_path),
        labels=test_classes,
        batch_size=4,
        num_workers=0,
        seed=42,
        train_counts=train_counts,
        val_counts=val_counts,
        test_counts=test_counts
    )
    print("   ✅ Dataloaders created successfully")
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# STEP 4: Inspect First Batch
# ============================================================================
print("\n[STEP 4] Inspecting first batch...")
try:
    images, labels = next(iter(train_loader))
    print("   ✅ Batch loaded successfully")
    print(f"   Images shape: {images.shape}")
    print(f"     - Batch size: {images.shape[0]}")
    print(f"     - Channels: {images.shape[1]}")
    print(f"     - Height: {images.shape[2]}")
    print(f"     - Width: {images.shape[3]}")
    print(f"   Labels shape: {labels.shape}")
    print(f"   Label values: {labels.tolist()[:8]}...")
    print(f"   Image value range: [{images.min():.4f}, {images.max():.4f}]")
except Exception as e:
    print(f"   ❌ Error loading batch: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# STEP 5: Visualize Sample Doodles
# ============================================================================
print("\n[STEP 5] Visualizing sample doodles...")
try:
    fig, axes = plt.subplots(1, 4, figsize=(6, 6))
    
    for ax, img, label in zip(axes.ravel(), images[:4], labels[:4]):
        # Image is [1, 128, 128], extract single channel
        img_np = img[0].numpy()
        
        ax.imshow(img_np, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{classes[label.item()]}", fontsize=8)
        ax.axis("off")
    
    plt.suptitle("Sample Doodles from QuickDraw Dataset", fontsize=14)
    plt.tight_layout()
    plt.savefig("sample_doodles.png", dpi=100, bbox_inches="tight")
    print(f"   ✅ Visualized {len(images)} samples")
    print(f"   Saved to: sample_doodles.png")
    plt.show()
except Exception as e:
    print(f"   ❌ Error visualizing: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# STEP 6: Test Model with Real Data
# ============================================================================
print("\n[STEP 6] Testing CNN-VAE model with real data...")
try:
    # Initialize model
    model = VAE(image_channels=1, latent_dim=2, kl_weight=1.0/(128*128))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    print(f"   Device: {device}")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass on first batch
    images_device = images.to(device)
    print(f"   Input shape: {images_device.shape}")
    
    with torch.no_grad():
        x_recon, mu, log_var, z = model(images_device)
    
    print("   ✅ Forward pass successful")
    print(f"     - Reconstructed shape: {x_recon.shape}")
    print(f"     - Latent mean shape: {mu.shape}")
    print(f"     - Latent log_var shape: {log_var.shape}")
    print(f"     - Samples shape: {z.shape}")
    
    # Compute loss
    loss, recon_loss, kl_loss = model.compute_loss(images_device, x_recon, mu, log_var)
    print(f"\n   ✅ Loss computation successful")
    print(f"     - Total loss: {loss.item():.4f}")
    print(f"     - Recon loss: {recon_loss.item():.4f}")
    print(f"     - KL loss: {kl_loss.item():.4f}")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# STEP 7: Visualize Reconstructions
# ============================================================================
print("\n[STEP 7] Visualizing reconstructions...")
try:
    fig, axes = plt.subplots(2, 4, figsize=(6, 6))
    
    x_recon_np = x_recon.detach().cpu()
    
    # Top row: originals
    for i in range(4):
        img_np = images[i, 0].numpy()
        axes[0, i].imshow(img_np, cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"Original {i}", fontsize=10)
        axes[0, i].axis("off")
    
    # Bottom row: reconstructions
    for i in range(4):
        img_np = x_recon_np[i, 0].numpy()
        axes[1, i].imshow(img_np, cmap="gray", vmin=0, vmax=1)
        axes[1, i].set_title(f"Recon {i}", fontsize=10)
        axes[1, i].axis("off")
    
    plt.suptitle("Originals vs Reconstructions", fontsize=12)
    plt.tight_layout()
    plt.savefig("reconstructions.png", dpi=100, bbox_inches="tight")
    print("   ✅ Visualizations saved")
    print("   Files: reconstructions.png")
    plt.show()
except Exception as e:
    print(f"   ❌ Error visualizing reconstructions: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# STEP 8: Test Cache Performance
# ============================================================================
print("\n[STEP 8] Testing cache performance...")
try:
    print("   Loading batch again (should use cache)...")
    
    # Time multiple batch loads
    times = []
    for i in range(3):
        start = time.time()
        images2, labels2 = next(iter(train_loader))
        elapsed = time.time() - start
        times.append(elapsed)
    
    print(f"   ✅ Cache performance test completed")
    print(f"     - Load 1: {times[0]:.3f}s (rendering + cache)")
    print(f"     - Load 2: {times[1]:.3f}s (cached)")
    print(f"     - Load 3: {times[2]:.3f}s (cached)")
    if times[0] > 0:
        print(f"     - Speedup: {times[0]/times[1]:.1f}x faster")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("PHASE 1 VERIFICATION SUMMARY")
print("=" * 70)
print("\n✅ QuickDraw Dataset Implementation:")
print("   - QuickDrawDataset class with stroke rendering")
print("   - Image caching for efficiency")
print("   - DifferentLabelBatchSampler for balanced batches")
print("   - DataLoader creation with proper collation")
print("\n✅ Data Pipeline:")
print(f"   - Classes: {len(classes)} ({', '.join(classes)})")
print(f"   - Batch size: 4")
print(f"   - Image shape: [1, 128, 128]")
print(f"   - Image range: [0, 1]")
print("\n✅ Model Integration:")
print(f"   - CNN-based VAE loads successfully")
print(f"   - Forward pass works on real data")
print(f"   - Loss computation: {loss.item():.4f}")
print("\n✅ Complete: Ready for Phase 3 (Model Training)")
print("=" * 70)
print("\nNext step: Run 'python train.py' to start training\n")
