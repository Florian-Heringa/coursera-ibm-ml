## Doodle Generator and Latent Space Visualizer

**TL;DR**: Build a PyTorch VAE trained on Google QuickDraw, then create an interactive Streamlit app where users can explore a 2D latent space and generate custom doodles by selecting points.

---

## Implementation Phases & Steps

### **Phase 1: Data Preparation & Setup** (Steps 1-4)

1. **Acquire Google QuickDraw dataset**
   - ~~Download or reference QuickDraw `.npz` files (stroke data) for selected classes~~
   - ~~Store in project data directory: `./data/quickdraw/`~~
   - Dataset is available (as .npz files) in `C:\datasets\google-quickdraw\sketches`
   - Document: which classes included, data size, format

2. **Create custom PyTorch Dataset class** (`QuickDrawDataset`)
   - Inherit from `torch.utils.data.Dataset`
   - Load `.npz` stroke files
   - Render stroke data → 128×128 grayscale images
   - Cache rendered images to disk for speed
   - Implement `__getitem__` and `__len__` methods
   - Normalize images to [0, 1] range (divide by 255)

3. **Set up data pipeline**
   - Create train/validation/test split (70/15/15)
   - Implement balanced batch samplers
   - Initialize PyTorch `DataLoader` with batch_size=32, shuffle=True for train
   - Verify loader returns correct tensor shapes: `[batch_size, 1, 128, 128]`

4. **Define model input/output specifications**
   - Input: image tensors `[batch_size, 1, 128, 128]` (convolutional, not flattened)
   - Latent space dimension: 2 (for interactive visualization)
   - Output: reconstructed image tensors `[batch_size, 1, 128, 128]` with sigmoid activation [0, 1]
   - Loss: Mean Squared Error (reconstruction) + KL divergence

### **Phase 2: Model Architecture** (Steps 5-8)

5. **Design VAE encoder** (`Encoder` class) - Uses Conv2D for dimensionality reduction
   - Input: `[batch_size, 1, 128, 128]` image
   - Conv2D (32 filters, 3×3 kernel, stride=2, padding="same"): 128×128 → 64×64
   - Conv2D (64 filters, 3×3 kernel, stride=2, padding="same"): 64×64 → 32×32
   - Conv2D (128 filters, 3×3 kernel, stride=2, padding="same"): 32×32 → 16×16
   - Flatten: [batch_size, 128×16×16] = [batch_size, 32768]
   - Dense (256 units, ReLU): fully connected bottleneck
   - Output heads: 
     - μ: [batch_size, 2]
     - log_σ²: [batch_size, 2]
   - Reparameterization: z = μ + ε ⊙ σ where ε ~ N(0,1)

6. **Design VAE decoder** (`Decoder` class) - Uses Conv2DTranspose for upsampling
   - Input: latent vector z [batch_size, 2]
   - Dense (256 units, ReLU): expand representations
   - Dense (32768 units, ReLU): expand to [batch_size, 32768]
   - Reshape: [batch_size, 128, 16, 16]
   - Conv2DTranspose (128 filters, 3×3, stride=2, padding="same"): 16×16 → 32×32
   - Conv2DTranspose (64 filters, 3×3, stride=2, padding="same"): 32×32 → 64×64
   - Conv2DTranspose (32 filters, 3×3, stride=2, padding="same"): 64×64 → 128×128
   - Conv2DTranspose (1 filter, 3×3, stride=1, sigmoid): [batch_size, 1, 128, 128]
   - Output: reconstructed image in [0, 1]

7. **Assemble full VAE model** (`VAE` class)
   - Compose encoder + decoder
   - Forward pass: encode → sample z → decode
   - Loss: MSE(input, output) + weighted KL divergence
   - Reconstruction loss: mean((input - output)²)
   - KL divergence: -0.5 × mean(1 + log_σ² - μ² - exp(log_σ²))
   - Total loss: recon_loss + (1/(128×128)) × kl_loss  (scales KL by image size)

8. **Initialize model parameters**
   - Set random seed for reproducibility
   - Xavier/He weight initialization
   - Move to device (GPU if available)

### **Phase 3: Training Infrastructure** (Steps 9-12)

9. **Set up training loop basics**
   - Optimizer: Adam (lr=1e-3)
   - Gradient clipping: `torch.nn.utils.clip_grad_norm_` (max_norm=1.0)
   - Device handling and batch processing

10. **Implement training function**
    - Loop over epochs
    - Compute MSE + KL loss per batch
    - Backward pass + optimizer step
    - Track reconstruction loss and KL loss separately

11. **Implement validation function**
    - Evaluate on validation set (no updates)
    - Compute validation loss
    - Early stopping logic

12. **Add monitoring & checkpointing**
    - Log losses per epoch
    - Save best model checkpoint
    - Plot training curves

### **Phase 4: Train the VAE Model** (Steps 13-15)

13. **Execute training** (50-100 epochs)
    - Monitor convergence
    - Adjust β (KL weight) if needed
    - Save best checkpoint

14. **Qualitative validation**
    - Sample reconstructed images
    - Verify doodle-like appearance
    - Check for mode collapse/artifacts

15. **Latent space interpolation test**
    - Sample two latent points
    - Interpolate between them
    - Verify smooth transitions

### **Phase 5: Interactive Latent Space Visualizer** (Steps 16-19)

16. **Design latent space visualization**
    - Use 2D latent directly or reduce higher dims with PCA/t-SNE
    - Encode all training samples → 2D coordinates

17. **Create Streamlit app** (`app.py`)
    - Load trained VAE on startup
    - Display 2D scatter plot of latent space
    - Add interactive point selection

18. **Implement latent → image decoder**
    - User selects latent coordinate
    - Decode through VAE
    - Display reconstructed doodle

19. **Add user controls**
    - Sliders for coordinate adjustment
    - "Generate random" button
    - Interpolation feature
    - Class filtering
    - Download PNG option

### **Phase 6: Deployment & Polish** (Steps 20-23)

20. **Package application**
    - Organize code: `model.py`, `data.py`, `train.py`, `app.py`, `utils.py`
    - Create `requirements.txt`

21. **Add documentation**
    - `README.md`: setup instructions, usage guide
    - Docstrings in code

22. **Test end-to-end**
    - Training completes successfully
    - Model loads in Streamlit
    - Latent interaction works
    - Image generation quality verified

23. **Optional cloud deployment**
    - Streamlit Cloud / Hugging Face Spaces
    - Document deployment steps

---

## Key Reference Files

- [5-deep-learning-reinforcement-learning/final-project/final-project.ipynb](../../5-deep-learning-reinforcement-learning/final-project/final-project.ipynb) — PyTorch VAE + QuickDraw patterns
- [5-deep-learning-reinforcement-learning/27-variational-autoencoders.ipynb](../../5-deep-learning-reinforcement-learning/27-variational-autoencoders.ipynb) — VAE theory

---

## Verification Checklist

- ✓ Dataset loads and caches correctly
- ✓ Model forward pass shapes correct
- ✓ Training reduces loss; KL ≠ 0
- ✓ Reconstructions resemble originals
- ✓ Streamlit app interactive and functional
- ✓ Generated doodles recognizable

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| PyTorch | Reference implementation; flexible sampling |
| Fully-connected VAE | Simple, fast, sufficient for 128×128 |
| Streamlit | Rapid interactive UI; built-in widgets |
| 2D latent | Directly visualizable |
| Batch size 32 | Balance of speed/stability |