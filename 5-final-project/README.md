# VAE Doodle Generator & Latent Space Visualizer

**A PyTorch-based Variational Autoencoder for generating and visualizing QuickDraw doodles**

## 🎨 Project Overview

This project builds a convolutional VAE trained on Google QuickDraw dataset. Users can explore a 2D latent space interactively and generate custom doodles using Streamlit.

### Architecture
- **Encoder**: Conv2D with stride=2 downsampling (128×128 → ... → 2D latent)
- **Decoder**: Conv2DTranspose with stride=2 upsampling (2D latent → ... → 128×128)
- **Loss**: MSE reconstruction + KL divergence (following course MNIST VAE pattern)
- **Training**: PyTorch with Adam optimizer, gradient clipping, early stopping

### Features
✅ Fast image rendering with PIL ImageDraw
✅ Disk caching for rendered doodles
✅ Balanced batch sampling across classes
✅ Interactive Streamlit web interface
✅ Latent space interpolation
✅ 2D visualization of doodle generation

## 📂 Project Structure

```
5-final-project/
├── plan.md                      # Detailed implementation plan (6 phases, 23 steps)
├── progress.md                  # Phase-by-phase tracking
├── PHASE_1_README.md           # Phase 1 data pipeline documentation
├── requirements.txt             # Python dependencies
│
├── model.py                     # VAE architecture (Encoder, Decoder, VAE)
├── data.py                      # QuickDrawDataset and DataLoader setup
├── train.py                     # Training script with VAETrainer class
├── app.py                       # Streamlit web interface
├── utils.py                     # Helper functions
│
├── 01-data-exploration.py       # Interactive Phase 1 verification notebook
└── checkpoints/                 # Saved model checkpoints (auto-created)
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- QuickDraw dataset: `C:\datasets\google-quickdraw\sketches\`
- ~5 GB free disk space (for image cache)

### Installation

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 1. Verify Data Pipeline (Phase 1)

```bash
python 01-data-exploration.py
```

This will:
- Load QuickDraw dataset
- Render and cache ~5k doodles
- Visualize samples
- Test model forward pass
- Verify loss computation

**Expected time**: 5-10 minutes first run, <1 minute cached

### 2. Train the Model (Phase 3)

```bash
python train.py
```

This will:
- Load full QuickDraw dataset (auto-caches)
- Train for 50 epochs (early stopping at 10 epochs no improvement)
- Save best model checkpoint
- Plot training curves

**Expected time**: 2-4 hours (GPU), ~24 hours (CPU)
**GPU**: Recommended (Tesla V100: ~2 hours)

### 3. Launch Streamlit App (Phase 5)

```bash
streamlit run app.py
```

Open browser to `http://localhost:8501`

Features:
- **Generate tab**: Sliders to explore latent coordinates
- **Explore tab**: 2D scatter plot of latent space
- **Interpolate tab**: Smooth transitions between doodles

## 📊 Implementation Status

| Phase | Step | Status | Component |
|-------|------|--------|-----------|
| 1 | 1-4 | ✅ Complete | Data pipeline, QuickDrawDataset, DataLoaders |
| 2 | 5-8 | ✅ Complete | Convolutional VAE (Encoder/Decoder) |
| 3 | 9-12 | ✅ Complete | Training infrastructure (VAETrainer) |
| 4 | 13-15 | 🔲 Ready | Training runs + validation |
| 5 | 16-19 | ✅ Scaffold | Streamlit interface |
| 6 | 20-23 | 🔲 Ready | Documentation + deployment |

## 📝 Configuration

Key hyperparameters in `train.py`:
- `batch_size`: 32 (balanced sampling across labels)
- `num_epochs`: 50 (with early stopping)
- `latent_dim`: 2 (for visualization)
- `kl_weight`: 1.0 / (128 × 128) (scales KL by image size)
- `learning_rate`: 1e-3 (Adam optimizer)

Modify in `main()` function before running.

## 🔬 Model Architecture Details

### Encoder
```
Input: [batch, 1, 128, 128]
  ↓
Conv2D(32, 3×3, stride=2) + ReLU → [batch, 32, 64, 64]
  ↓
Conv2D(64, 3×3, stride=2) + ReLU → [batch, 64, 32, 32]
  ↓
Conv2D(128, 3×3, stride=2) + ReLU → [batch, 128, 16, 16]
  ↓
Flatten → [batch, 32768]
  ↓
Dense(256) + ReLU → [batch, 256]
  ↓
Dense(latent_dim) → μ: [batch, 2], log_σ²: [batch, 2]
```

### Decoder
```
Input: [batch, 2] (latent)
  ↓
Dense(256) + ReLU → [batch, 256]
  ↓
Dense(32768) + ReLU → [batch, 32768]
  ↓
Reshape → [batch, 128, 16, 16]
  ↓
ConvTranspose2d(128, 64, 3×3, stride=2) + ReLU → [batch, 64, 32, 32]
  ↓
ConvTranspose2d(64, 32, 3×3, stride=2) + ReLU → [batch, 32, 64, 64]
  ↓
ConvTranspose2d(32, 1, 3×3, stride=2) + Sigmoid → [batch, 1, 128, 128] ∈ [0,1]
```

### Loss Function
```
Total Loss = Reconstruction Loss + (1/16384) × KL Divergence

Reconstruction Loss = E[MSE(x, x̂)]
KL Divergence = -0.5 × E[1 + log_σ² - μ² - exp(log_σ²)]
```

## 📚 Course References

Adaptive from *IBM + Coursera - Machine Learning Capstone*:

1. **Data pipeline**: [final-project.ipynb](../5-deep-learning-reinforcement-learning/final-project/final-project.ipynb)
   - CachedQuickDrawDataset implementation
   - DifferentLabelBatchSampler concept
   - PIL stroke rendering

2. **VAE architecture**: [27-variational-autoencoders.ipynb](../5-deep-learning-reinforcement-learning/27-variational-autoencoders.ipynb)
   - MNIST VAE theory
   - Conv2D encoder + Dense bottleneck
   - ConvTranspose2d decoder with upsampling
   - MSE + KL loss formulation

## 🔧 Files Reference

### model.py
- `Encoder`: CNN encoder to latent space
- `Decoder`: CNN decoder from latent space
- `VAE`: Complete model with loss computation

### data.py
- `QuickDrawDataset`: PyTorch Dataset with caching
- `DifferentLabelBatchSampler`: Balanced batch creation
- `get_data_loaders()`: Factory for train/val/test loaders
- `create_dummy_loader()`: For testing without real data

### train.py
- `set_seed()`: Reproducibility
- `VAETrainer`: Custom training loop, checkpointing, early stopping
- `main()`: Entry point with real data loading

### app.py
- Streamlit interface with model loading
- Generate, Explore, Interpolate tabs
- Interactive latent space exploration

### utils.py
- Image visualization, PCA/t-SNE reduction
- Caching, grid creation
- Helper functions for model analysis

## 🐛 Troubleshooting

### "CUDA out of memory"
Reduce `batch_size` in train.py or use CPU: `device='cpu'`

### "QuickDraw data not found"
Ensure dataset at: `C:\datasets\google-quickdraw\sketches\`

### First batch is slow
Expected behavior: PIL rendering takes 2-5s. Cache is saved for future use.

### Training loss is high
- Normal: KL term scales by 1/16384, so expect high reconstruction values
- Check: Ensure images are [-1, 1] or [0, 1] normalized

## 📖 Documentation

- **[plan.md](plan.md)** - Full 6-phase implementation plan with 23 steps
- **[progress.md](progress.md)** - Real-time phase tracking and notes
- **[PHASE_1_README.md](PHASE_1_README.md)** - Data pipeline deep-dive
- **Code docstrings** - Comprehensive per-function documentation

## 🎯 Next Steps

1. **Verify installation**: `python 01-data-exploration.py`
2. **Train model**: `python train.py` (2-4 hours GPU)
3. **Launch app**: `streamlit run app.py`
4. **Explore**: Use sliders and interpolation to generate doodles

## 💡 Tips

- **First run**: Full dataset caching takes 5-30 minutes. Plan accordingly.
- **Development**: Use subset of classes/samples for faster iteration (see Phase 1 README)
- **GPU**: Strongly recommended for training. CPU training ~10-12x slower.
- **Reproducibility**: Set `seed=42` in all functions for deterministic runs

## 📄 License

Course project - Educational use only

## 🔗 Resources

- [VAE Explained](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f40)
- [PyTorch Docs](https://pytorch.org/docs/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [QuickDraw Dataset](https://github.com/googlecreativelab/quickdraw-dataset)

---

**Status**: Phase 1 ✅ Complete | Phases 3-6 🔲 Ready to start
**Last Updated**: 2026-04-13
**Questions?** See PHASE_1_README.md or check docstrings in code
