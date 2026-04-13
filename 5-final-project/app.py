"""
Streamlit web interface for VAE Doodle Generator
Users can explore latent space and generate doodles interactively
"""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

from model import VAE


@st.cache_resource
def load_model(model_path, device='cpu'):
    """Load trained VAE model (cached)"""
    try:
        model = VAE(image_channels=1, latent_dim=2)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def tensor_to_image(tensor, image_size=128):
    """Convert [1, 128, 128] tensor to PIL Image"""
    # Handle both [1, 128, 128] and [batch, 1, 128, 128] inputs
    if tensor.dim() == 4:
        tensor = tensor[0]  # Take first from batch
    
    # Ensure [1, H, W] or [H, W]
    if tensor.dim() == 3 and tensor.shape[0] == 1:
        img_array = tensor[0].detach().cpu().numpy()
    else:
        img_array = tensor.detach().cpu().numpy()
    
    # Convert to [0, 255]
    img_array = (img_array * 255).astype(np.uint8)
    
    # Convert to PIL Image (L mode for grayscale)
    img = Image.fromarray(img_array, mode='L')
    
    return img


def decode_latent(model, z, device='cpu'):
    """
    Decode latent vector to image
    
    Args:
        model: VAE model
        z: [latent_dim] or [1, latent_dim] tensor
        device: device to use
        
    Returns:
        image: PIL Image [128, 128]
    """
    if z.dim() == 1:
        z = z.unsqueeze(0)  # Add batch dimension: [1, 2]
    
    z = z.to(device)
    
    with torch.no_grad():
        x_recon = model.decode(z)  # [1, 1, 128, 128]
    
    image = tensor_to_image(x_recon)
    return image


def plot_latent_space(model, num_samples=100, device='cpu'):
    """
    Plot 2D latent space with example samples
    
    TODO: Encode training data if available and plot
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot grid of latent points
    x_range = np.linspace(-3, 3, num_samples // 10)
    y_range = np.linspace(-3, 3, num_samples // 10)
    
    X, Y = np.meshgrid(x_range, y_range)
    ax.scatter(X, Y, alpha=0.3, s=10, c='gray', label='Latent Grid')
    
    ax.set_xlabel('Latent Dim 1')
    ax.set_ylabel('Latent Dim 2')
    ax.set_title('2D Latent Space')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    return fig


def interpolate_latent(model, z1, z2, num_steps=10, device='cpu'):
    """
    Interpolate between two latent points
    
    Args:
        model: VAE model
        z1, z2: [latent_dim] tensors
        num_steps: number of interpolation steps
        
    Returns:
        images: list of PIL Images
    """
    images = []
    
    for t in np.linspace(0, 1, num_steps):
        z_interp = (1 - t) * z1 + t * z2
        img = decode_latent(model, z_interp, device)
        images.append(img)
    
    return images


def main():
    st.set_page_config(page_title="VAE Doodle Generator", layout="wide")
    
    st.title("🎨 VAE Doodle Generator")
    st.write("""
    Explore a 2D latent space and generate doodles interactively.
    Click on the plot or use sliders to select coordinates.
    """)
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = "./checkpoints/vae_best_epoch0.pt"  # TODO: Update path
    
    # Load model
    with st.spinner("Loading model..."):
        model = load_model(model_path, device)
    
    if model is None:
        st.error("❌ Could not load model. Train the model first using `python train.py`")
        st.info(f"Expected model at: {model_path}")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["Generate", "Explore Latent Space", "Interpolate"])
    
    # Tab 1: Generate
    with tab1:
        st.header("Generate Doodles")
        
        col1, col2 = st.columns(2)
        
        with col1:
            z1_val = st.slider("Latent Dim 1", -3.0, 3.0, 0.0, step=0.1, key="z1_main")
            z2_val = st.slider("Latent Dim 2", -3.0, 3.0, 0.0, step=0.1, key="z2_main")
        
        # Decode and display
        z = torch.tensor([z1_val, z2_val], dtype=torch.float32)
        img = decode_latent(model, z, device)
        
        with col2:
            st.image(img, caption="Generated Doodle", width=250)
        
        # Download button (placeholder)
        st.download_button(
            label="Download PNG",
            data=None,  # TODO: Implement
            file_name="doodle.png",
            mime="image/png"
        )
    
    # Tab 2: Explore Latent Space
    with tab2:
        st.header("Latent Space Visualization")
        
        fig = plot_latent_space(model, num_samples=100, device=device)
        st.pyplot(fig)
        
        st.write("TODO: Interactive click-to-select on latent space plot")
    
    # Tab 3: Interpolate
    with tab3:
        st.header("Interpolate Between Points")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Point 1")
            z1_1 = st.slider("Dim 1", -3.0, 3.0, 0.0, key="z1_p1")
            z1_2 = st.slider("Dim 2", -3.0, 3.0, 0.0, key="z2_p1")
        
        with col2:
            st.subheader("Point 2")
            z2_1 = st.slider("Dim 1", -3.0, 3.0, 1.0, key="z1_p2")
            z2_2 = st.slider("Dim 2", -3.0, 3.0, 1.0, key="z2_p2")
        
        num_steps = st.slider("Interpolation Steps", 3, 20, 10)
        
        if st.button("Generate Interpolation"):
            z1 = torch.tensor([z1_1, z1_2], dtype=torch.float32)
            z2 = torch.tensor([z2_1, z2_2], dtype=torch.float32)
            
            images = interpolate_latent(model, z1, z2, num_steps, device)
            
            # Display as grid
            cols = st.columns(5)
            for i, img in enumerate(images):
                with cols[i % 5]:
                    st.image(img, caption=f"Step {i+1}")
    
    # Footer
    st.divider()
    st.caption("VAE Doodle Generator - Streamlit Interface")


if __name__ == '__main__':
    main()
