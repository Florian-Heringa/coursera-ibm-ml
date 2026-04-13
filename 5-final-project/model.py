"""
Variational Autoencoder (VAE) for QuickDraw Doodle Generation
Uses convolutional architecture (inspired by MNIST VAE example from course)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Encoder network: maps images to latent space parameters (μ, log_σ²)
    
    Uses Conv2D layers with stride=2 for progressive dimensionality reduction.
    Architecture (for 128×128 input):
    - Conv2D(32, 3×3, stride=2): [1, 128, 128] → [32, 64, 64]
    - Conv2D(64, 3×3, stride=2): [32, 64, 64] → [64, 32, 32]
    - Conv2D(128, 3×3, stride=2): [64, 32, 32] → [128, 16, 16]
    - Flatten → [128*16*16] = [32768]
    - Dense(256) → [256]
    - Output heads: μ [latent_dim], log_σ² [latent_dim]
    """
    
    def __init__(self, image_channels=1, latent_dim=2):
        super(Encoder, self).__init__()
        
        self.image_channels = image_channels
        self.latent_dim = latent_dim
        
        # Conv2D layers with stride=2 for downsampling
        # 128×128 → 64×64
        self.conv1 = nn.Conv2d(image_channels, 32, kernel_size=3, stride=2, padding=1)
        # 64×64 → 32×32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        # 32×32 → 16×16
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        
        # After flattening: 128 * 16 * 16 = 32768
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        
        # Latent space heads
        self.mu_head = nn.Linear(256, latent_dim)
        self.logvar_head = nn.Linear(256, latent_dim)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, 1, 128, 128] image tensor in [0, 1]
            
        Returns:
            mu: [batch_size, latent_dim] mean of latent distribution
            log_var: [batch_size, latent_dim] log variance of latent distribution
        """
        # Conv layers with ReLU activation
        h = F.relu(self.conv1(x))  # [batch, 32, 64, 64]
        h = F.relu(self.conv2(h))  # [batch, 64, 32, 32]
        h = F.relu(self.conv3(h))  # [batch, 128, 16, 16]
        
        # Flatten
        h = h.view(h.size(0), -1)  # [batch, 32768]
        h = F.relu(self.fc1(h))  # [batch, 256]
        
        # Output heads
        mu = self.mu_head(h)
        log_var = self.logvar_head(h)
        
        return mu, log_var


class Decoder(nn.Module):
    """
    Decoder network: maps latent vectors back to image space
    
    Uses Conv2DTranspose layers for progressive dimensionality increase (upsampling).
    Architecture (for 128×128 output):
    - Dense(256) → [256]
    - Dense(128*16*16) → [32768]
    - Reshape → [128, 16, 16]
    - Conv2DTranspose(128, 3×3, stride=2): [128, 16, 16] → [64, 32, 32]
    - Conv2DTranspose(64, 3×3, stride=2): [64, 32, 32] → [32, 64, 64]
    - Conv2DTranspose(32, 3×3, stride=2): [32, 64, 64] → [1, 128, 128]
    - Sigmoid activation → [0, 1] range
    """
    
    def __init__(self, latent_dim=2, image_channels=1):
        super(Decoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.image_channels = image_channels
        
        # Fully connected layer to recover spatial dimension 
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 128 * 16 * 16)  # 32768 = 128 * 16 * 16
        
        # Conv2DTranspose layers with stride=2 for upsampling
        # 16×16 → 32×32
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        # 32×32 → 64×64
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        # 64×64 → 128×128
        self.deconv3 = nn.ConvTranspose2d(32, image_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        
    def forward(self, z):
        """
        Args:
            z: [batch_size, latent_dim] sampled latent vector
            
        Returns:
            x_recon: [batch_size, 1, 128, 128] reconstructed image in [0, 1]
        """
        # Expand from latent
        h = F.relu(self.fc1(z))  # [batch, 256]
        h = F.relu(self.fc2(h))  # [batch, 32768]
        
        # Reshape to spatial dims
        h = h.view(h.size(0), 128, 16, 16)  # [batch, 128, 16, 16]
        
        # Upsample with ConvTranspose
        h = F.relu(self.deconv1(h))  # [batch, 64, 32, 32]
        h = F.relu(self.deconv2(h))  # [batch, 32, 64, 64]
        x_recon = torch.sigmoid(self.deconv3(h))  # [batch, 1, 128, 128] in [0, 1]
        
        return x_recon


class VAE(nn.Module):
    """
    Variational Autoencoder: combines encoder and decoder with reparameterization trick
    
    Loss = Reconstruction Loss (MSE) + β × KL Divergence
    This follows the MNIST VAE pattern from the course with convolutional architecture.
    """
    
    def __init__(self, image_channels=1, latent_dim=2, kl_weight=1.0/(128*128)):
        super(VAE, self).__init__()
        
        self.image_channels = image_channels
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight  # Scale KL by image size for balance
        
        self.encoder = Encoder(image_channels, latent_dim)
        self.decoder = Decoder(latent_dim, image_channels)
        
    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick: z = μ + ε ⊙ σ where ε ~ N(0, 1)
        
        Args:
            mu: [batch_size, latent_dim] mean
            log_var: [batch_size, latent_dim] log variance
            
        Returns:
            z: [batch_size, latent_dim] sampled latent vector
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def forward(self, x):
        """
        Full forward pass through VAE
        
        Args:
            x: [batch_size, 1, 128, 128] image tensor in [0, 1]
            
        Returns:
            x_recon: [batch_size, 1, 128, 128] reconstructed image
            mu: [batch_size, latent_dim] latent mean
            log_var: [batch_size, latent_dim] latent log variance
            z: [batch_size, latent_dim] sampled latent vector
        """
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)
        
        return x_recon, mu, log_var, z
    
    def compute_loss(self, x, x_recon, mu, log_var):
        """
        Compute VAE loss: reconstruction loss + KL divergence
        Uses MSE for reconstruction (appropriate for continuous image values in [0, 1])
        
        Args:
            x: [batch_size, 1, 128, 128] original image
            x_recon: [batch_size, 1, 128, 128] reconstructed image
            mu: [batch_size, latent_dim] latent mean
            log_var: [batch_size, latent_dim] latent log variance
            
        Returns:
            total_loss: scalar loss
            recon_loss: reconstruction loss (MSE)
            kl_loss: KL divergence
        """
        # Reconstruction loss: Mean Squared Error
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        
        # KL divergence: -0.5 * mean(1 + log_var - mu^2 - exp(log_var))
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        
        # Total loss (KL is scaled by image size for balance)
        total_loss = recon_loss + self.kl_weight * kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    def encode(self, x):
        """
        Encode image to latent space (use mean, no sampling)
        
        Args:
            x: [batch_size, 1, 128, 128] image tensor
            
        Returns:
            mu: [batch_size, latent_dim] latent mean
        """
        mu, _ = self.encoder(x)
        return mu
    
    def decode(self, z):
        """
        Decode latent vector to image space
        
        Args:
            z: [batch_size, latent_dim] latent vector
            
        Returns:
            x_recon: [batch_size, 1, 128, 128] reconstructed image
        """
        return self.decoder(z)
