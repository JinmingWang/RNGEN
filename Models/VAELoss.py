import torch
import torch.nn as nn
import torch.nn.functional as F

class KLLoss(nn.Module):
    def __init__(self, kl_weight: float = 1.0):
        """
        Initializes the VAELoss class.

        Parameters:
        - kl_weight (float): Weight to scale the KL divergence loss. Default is 1.0.
        """
        super(KLLoss, self).__init__()
        self.kl_weight = kl_weight

    def forward(self, z_mean: torch.Tensor, z_logvar: torch.Tensor):
        """
        Computes the total VAE loss which includes the KL divergence loss and the reconstruction loss (MSE).

        Parameters:
        - reconstructed (Tensor): The reconstructed input from the decoder (B, N, d_in).
        - original (Tensor): The original input to the encoder (B, N, d_in).
        - z_mean (Tensor): Mean of the latent distribution (B, N, latent_dim).
        - z_logvar (Tensor): Log variance of the latent distribution (B, N, latent_dim).

        Returns:
        - total_loss (Tensor): The total loss, which is a sum of KL divergence and MSE loss.
        - mse_loss (Tensor): The MSE loss (reconstruction loss).
        - kl_loss (Tensor): The KL divergence loss.
        """

        # Compute KL Divergence loss (KL divergence between N(z_mean, exp(z_logvar)) and N(0, 1))
        kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp(), dim=-1)  # Per node
        kl_loss = kl_loss.mean() * self.kl_weight

        return kl_loss
