from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 16, latent_dim: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def reconstruction_loss(input_tensor: torch.Tensor, output_tensor: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.pow(input_tensor - output_tensor, 2), dim=1)


def compute_loss_for_batch(model: Autoencoder, batch: torch.Tensor, device: torch.device) -> torch.Tensor:
    model.to(device)
    batch = batch.to(device)
    recon = model(batch)
    return reconstruction_loss(batch, recon)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
