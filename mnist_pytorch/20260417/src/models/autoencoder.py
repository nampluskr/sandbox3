import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.image import StructuralSimilarityIndexMeasure


class AutoEncoder:
    def __init__(self, encoder, decoder, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)

        parameters = list(encoder.parameters()) + list(decoder.parameters())
        self.optimizer = optim.Adam(parameters, lr=1e-3)
        self.loss_fn = nn.BCELoss()     # or nn.MSELoss()
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device) # (B, C, H, W)

    def forward(self, images):
        latent = self.encoder(images)
        recon = self.decoder(latent)
        return recon, latent

    def train_step(self, batch):
        self.encoder.train()
        self.decoder.train()
        images = batch["image"].to(self.device)
        recon, latent = self.forward(images)
        loss = self.loss_fn(recon, images)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        ssim = self.ssim_metric(recon, images)
        return {
            "loss": loss.item(), 
            "ssim": ssim.item(), 
            "batch_size": images.size(0)
        }

    @torch.no_grad()
    def eval_step(self, batch):
        self.encoder.eval()
        self.decoder.eval()
        images = batch["image"].to(self.device)
        recon, latent = self.forward(images)
        loss = self.loss_fn(recon, images)
        ssim = self.ssim_metric(recon, images)
        return {
            "loss": loss.item(), 
            "ssim": ssim.item(), 
            "batch_size": images.size(0)
        }
