# src/models/gan.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.models.blocks import ConvBlock, DeconvBlock
from src.models.weights import init_weights


class Generator(nn.Module):
    def __init__(self, img_size=32, latent_dim=100, out_channels=3, base=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size

        num_blocks = {32: 2, 64: 3, 128: 4, 256: 5}
        if img_size not in num_blocks:
            raise ValueError(f"Unsupported img_size: {img_size}")

        in_channels = base * (2 ** num_blocks[img_size])
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, in_channels,
                kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
        )

        blocks = []
        for i in range(num_blocks[img_size]):
            blocks.append(DeconvBlock(in_channels, in_channels // 2))
            in_channels //= 2

        self.blocks = nn.Sequential(*blocks)
        self.final = nn.ConvTranspose2d(base, out_channels,
            kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, z):
        z = z.view(-1, self.latent_dim, 1, 1)
        x = self.initial(z)
        x = self.blocks(x)
        x = self.final(x)
        return torch.tanh(x)


class Discriminator(nn.Module):
    def __init__(self, img_size=32, in_channels=3, base=64):
        super().__init__()
        self.img_size = img_size

        num_blocks = {32:  2, 64:  3, 128: 4, 256: 5}
        if img_size not in num_blocks:
            raise ValueError(f"Unsupported img_size: {img_size}")

        out_channels = base
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        blocks = []
        for i in range(num_blocks[img_size]):
            blocks.append(ConvBlock(out_channels, out_channels * 2))
            out_channels *= 2

        self.blocks = nn.Sequential(*blocks)
        self.final = nn.Conv2d(out_channels, 1, kernel_size=4, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.initial(x)
        x = self.blocks(x)
        x = self.final(x)
        return x.view(-1, 1)


class VanillaGAN(nn.Module):
    def __init__(self, generator, discriminator, latent_dim=None, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)

        self.generator.apply(init_weights)
        self.discriminator.apply(init_weights)

        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

        self.latent_dim = latent_dim or self.generator.latent_dim

    def d_loss_fn(self, real_preds, fake_preds):
        real_labels = torch.ones_like(real_preds)
        fake_labels = torch.zeros_like(fake_preds)
        d_real_loss = F.binary_cross_entropy_with_logits(real_preds, real_labels)
        d_fake_loss = F.binary_cross_entropy_with_logits(fake_preds, fake_labels)
        d_loss = d_real_loss + d_fake_loss
        return d_loss, d_real_loss, d_fake_loss

    def g_loss_fn(self, fake_preds):
        real_labels = torch.ones_like(fake_preds)
        g_loss = F.binary_cross_entropy_with_logits(fake_preds, real_labels)
        return g_loss

    def train_step(self, batch):
        self.train()
        real_images = batch['image'].to(self.device)
        batch_size = real_images.size(0)

        # Discriminator training
        real_preds = self.discriminator(real_images)
        noises = torch.randn(batch_size, self.latent_dim).to(self.device)
        fake_images = self.generator(noises).detach()
        fake_preds = self.discriminator(fake_images)

        d_loss, d_real_loss, d_fake_loss = self.d_loss_fn(real_preds, fake_preds)

        self.d_optimizer.zero_grad()
        d_loss.backward()
        self.d_optimizer.step()

        # Generator training
        noises = torch.randn(batch_size, self.latent_dim).to(self.device)
        fake_images = self.generator(noises)
        fake_preds = self.discriminator(fake_images)
        g_loss = self.g_loss_fn(fake_preds)

        self.g_optimizer.zero_grad()
        g_loss.backward()
        self.g_optimizer.step()

        return {
            "g_loss": g_loss.item(),
            "d_loss": d_loss.item(),
            "real_loss": d_real_loss.item(),
            "fake_loss": d_fake_loss.item(),
            "batch_size": batch_size
        }

    @torch.no_grad()
    def predict(self, noises):
        self.eval()
        noises = noises.to(self.device)
        images = self.generator(noises)
        preds = self.discriminator(images)

        images = (images + 1) / 2.0
        images = images.permute(0, 2, 3, 1).cpu().numpy()
        preds = torch.sigmoid(preds).cpu().numpy()
        return images.squeeze(-1), preds.squeeze(-1)
