import torch
import torch.nn.functional as F
from mamba_ssm import Mamba2
from torch import nn


class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size=16, in_channels=3, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)  # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N_patches, D)
        x = self.norm(x)
        return x


class AttentionPooling(nn.Module):

    def __init__(self, embed_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, x):
        # x: (B, N, D)
        weights = F.softmax(self.attn(x), dim=1)  # (B, N, 1)
        pooled = (weights * x).sum(dim=1)  # (B, D)
        return pooled


class RecSSM(nn.Module):
    def __init__(
        self,
        img_size=128,
        patch_size=16,
        embed_dim=128,
        embedding_size=128,
        num_layers=4,
        dropout=0.1,
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)

        self.ssm_blocks = nn.ModuleList(
            [
                Mamba2(d_model=embed_dim, d_state=64, d_conv=4, expand=2, headdim=32)
                for _ in range(num_layers)
            ]
        )

        self.norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers)])

        self.final_norm = nn.LayerNorm(embed_dim)
        self.pool = AttentionPooling(embed_dim)

        # если embed_dim != embedding_size, добавляем линейный слой
        self.proj = (
            nn.Linear(embed_dim, embedding_size)
            if embed_dim != embedding_size
            else nn.Identity()
        )

    def forward(self, x):
        x = self.patch_embed(x)  # (B, N, D)

        for ssm, norm, drop in zip(self.ssm_blocks, self.norms, self.dropouts):
            residual = x
            x = norm(x)
            x = ssm(x)  # Mamba2 ожидает (B, L, D)
            x = drop(x)
            x = x + residual  # residual connection

        x = self.final_norm(x)
        x = self.pool(x)  # attention pooling вместо mean

        embeddings = self.proj(x)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings
