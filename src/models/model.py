from torch import nn 
from mamba_ssm import Mamba2
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size=16, in_channels=3, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        
        self.proj = nn.Conv2d(in_channels, embed_dim, 
                             kernel_size=patch_size, 
                             stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x

class RecSSM(nn.Module):
    def __init__(self, img_size=128, patch_size=16, embed_dim=128):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        
        self.ssm_blocks = nn.ModuleList([
            Mamba2(
                d_model=embed_dim,
                d_state=64,
                d_conv=4,
                expand=2,
                headdim=32
            ) for _ in range(4)
        ])
        
        self.norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(4)
        ])
        
        self.final_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, 128)

    def forward(self, x):
        x = self.patch_embed(x)
        
        for ssm, norm in zip(self.ssm_blocks, self.norms):
            x = x + ssm(norm(x))
        
        x = self.final_norm(x)
        
        x = x.mean(dim=1) 
        
        x = self.head(x)
        return F.normalize(x, p=2, dim=1)