import torch.nn.functional as F
from torch import nn


class DummySSMBlock(nn.Module):
    """
    A dummy SSM block for testing purposes.
    It simulates the behavior of a real SSM block without complex computations.
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(self.linear(x)))


class DummySSMBlocks(nn.ModuleList):
    def __init__(self, num_blocks=4, embed_dim=128):
        super().__init__([DummySSMBlock(embed_dim) for _ in range(num_blocks)])


class PatchEmbedding(nn.Module):
    """
    Image to Patch Embedding using Conv2d.
    Args:
        img_size (int): Size of the input image (assumed square).
        patch_size (int): Size of each patch (assumed square).
        in_channels (int): Number of input channels (e.g., 3 for RGB images).
        embed_dim (int): Dimension of the embedding.
    """

    def __init__(self, img_size, patch_size=16, in_channels=3, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x


class RecSSM(nn.Module):
    """
    Recurrent SSM-based model for image embedding.
    Args:
        img_size (int): Size of the input image (assumed square).
        patch_size (int): Size of each patch (assumed square).
        embed_dim (int): Dimension of the embedding.
    """

    def __init__(self, img_size=128, patch_size=16, embed_dim=128):
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)

        self.ssm_blocks = nn.ModuleList([DummySSMBlock(embed_dim) for _ in range(6)])

        self.norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(6)])

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

        self.norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(6)])

        self.final_norm = nn.LayerNorm(embed_dim)
        s
