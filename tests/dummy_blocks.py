from torch import nn


class DummySSMBlock(nn.Module):
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
