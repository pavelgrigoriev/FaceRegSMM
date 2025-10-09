from torch import nn

class DummySSMBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(self.conv((x))))
    
class DummySSMBlocks(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssmblocks = nn.Sequential(
            DummySSMBlock(24, 12),
            DummySSMBlock(12, 6),
            DummySSMBlock(6, 8),
        )

    def forward(self, x):
        return self.ssmblocks((x))