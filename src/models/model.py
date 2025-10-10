from torch import nn 
from mamba_ssm import Mamba2
import torch.nn.functional as F

# img size 3 640 640
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, (3,3), padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
        self.pool = nn.MaxPool2d(4)

    def forward(self, x):
        return self.pool(self.act(self.norm(self.conv(x))))
    
class SSMBlock(nn.Module):
    def __init__(self, in_channels, out_channels, d_model, headdim):
        super().__init__()

        self.ssm = Mamba2(
            d_model=d_model, # Model dimension d_model
            d_state=64,  # SSM state expansion factor, typically 64 or 128
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
            headdim=headdim

        )
        self.conv = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.SiLU()
        
    def forward(self, x):
        return self.act(self.norm(self.conv(self.ssm(x))))
        

class RecSSM(nn.Module):
    def __init__(self):
        super().__init__()
        self.convblocks = nn.Sequential(
            ConvBlock(3, 6),
            ConvBlock(6, 12),
            ConvBlock(12, 24)
        )
        self.ssmblocks = nn.Sequential(
            SSMBlock(24, 12, 4, 2),
            SSMBlock(12, 6, 4, 2),
            SSMBlock(6, 8, 4, 2),
        )
        #TODO need rewrite
        self.avg = nn.AdaptiveMaxPool1d(16)

    def forward(self,x):
        x = self.convblocks(x)
        # batch 3*8 640/8 640/8
        x = x.view(x.shape[0], x.shape[1], -1)
        x = self.ssmblocks(x)
        x = self.avg(x)
        
        return  F.normalize(x.view(x.shape[0], -1), p=2, dim=1)
        # b 128

