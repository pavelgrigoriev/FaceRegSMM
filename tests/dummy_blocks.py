from torch import nn


class DummySSMBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, 3, padding=1)
        self.norm = nn.BatchNorm1d(d_model)
        self.act = nn.SiLU()

    def forward(self, x):
        x_transposed = x.transpose(1, 2)
        x_conv = self.act(self.norm(self.conv(x_transposed)))
        return x_conv.transpose(1, 2)


def get_dummy_ssm_blocks(d_model=128, num_blocks=4):
    return nn.ModuleList([DummySSMBlock(d_model) for _ in range(num_blocks)])
