import os
import sys

import torch

project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_dir)

from dummy_blocks import DummySSMBlocks

from src.models.model import RecSSM


def test_model():
    model = RecSSM(640)
    model.ssm_blocks = DummySSMBlocks()  # type: ignore this need cause for ssm need only cuda
    sample_1 = torch.rand((1, 3, 640, 640))
    output_1 = model(sample_1)
    sample_2 = torch.rand((1, 3, 320, 320))
    output_2 = model(sample_2)
    sample_3 = torch.rand((1, 3, 128, 128))
    output_3 = model(sample_3)
    sample_4 = torch.rand((1, 3, 1280, 1280))
    output_4 = model(sample_4)

    assert output_1.shape == (1, 128), "Неверая длина эмбеддинга"
    assert output_2.shape == (1, 128), "Неверая длина эмбеддинга"
    assert output_3.shape == (1, 128), "Неверая длина эмбеддинга"
    assert output_4.shape == (1, 128), "Неверая длина эмбеддинга"
