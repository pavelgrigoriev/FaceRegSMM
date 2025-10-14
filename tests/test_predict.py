import os
import sys

import numpy as np
import torch.nn.functional as F

project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_dir)

from dummy_blocks import DummySSMBlocks

from src.models.model import RecSSM
from src.models.predict import predict


def test_predict():
    device = "cpu"
    model = RecSSM(640)
    model.ssmblocks = DummySSMBlocks()  # type: ignore

    first_img = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
    second_img = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)

    embedding1 = predict(first_img, model, 640, device)
    embedding2 = predict(second_img, model, 640, device)
    similarity = F.cosine_similarity(embedding1, embedding2)
    if similarity > 0.5:
        print("its ones men")
    else:
        print("different men")
    assert embedding1.shape == embedding2.shape, "Не совпадают эмбеддинги изображений"
