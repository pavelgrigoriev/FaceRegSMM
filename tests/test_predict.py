import os, sys



project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_dir)

import numpy as np
import torch.nn.functional as F
from src.models.model import RecSSM
from dummy_blocks import DummySSMBlocks
from PIL import Image
from src.models.predict import predict

def test_predic():
    device = "cpu"
    model = RecSSM()
    model.ssmblocks = DummySSMBlocks() # type: ignore

    first_img  = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
    second_img = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)

    embedding1 = predict(first_img, model, 640, device)
    embedding2 = predict(second_img, model, 640, device)
    similarity = F.cosine_similarity(embedding1, embedding2)
    if similarity > 0.5:
        print("its ones men")
    else:
        print("different men")
    assert embedding1.shape == embedding2.shape, "Не совпадают эмбеддинги изображений"

