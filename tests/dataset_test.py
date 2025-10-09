import os, sys

project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_dir)

import numpy as np
from src.dataset.dataset import TripletDataset

from utils import create_images
from torch.utils.data import DataLoader
import tempfile
from src.utils.transform import transform

def test_dataset_loader():
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Temporary directory created at: {temp_dir}")
        create_images(temp_dir)
        batch_size = np.random.randint(1,16)
        dataset = TripletDataset(temp_dir, transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        iterator = iter(dataloader)
        a,p,n = next(iterator)

    assert len(dataset) == 100, "Неверая длина датасета"
    assert a.shape == (batch_size, 3, 640, 640), "Неверная форма anchor_image"
    assert p.shape == (batch_size, 3, 640, 640), "Неверная форма positive_image"
    assert n.shape == (batch_size, 3, 640, 640), "Неверная форма negative_image"