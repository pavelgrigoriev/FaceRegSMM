import os
import sys
import tempfile

import numpy as np
from torch.utils.data import DataLoader

project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_dir)

from utils import create_images
from src.dataset.dataset import TripletDataset
from src.utils.transform import get_transforms

def test_dataset_loader():
    train_transform, base_stransform = get_transforms()
    with tempfile.TemporaryDirectory() as temp_dir:
        create_images(temp_dir)
        batch_size = np.random.randint(1,16)
        dataset = TripletDataset(temp_dir, train_transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        iterator = iter(dataloader)
        a,p,n = next(iterator)

    assert len(dataset) == 100, "Неверая длина датасета"
    assert a.shape == (batch_size, 3, 640, 640), "Неверная форма anchor_image с train_transform"
    assert p.shape == (batch_size, 3, 640, 640), "Неверная форма positive_image с train_transform"
    assert n.shape == (batch_size, 3, 640, 640), "Неверная форма negative_image с train_transform"

    with tempfile.TemporaryDirectory() as temp_dir:
        create_images(temp_dir, 20)
        batch_size = np.random.randint(1,16)
        dataset = TripletDataset(temp_dir, base_stransform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        iterator = iter(dataloader)
        a,p,n = next(iterator)

    assert len(dataset) == 20, "Неверая длина датасета"
    assert a.shape == (batch_size, 3, 640, 640), "Неверная форма anchor_image с base_stransform"
    assert p.shape == (batch_size, 3, 640, 640), "Неверная форма positive_image с base_stransform"
    assert n.shape == (batch_size, 3, 640, 640), "Неверная форма negative_image с base_stransform"