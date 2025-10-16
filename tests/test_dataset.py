import os
import sys
import tempfile

import numpy as np
from torch.utils.data import DataLoader

from utils import create_images

project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_dir)

from src.dataset.dataset import PersonDataset
from src.utils.transform import get_transforms


def test_dataset():
    train_transform, _ = get_transforms(128)
    with tempfile.TemporaryDirectory() as temp_dir:
        create_images(temp_dir)
        batch_size = np.random.randint(1, 16)
        dataset = PersonDataset(temp_dir, train_transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        iterator = iter(dataloader)
        imgs, _ = next(iterator)

    assert len(dataset) == 100, "Неверая длина датасета"
    assert imgs.shape == (
        batch_size,
        3,
        128,
        128,
    ), "Неверная форма imgs в train_transform"
