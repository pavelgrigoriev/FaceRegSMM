import os, sys
project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_dir)

import numpy as np
from src.dataset.dataset import TripletDataset
from pathlib import Path
from PIL import Image
import random 
from torch.utils.data import DataLoader
import tempfile
from src.utils.transform import transform
import random

def test_dataset_loader():
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Temporary directory created at: {temp_dir}")
        extension_list = ["PNG", "png", "jpeg", "JPEG", "jpg","JPG"]
        folder = Path(temp_dir)
        for i in range(100):
            array = np.random.randint(0, 256, (np.random.randint(128,2880), np.random.randint(128,2880), 3), dtype=np.uint8)
            image = Image.fromarray(array)
            image.save(f"{folder}/temp_img_{i}.{random.choice(extension_list)}")
        extension_list = ["a", "..", "  ", "12", "aa",","]
        for i in range(20):
            with open(f"{folder}/temp_trash_{i}.{random.choice(extension_list)}", 'w') as f:
                pass 
        batch_size = np.random.randint(1,16)
        dataset = TripletDataset(temp_dir, transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        iterator = iter(dataloader)
        a,p,n = next(iterator)

    assert len(dataset) == 100, "Неверая длина датасета"
    assert a.shape == (batch_size, 3, 640, 640), "Неверная форма anchor_image"
    assert p.shape == (batch_size, 3, 640, 640), "Неверная форма positive_image"
    assert n.shape == (batch_size, 3, 640, 640), "Неверная форма negative_image"