# dataset.py

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from torch.utils.data import Dataset

from src.dataset.utils import load_image


class PersonDataset(Dataset):
    def __init__(
        self,
        root_dir,
        transform,
        img_size,
        person_to_id,
        image_paths,
    ):
        super().__init__()
        self.root = Path(root_dir)
        self.transform = transform
        self.img_size = img_size
        self.person_to_id = person_to_id

        self.paths = image_paths

    def __getitem__(self, index):
        img_path = self.paths[index]
        person_name = img_path.parent.name
        label = self.person_to_id.get(person_name, -1)

        image = load_image(img_path, self.img_size, self.transform)
        return image, label

    def __len__(self):
        return len(self.paths)
