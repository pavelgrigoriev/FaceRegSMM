import random
from pathlib import Path

from torch.utils.data import Dataset

from src.dataset.utils import load_image


class TripletDataset(Dataset):
    def __init__(self, root_dir, transform=None, img_size=128):
        self.img_size = img_size
        self.root = Path(root_dir)
        self.transform = transform
        self.random = random.Random(123)
        self.person_dirs = [p for p in self.root.iterdir() if p.is_dir()]

        self.images_by_person = {p: list(p.glob("*")) for p in self.person_dirs}

        self.valid_persons = [
            p for p, imgs in self.images_by_person.items() if len(imgs) >= 2
        ]

    def __getitem__(self, index):
        anchor_person = self.random.choice(self.valid_persons)
        imgs = self.images_by_person[anchor_person]

        a_path, p_path = self.random.sample(imgs, 2)

        negative_person = self.random.choice(
            [p for p in self.person_dirs if p != anchor_person]
        )
        n_path = self.random.choice(self.images_by_person[negative_person])

        anchor_image = load_image(a_path, self.img_size, self.transform)
        positive_image = load_image(p_path, self.img_size, self.transform)
        negative_image = load_image(n_path, self.img_size, self.transform)

        return anchor_image, positive_image, negative_image

    def __len__(self):
        return len(self.valid_persons) * 30
