import random
from pathlib import Path

from torch.utils.data import Dataset
from src.dataset.utils import load_image
from src.utils.transform import get_transforms

class TripletDataset(Dataset):
    def __init__(self, root_dir, transform=None, imgsz=640):
        self.imgsz = imgsz
        self.root = Path(root_dir)
        self.transform = transform

        self.person_dirs = [p for p in self.root.iterdir() if p.is_dir()]

        self.images_by_person = {p: list(p.glob("*")) for p in self.person_dirs}

        self.valid_persons = [p for p, imgs in self.images_by_person.items() if len(imgs) >= 2]

        self.transform, _ = get_transforms(self.imgsz)
        
    def __getitem__(self, index):
        anchor_person = random.choice(self.valid_persons)
        imgs = self.images_by_person[anchor_person]

        a_path, p_path = random.sample(imgs, 2)

        negative_person = random.choice([p for p in self.person_dirs if p != anchor_person])
        n_path = random.choice(self.images_by_person[negative_person])


        anchor_image = load_image(a_path, self.imgsz, self.transform)
        positive_image = load_image(p_path, self.imgsz, self.transform)
        negative_image = load_image(n_path, self.imgsz, self.transform)

        return anchor_image, positive_image, negative_image
    
    def __len__(self):
        return len(self.valid_persons) * 10