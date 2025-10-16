from pathlib import Path

from torch.utils.data import Dataset

from src.constants import EXTENSION_LIST
from src.dataset.utils import load_image


class PersonDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root = Path(root_dir)
        self.transform = transform
        self.paths = sorted(
            [p for p in self.root.rglob("*") if p.suffix[1:] in EXTENSION_LIST]
        )
        self.person_dirs = sorted([p for p in self.root.iterdir() if p.is_dir()])
        self.person_to_id = {
            person_dir.name: i for i, person_dir in enumerate(self.person_dirs)
        }

    def __getitem__(self, index):
        img_path = self.paths[index]
        person_name = img_path.parent.name
        label = self.person_to_id[person_name]

        image = load_image(img_path, self.transform)
        return image, label

    def __len__(self):
        return len(self.paths)
