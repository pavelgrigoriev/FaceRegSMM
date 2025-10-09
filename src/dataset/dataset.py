import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import random

from src.utils.transform import get_transforms
class TripletDataset(Dataset):
    def __init__(self, root_dir, transform=None, imgsz=640):
        extension_list = ["PNG", "png", "jpeg", "JPEG", "jpg", "JPG"]
        self.imgsz = imgsz
        self.root = Path(root_dir)
        self.images = [f for ext in extension_list for f in self.root.glob(f"*.{ext}")]
        self.transform = transform
        
    def __getitem__(self, index):
        anchor_image = Image.open(self.images[index]).convert("RGB").resize((self.imgsz, self.imgsz))
        positive_image = anchor_image.copy()
        if self.transform:
            anchor_image = self.transform(anchor_image)
        pos_trans, _ = get_transforms()
        positive_image = pos_trans(positive_image)
        neg_index = random.randint(0, len(self.images)-1)
        while neg_index == index:
            neg_index = random.randint(0, len(self.images)-1)
        negative_image = Image.open(self.images[neg_index]).convert("RGB").resize((self.imgsz, self.imgsz))
        if self.transform:
            negative_image = self.transform(negative_image)

        return anchor_image, positive_image, negative_image
    
    def __len__(self):
        return len(self.images)