from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import random 
class TripletDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        extension_list = ["PNG", "png", "jpeg", "JPEG", "jpg", "JPG"]
        self.root = Path(root_dir)
        self.images = [f for ext in extension_list for f in self.root.glob(f"*.{ext}")]
        self.transform = transform
        
    def __getitem__(self, index):
        anchor_image = Image.open(self.images[index]).convert("RGB").resize((640, 640))
        positive_image = anchor_image.copy()
        if self.transform:
            anchor_image = self.transform(anchor_image)
            positive_image = self.transform(positive_image)
        neg_index = random.randint(0, len(self.images)-1)
        while neg_index == index:
            neg_index = random.randint(0, len(self.images)-1)
        negative_image = Image.open(self.images[neg_index]).convert("RGB").resize((640, 640))
        if self.transform:
            negative_image = self.transform(negative_image)

        return anchor_image, positive_image, negative_image
    
    def __len__(self):
        return len(self.images)