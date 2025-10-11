import torch
from torchvision.transforms import v2

def get_transforms(img_size=640):
    train_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((img_size, img_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        
    
        v2.RandomHorizontalFlip(p=0.5), 
        v2.RandomApply([
            v2.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,  
                hue=0.02         
            )
        ], p=0.5),
        v2.RandomRotation(degrees=5),   # type: ignore
        
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])

    base_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((img_size, img_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, base_transform