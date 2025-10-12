import torch
from torchvision.transforms import v2

def get_transforms(img_size=640):
    train_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((img_size, img_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        
        v2.RandomHorizontalFlip(p=0.5),
        
        v2.RandomApply([
            v2.RandomRotation(degrees=15),  # type: ignore
        ], p=0.5),
        
        v2.RandomApply([
            v2.RandomAffine(
                degrees=0, # type: ignore
                translate=(0.1, 0.1), 
                scale=(0.85, 1.15),    
                shear=(-8, 8, -8, 8),  
            ),
        ], p=0.5),
        
        v2.RandomPerspective(distortion_scale=0.25, p=0.4), 
        
        v2.RandomResizedCrop(
            size=(img_size, img_size),
            scale=(0.8, 1.0), 
            ratio=(0.95, 1.05),
            antialias=True
        ),
        
        v2.RandomApply([
            v2.ColorJitter(
                brightness=0.4,  
                contrast=0.4,     
                saturation=0.3,
                hue=0.03         
            )
        ], p=0.8),  
        
        v2.RandomApply([
            v2.Lambda(lambda x: torch.clamp(x * torch.FloatTensor(1).uniform_(0.7, 1.3), 0, 1))
        ], p=0.3),
        
        v2.RandomApply([
            v2.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 2.0))
        ], p=0.25),
        
        v2.RandomApply([
            v2.RandomAdjustSharpness(sharpness_factor=2.0)
        ], p=0.3),
        
        v2.RandomApply([
            v2.RandomAutocontrast()
        ], p=0.3),
        
        v2.RandomApply([
            v2.Lambda(lambda x: torch.pow(x, torch.FloatTensor(1).uniform_(0.8, 1.2)))
        ], p=0.3),
        
        v2.RandomApply([
            v2.Lambda(lambda x: torch.clamp(x + 0.015 * torch.randn_like(x), 0, 1))
        ], p=0.25),
        
        v2.RandomErasing(
            p=0.3,
            scale=(0.02, 0.15),  
            ratio=(0.3, 3.3),
            value=0,
        ),
        
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