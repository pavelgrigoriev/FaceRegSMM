from torchvision.transforms import v2
import torch

def get_transforms(imgsz=640):
    train_transform = v2.Compose([
        v2.ToImage(),
        v2.RandomApply([
            v2.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05
            )
        ], p=0.8),
        v2.RandomGrayscale(p=0.02),
        v2.RandomResizedCrop(
            size=(640, 640),
            scale=(0.7, 1.0),
            ratio=(3/4, 4/3),
            antialias=True
        ),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.1),
        v2.RandomRotation(degrees=15), # type: ignore
        v2.RandomPerspective(distortion_scale=0.2, p=0.3),
        v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomErasing(
            p=0.2,
            scale=(0.02, 0.2),
            ratio=(0.3, 3.3),
            value="random" # type: ignore
        ),
        v2.Resize((imgsz, imgsz), antialias=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])

    base_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((imgsz, imgsz), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, base_transform