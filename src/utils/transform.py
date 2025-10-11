import torch
from torchvision.transforms import v2

def get_transforms(img_size=640):

    train_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((img_size, img_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),

        v2.RandomHorizontalFlip(p=0.5),

        v2.RandomApply([
            v2.RandomRotation(degrees=10), # type: ignore
        ], p=0.5),

        v2.RandomApply([
            v2.RandomAffine(
                degrees=0, # type: ignore
                translate=(0.05, 0.05),
                scale=(0.9, 1.1),
                shear=(-5, 5, -5, 5),
            ),
        ], p=0.4),

        v2.RandomPerspective(distortion_scale=0.2, p=0.3),

        v2.RandomApply([
            v2.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.05
            )
        ], p=0.7),

        v2.RandomApply([
            v2.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))
        ], p=0.3),

        v2.RandomApply([
            v2.RandomAdjustSharpness(sharpness_factor=2)
        ], p=0.3),

        v2.RandomApply([
            v2.RandomEqualize()
        ], p=0.2),

        v2.RandomApply([
            v2.Lambda(lambda x: (x + 0.02 * torch.randn_like(x)).clamp(0, 1))
        ], p=0.3),

        v2.RandomApply([
            v2.RandAugment(num_ops=2, magnitude=3)
        ], p=0.5),

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