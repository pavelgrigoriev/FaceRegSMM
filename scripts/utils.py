

import os
import hydra
import torch
from PIL import Image

def image_from_dataloader(img):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img * std + mean  

    img = img.permute(1, 2, 0).numpy()
    img = (img * 255).clip(0, 255).astype('uint8')
    return img

def save_samples(a,p,n):
    a_img = Image.fromarray(image_from_dataloader(a[0,:,:,:]))
    a_img.save(os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, "anchor_sample.jpg")) # type: ignore
    p_img = Image.fromarray(image_from_dataloader(p[0,:,:,:]))
    p_img.save(os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, "positive_sample.jpg")) # type: ignore
    n_img = Image.fromarray(image_from_dataloader(n[0,:,:,:]))
    n_img.save(os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, "negative_sample.jpg"))  # type: ignore