import logging
import os

import hydra
import torch
import torchvision.utils as vutils
from PIL import Image

log = logging.getLogger(__name__)


def image_from_dataloader(img):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img * std + mean

    img = img.permute(1, 2, 0).numpy()
    img = (img * 255).clip(0, 255).astype("uint8")
    return img


def save_samples(a, p, n):

    try:
        save_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir  # type: ignore
        os.makedirs(save_dir, exist_ok=True)

        for name, batch in zip(["anchor", "positive", "negative"], [a, p, n]):
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            batch = batch * std + mean
            batch = torch.clamp(batch, 0, 1)

            grid = vutils.make_grid(batch, nrow=4)
            grid_img = (grid.permute(1, 2, 0).numpy() * 255).astype("uint8")
            img = Image.fromarray(grid_img)
            img.save(os.path.join(save_dir, f"{name}_batch.jpg"))
    except Exception as e:
        log.error(f"Error saving samples: {e}")
