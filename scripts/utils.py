import logging
import os

import hydra
import torch
import torchvision.utils as vutils
from numpy import sqrt
from PIL import Image

from src.models.model import RecSSM

log = logging.getLogger(__name__)


def save_samples(imgs):
    try:
        save_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir  # type: ignore
        os.makedirs(save_dir, exist_ok=True)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = imgs * std + mean
        img = torch.clamp(img, 0, 1)

        grid = vutils.make_grid(img, nrow=int(sqrt(imgs.size(0))))
        grid_img = (grid.permute(1, 2, 0).numpy() * 255).astype("uint8")
        img_pil = Image.fromarray(grid_img)
        img_pil.save(os.path.join(save_dir, "imgs_batch.jpg"))
    except Exception as e:
        log.error(f"Error saving samples: {e}")


def load_model(model_path, img_size, device):
    model_state_dict = torch.load(model_path, map_location=device)["model_state_dict"]
    path_size = torch.load(model_path, map_location=device)["pathch_size"]
    model = RecSSM(img_size, path_size).to(device)
    model.load_state_dict(model_state_dict)
    return model
