import logging
from pathlib import Path

import hydra
import torch
import torchvision.utils as vutils
from numpy import sqrt
from PIL import Image

from src.models.model import RecSSM

log = logging.getLogger(__name__)


def save_samples(imgs) -> None:
    """Save a batch of sample images to the output directory.
    Args:
        imgs (torch.Tensor): Batch of images to be saved.
    Returns:
        None
    """
    try:
        save_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)  # type: ignore
        save_dir.mkdir(parents=True, exist_ok=True)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = imgs * std + mean
        img = torch.clamp(img, 0, 1)

        grid = vutils.make_grid(img, nrow=int(sqrt(imgs.size(0))))
        grid_img = (grid.permute(1, 2, 0).numpy() * 255).astype("uint8")
        img_pil = Image.fromarray(grid_img)
        img_pil.save((save_dir / "imgs_batch.jpg"))
    except Exception as e:
        raise RuntimeError(f"Error in saving sample images: {e}")


def load_model(model_path, img_size, device) -> RecSSM:
    """
    Load a trained model from a specified path.
    Args:
        model_path (Path): Path to the model file.
        img_size (int): Size of the input images.
        device (str): Device to load the model onto ('cpu' or 'cuda'). At on 16.10.2025 only 'cuda' is supported.
    Returns:
        RecSSM: Loaded model."""
    try:
        model_state_dict = torch.load(model_path, map_location=device)[
            "model_state_dict"
        ]
        path_size = torch.load(model_path, map_location=device)["patch_size"]
        model = RecSSM(img_size, path_size).to(device)
        model.load_state_dict(model_state_dict)
    except Exception as e:
        raise RuntimeError(f"Error in loading the model: {e}")
    return model
