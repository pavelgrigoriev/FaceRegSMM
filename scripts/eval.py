import logging
import os
import sys

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_dir)

from src.dataset.dataset import PersonDataset
from src.models.evaluate import evaluate
from src.models.model import RecSSM
from src.utils.transform import get_transforms

log = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
if device != "cuda":
    raise SystemError("Sorry, ssm only support with gpu/cuda")


@hydra.main(
    version_base=None,
    config_path=os.path.join(project_dir, "configs"),
    config_name="eval",
)
def main(cfg: DictConfig):
    data_path = cfg["data_path"]
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"data_path not found: {data_path}")
    model_path = cfg["model_path"]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    img_size = cfg["img_size"]

    batch_size = cfg["batch_size"]
    num_workers = cfg["num_workers"]

    log.info(f"data_path: {data_path}")
    log.info(f"model_path: {model_path}")
    log.info(f"img_size: {img_size}")
    log.info(f"batch_size: {batch_size}")
    log.info(f"num_workers: {num_workers}")

    model = RecSSM(img_size).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device)["model_state_dict"]
    )
    _, base_transform = get_transforms(img_size)
    eval_dataset = PersonDataset(data_path, base_transform, img_size=img_size)

    log.info(f"Len eval_dataset: {len(eval_dataset)}")

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    evaluate(model, eval_dataloader, device)


if __name__ == "__main__":
    main()
