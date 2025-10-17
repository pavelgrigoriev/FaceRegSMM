import logging
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

project_dir = Path(__file__).resolve().parents[1]
sys.path.append(project_dir.as_posix())

from scripts.utils import load_model
from src.dataset.dataset import PersonDataset
from src.models.evaluate import evaluate
from src.utils.transform import get_transforms

log = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
if device != "cuda":
    raise SystemError("Sorry, ssm only support with gpu/cuda")


@hydra.main(
    version_base=None,
    config_path=str(project_dir / "configs"),
    config_name="eval",
)
def main(cfg: DictConfig) -> None:

    data_path = cfg.get("data_path")
    if not data_path:
        raise ValueError("data_path must be specified in the config file.")

    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"data_path not found: {data_path}")

    model_path = cfg.get("model_path")
    if not model_path:
        raise ValueError("model_path must be specified in the config file.")
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    img_size = cfg.get("img_size")

    batch_size = cfg.get("batch_size")
    num_workers = cfg.get("num_workers")

    log.info(f"data_path: {data_path}")
    log.info(f"model_path: {model_path}")
    log.info(f"img_size: {img_size}")
    log.info(f"batch_size: {batch_size}")
    log.info(f"num_workers: {num_workers}")

    model = load_model(model_path, img_size, device)
    _, base_transform = get_transforms(img_size)
    eval_dataset = PersonDataset(data_path, base_transform)

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
