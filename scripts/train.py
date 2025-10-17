import logging
import sys
from pathlib import Path

import hydra
import pytorch_warmup as warmup
import torch
from omegaconf import DictConfig
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.samplers import MPerClassSampler
from torch.utils.data import DataLoader

project_dir = Path(__file__).resolve().parents[1]
sys.path.append(project_dir.as_posix())

from scripts.utils import load_model, save_samples
from src.dataset.dataset import PersonDataset
from src.models.evaluate import evaluate
from src.models.model import RecSSM
from src.models.trainer import train
from src.utils.transform import get_transforms

log = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"

if device != "cuda":
    raise SystemError("Sorry, ssm only support with gpu/cuda")

log.info(f"Using device: {device}")


@hydra.main(
    version_base=None,
    config_path=str(project_dir / "configs"),
    config_name="train",
)
def main(cfg: DictConfig) -> None:
    data_path = cfg.get("data_path")
    if not data_path:
        raise ValueError("data_path must be specified in the config file.")
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"data_path not found: {data_path}")

    epochs = cfg.get("epochs")
    batch_size = cfg.get("batch_size")
    img_size = cfg.get("img_size")
    lr = cfg.get("lr")
    num_workers = cfg.get("num_workers")
    warmup_period = cfg.get("warmup_period")
    eta_min = cfg.get("eta_min")
    margin = cfg.get("margin")
    patch_size = cfg.get("patch_size")
    model_path = cfg.get("model_path")
    if model_path:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"model_path not found: {model_path}")
    log.info(f"data_path: {data_path}")
    log.info(f"epochs: {epochs}")
    log.info(f"batch_size: {batch_size}")
    log.info(f"img_size: {img_size}")
    log.info(f"lr: {lr}")
    log.info(f"warmup_period: {warmup_period}")
    log.info(f"num_workers: {num_workers}")
    log.info(f"eta_min: {eta_min}")
    log.info(f"margin: {margin}")
    log.info(f"patch_size: {patch_size}")
    train_transform, base_transform = get_transforms(img_size)

    train_dataset = PersonDataset((data_path / "train"), train_transform)
    val_dataset = PersonDataset((data_path / "val"), base_transform)
    test_dataset = PersonDataset((data_path / "test"), base_transform)

    log.info(f"Len train_dataset: {len(train_dataset)}")
    log.info(f"Len test_dataset: {len(test_dataset)}")
    log.info(f"Len val_dataset: {len(val_dataset)}")

    labels = [train_dataset.person_to_id[p.parent.name] for p in train_dataset.paths]
    sampler = MPerClassSampler(labels, m=2, length_before_new_iter=len(labels))

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    if model_path:
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model = load_model(model_path, img_size, device)
        log.info(f"Loaded model from {model_path}")
    else:
        model = RecSSM(img_size, patch_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, epochs - warmup_period, eta_min=1e-6
    )
    if warmup_period > 0:
        warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period)
    else:
        warmup_scheduler = None
    loss_fn = losses.TripletMarginLoss(margin=margin)
    miner = miners.TripletMarginMiner(margin=margin, type_of_triplets="all")
    train_iter = iter(train_dataloader)
    imgs, _ = next(train_iter)
    save_samples(imgs)

    model = train(
        epochs,
        model,
        train_dataloader,
        val_dataloader,
        loss_fn,
        miner,
        optimizer,
        scheduler,
        device,
        warmup_scheduler,
        warmup_period,
    )

    evaluate(model, test_dataloader, device)


if __name__ == "__main__":
    main()
