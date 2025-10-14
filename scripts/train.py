import logging
import os
import sys

import hydra
import torch
from omegaconf import DictConfig
from pytorch_metric_learning import losses
from pytorch_metric_learning.samplers import MPerClassSampler
from torch.utils.data import DataLoader

project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_dir)

from src.dataset.dataset import PersonDataset
from src.dataset.utils import create_train_val_test_splits
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
    config_path=os.path.join(project_dir, "configs"),
    config_name="train",
)
def main(cfg: DictConfig):
    data_path = cfg["data_path"]
    if data_path == "":
        raise ValueError("Please check your data path")

    epochs = cfg["epochs"]
    batch_size = cfg["batch_size"]
    img_size = cfg["img_size"]
    lr = cfg["lr"]

    min_images_for_split = cfg["min_images_for_split"]
    embedding_size = cfg["embedding_size"]
    val_ratio = cfg["val_ratio"]
    test_ratio = cfg["test_ratio"]
    seed = cfg["seed"]

    arcface_margin = cfg["arcface_margin"]
    arcface_scale = cfg["arcface_scale"]

    log.info(f"data_path: {data_path}")
    log.info(f"epochs: {epochs}")
    log.info(f"batch_size: {batch_size}")
    log.info(f"img_size: {img_size}")
    log.info(f"lr: {lr}")
    log.info(f"embedding_size: {embedding_size}")
    log.info(f"seed: {seed}")

    log.info("Creating train/val/test splits...")

    train_paths, val_paths, test_paths, person_to_id = create_train_val_test_splits(
        root_dir=data_path,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        min_images_for_split=min_images_for_split,
        seed=seed,
    )

    num_classes = len(person_to_id)

    log.info(f"\nDataset statistics:")
    log.info(f"Number of persons: {num_classes}")
    log.info(f"Train images: {len(train_paths)}")
    log.info(f"Val images: {len(val_paths)}")
    log.info(f"Test images: {len(test_paths)}")
    log.info(f"Total images: {len(train_paths) + len(val_paths) + len(test_paths)}")

    train_transform, base_transform = get_transforms(img_size)

    train_labels = [person_to_id[p.parent.name] for p in train_paths]
    m = 4
    train_sampler = MPerClassSampler(train_labels, m=m, batch_size=batch_size)

    train_dataset = PersonDataset(
        root_dir=data_path,
        transform=train_transform,
        img_size=img_size,
        person_to_id=person_to_id,
        image_paths=train_paths,
    )

    val_dataset = PersonDataset(
        root_dir=data_path,
        transform=base_transform,
        img_size=img_size,
        person_to_id=person_to_id,
        image_paths=val_paths,
    )

    test_dataset = PersonDataset(
        root_dir=data_path,
        transform=base_transform,
        img_size=img_size,
        person_to_id=person_to_id,
        image_paths=test_paths,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    log.info("Initializing model...")

    model = RecSSM(img_size, patch_size=16, embedding_size=embedding_size).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Total parameters: {total_params:,}")
    log.info(f"Trainable parameters: {trainable_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=1e-4, betas=(0.9, 0.999)
    )

    log.info("Setting up loss...")

    loss_fn = losses.ArcFaceLoss(
        num_classes=num_classes,
        embedding_size=embedding_size,
        margin=arcface_margin,
        scale=arcface_scale,
    ).to(device)

    loss_optimizer = torch.optim.SGD(loss_fn.parameters(), lr=0.1, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )

    log.info(f"Loss function: {loss_fn.__class__.__name__}")
    log.info(f"num_classes: {num_classes}")
    log.info(f"embedding_size: {embedding_size}")
    log.info(f"margin: {arcface_margin}°")
    log.info(f"scale: {arcface_scale}")

    log.info("Starting training...")

    model = train(
        epochs=epochs,
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        loss_optimizer=loss_optimizer,
        scheduler=scheduler,
        device=device,
    )

    log.info("Final evaluation on test set...")

    test_loss = evaluate(model, test_dataloader, loss_fn, device)
    log.info(f"Test loss: {test_loss:.4f}")


if __name__ == "__main__":
    main()
