import os
import sys
import logging


import hydra
import torch
from torch import nn 
from omegaconf import DictConfig
from torch.utils.data import DataLoader

project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_dir)

from src.models.model import RecSSM
from src.models.trainer import train
from scripts.utils import save_samples
from src.models.evaluate import evaluate
from src.dataset.dataset import TripletDataset
from src.utils.transform import get_transforms

log = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"

if device != "cuda":
    raise SystemError("Sorry, ssm only support with gpu/cuda")

log.info(f"Using device: {device}")


@hydra.main(version_base=None, config_path=os.path.join(project_dir, "configs"), config_name="train")
def main(cfg : DictConfig) -> None:
    data_path = cfg["data_path"]
    epochs = cfg["epochs"]
    batch_size = cfg["batch_size"]
    img_size = cfg["img_size"]
    lr = cfg["lr"]
    log.info(f"data_path: {data_path}")
    log.info(f"epochs: {epochs}")
    log.info(f"batch_size: {batch_size}")
    log.info(f"img_size: {img_size}")
    log.info(f"lr: {lr}")
    train_transform, base_transform = get_transforms(img_size)
    train_dataset = TripletDataset(os.path.join(data_path, "train"), train_transform, img_size)
    test_dataset = TripletDataset(os.path.join(data_path, "test"), base_transform, img_size=img_size)
    val_dataset = TripletDataset(os.path.join(data_path, "val"), base_transform, img_size=img_size)
    log.info(f"Len train_dataset: {len(train_dataset)}")
    log.info(f"Len test_dataset: {len(test_dataset)}")
    log.info(f"Len val_dataset: {len(val_dataset)}")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8)
    
    model = RecSSM(img_size).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    loss_fn = nn.TripletMarginLoss(0.2)
    train_iter = iter(train_dataloader)
    a,p,n = next(train_iter)

    save_samples(a,p,n)

    model = train(epochs, model, train_dataloader, val_dataloader, loss_fn, optimizer, scheduler,device)
    loss = evaluate(model, test_dataloader, loss_fn, device)
    log.info(f"Loss on test set: {loss}")
    
if __name__ == "__main__":
    main()

