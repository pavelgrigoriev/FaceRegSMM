import os, sys


project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_dir)

import torch
from torch import nn 
from torch.utils.data import DataLoader
from src.dataset.dataset import TripletDataset
from src.models.model import RecSSM
from src.utils.transform import get_transforms
from src.models.trainer import train
from src.models.evaluate import evaluate
import logging
import hydra
from omegaconf import DictConfig

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
    imgsz = cfg["imgsz"]
    log.info(f"data_path: {data_path}")
    log.info(f"epochs: {epochs}")
    log.info(f"batch_size: {batch_size}")
    log.info(f"imgsz: {imgsz}")
    train_transform, base_transform = get_transforms()
    train_dataset = TripletDataset(os.path.join(data_path, "train"), train_transform, imgsz)
    test_dataset = TripletDataset(os.path.join(data_path, "test"), base_transform, imgsz=imgsz)
    val_dataset = TripletDataset(os.path.join(data_path, "val"), base_transform, imgsz=imgsz)
    log.info(f"Len train_dataset: {len(train_dataset)}")
    log.info(f"Len test_dataset: {len(test_dataset)}")
    log.info(f"Len val_dataset: {len(val_dataset)}")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8)
    
    model = RecSSM().to(device)

    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    loss_fn = nn.TripletMarginLoss(0.2)
    model = train(epochs, model, train_dataloader, val_dataloader, loss_fn, optimizer, scheduler,device)
    loss = evaluate(model, test_dataloader, loss_fn, device)
    log.info(f"Loss on test set: {loss}")
    
if __name__ == "__main__":
    main()

