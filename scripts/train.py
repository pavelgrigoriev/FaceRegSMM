import os, sys

project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_dir)

import torch
from torch import nn 
from torch.utils.data import DataLoader
from src.dataset.dataset import TripletDataset
from src.models.model import RecSSM
from src.utils.transform import transform
from src.models.trainer import train

import hydra
from omegaconf import DictConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

@hydra.main(version_base=None, config_path=os.path.join(project_dir, "configs"), config_name="train")
def main(cfg : DictConfig):
    data_path = cfg["data_path"]
    epochs = cfg["epochs"]
    batch_size = cfg["batch_size"]
    dataset = TripletDataset(data_path, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = RecSSM().to(device)

    optimizer = torch.optim.AdamW(model.parameters())
    loss_fn = nn.TripletMarginLoss(0.2)
    train(epochs, model, dataloader,loss_fn,optimizer,device)

if __name__ == "__main__":
    main()

