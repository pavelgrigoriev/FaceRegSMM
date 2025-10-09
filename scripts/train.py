import os, sys
project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_dir)

import torch
from torch import nn 
from torch.utils.data import DataLoader
from src.dataset.dataset import TripletDataset
from src.models.model import RecSSM
from src.utils.transform import transform
from tqdm import tqdm

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

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for anchor, positive, negative in progress_bar:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            a_emb = model(anchor)
            p_emb = model(positive)
            n_emb = model(negative)
            loss = loss_fn(a_emb, p_emb, n_emb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}]  Avg loss: {avg_loss:.6f}")

if __name__ == "__main__":
    main()

