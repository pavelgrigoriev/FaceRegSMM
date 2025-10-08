import torch
from torch import nn 
from torch.utils.data import DataLoader
from dataset import TripletDataset
from model import RecSSM
from transform import transform
from tqdm import tqdm


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
dataset = TripletDataset("/home/ecomg/Downloads/00000-20251008T085118Z-1-001/00000", transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

model = RecSSM().to(device)

optimizer = torch.optim.AdamW(model.parameters())
loss_fn = nn.TripletMarginLoss(0.2)

for epoch in range(10):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/10", leave=False)

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
    print(f"Epoch [{epoch+1}/10]  Avg loss: {avg_loss:.6f}")