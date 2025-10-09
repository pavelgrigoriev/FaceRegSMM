from tqdm import tqdm

def train(epochs, model, dataloader, loss_fn, optimizer, device):

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

