from tqdm import tqdm

def train(epochs, model, train_dataloader, val_dataloader, loss_fn, optimizer, device):

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        train_progress_bar = tqdm(train_dataloader, desc=f"Training... Epoch {epoch+1}/{epochs}", leave=False)

        for anchor, positive, negative in train_progress_bar:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            a_emb = model(anchor)
            p_emb = model(positive)
            n_emb = model(negative)
            loss = loss_fn(a_emb, p_emb, n_emb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            train_progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = total_train_loss / len(train_dataloader)

        model.eval()
        total_val_loss = 0.0
        val_progress_bar = tqdm(val_dataloader, desc=f"Validating... Epoch {epoch+1}/{epochs}", leave=False)

        for anchor, positive, negative in val_progress_bar:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            a_emb = model(anchor)
            p_emb = model(positive)
            n_emb = model(negative)
            loss = loss_fn(a_emb, p_emb, n_emb)
            total_val_loss += loss.item()
            val_progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_val_loss = total_val_loss / len(train_dataloader)

        print(f"Epoch [{epoch+1}/{epochs}]  Train avg loss: {avg_train_loss:.6f} Val avg loss: {avg_val_loss:.6f}")

