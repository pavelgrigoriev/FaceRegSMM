import logging
import os

import hydra
import torch
from tqdm import tqdm

log = logging.getLogger(__name__)


def train(
    epochs,
    model,
    train_dataloader,
    val_dataloader,
    loss_fn,
    optimizer,
    scheduler,
    device,
    patience=15,
):
    count = 0
    best_val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        train_progress_bar = tqdm(
            train_dataloader, desc=f"Training... Epoch {epoch+1}/{epochs}", leave=False
        )
        for anchor, positive, negative in train_progress_bar:
            anchor, positive, negative = (
                anchor.to(device),
                positive.to(device),
                negative.to(device),
            )
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
        val_progress_bar = tqdm(
            val_dataloader, desc=f"Validating... Epoch {epoch+1}/{epochs}", leave=False
        )
        with torch.no_grad():
            for anchor, positive, negative in val_progress_bar:
                anchor, positive, negative = (
                    anchor.to(device),
                    positive.to(device),
                    negative.to(device),
                )
                a_emb = model(anchor)
                p_emb = model(positive)
                n_emb = model(negative)
                loss = loss_fn(a_emb, p_emb, n_emb)
                total_val_loss += loss.item()
                val_progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_val_loss = total_val_loss / len(val_dataloader)
        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            count = 0
            try:
                torch.save(model.state_dict(), (os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, "best_model.pt")))  # type: ignore
            except:
                log.error("Error saving the model.")

        else:
            count += 1
        if count >= patience:
            log.info(f"Early stopping on {epoch}")
            break

        log.info(
            f"Epoch [{epoch+1}/{epochs}]  Train avg loss: {avg_train_loss:.6f} Val avg loss: {avg_val_loss:.6f}"
        )

    return model
