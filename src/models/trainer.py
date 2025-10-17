import logging
from pathlib import Path

import hydra
import torch
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

log = logging.getLogger(__name__)


def train(
    epochs,
    model,
    train_dataloader,
    val_dataloader,
    loss_fn,
    miner,
    optimizer,
    scheduler,
    device,
    warmup_scheduler=None,
    warmup_period=10,
    patience=35,
) -> torch.nn.Module:
    """
    train the model and validate it on the validation dataset.
    Args:
        epochs (int): number of epochs to train the model.
        model (torch.nn.Module): model to be trained.
        train_dataloader (DataLoader): dataloader for the training dataset.
        val_dataloader (DataLoader): dataloader for the validation dataset.
        loss_fn (torch.nn.Module): loss function to be used.
        miner (torch.nn.Module): miner to be used.
        optimizer (torch.optim.Optimizer): optimizer to be used.
        scheduler (torch.optim.lr_scheduler): learning rate scheduler to be used.
        device (str): device to be used for training. At on 16.10.2025 only 'cuda' is supported.
        warmup_scheduler (torch.optim.lr_scheduler, optional): warmup scheduler to be used. Defaults to None.
        warmup_period (int, optional): number of epochs for warmup. Defaults to 10.
        patience (int, optional): number of epochs to wait for improvement before early stopping. Defaults to 15.
    Returns:
        torch.nn.Module: trained model.
    """
    writer = SummaryWriter(
        log_dir=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir  # type: ignore
    )
    best_val_metric = 0.0
    accuracy_calculator = AccuracyCalculator(
        include=("precision_at_1", "r_precision", "mean_average_precision_at_r"),
        k="max_bin_count",
    )
    count = 0
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        train_progress_bar = tqdm(
            train_dataloader, desc=f"Training... Epoch {epoch+1}/{epochs}", leave=False
        )
        for images, labels in train_progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            embeddings = model(images)
            triplets = miner(embeddings, labels)
            loss = loss_fn(embeddings, labels, triplets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            train_progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = total_train_loss / len(train_dataloader)

        model.eval()
        val_progress_bar = tqdm(
            val_dataloader, desc=f"Validating... Epoch {epoch+1}/{epochs}", leave=False
        )
        emb = []
        lbs = []
        with torch.no_grad():
            for images, labels in val_progress_bar:
                images, labels = images.to(device), labels.to(device)

                embeddings = model(images)
                emb.append(embeddings.cpu())
                lbs.append(labels.cpu())

        val_embeddings = torch.cat(emb).cpu()
        val_labels = torch.cat(lbs).cpu()
        val_embeddings = torch.nn.functional.normalize(val_embeddings, p=2, dim=1)

        metrics = accuracy_calculator.get_accuracy(
            val_embeddings.numpy(),
            val_labels.numpy(),
            val_embeddings.numpy(),
            val_labels.numpy(),
            ref_includes_query=True,
        )

        recall_at_1 = metrics["precision_at_1"]
        r_precision = metrics["r_precision"]
        map_at_r = metrics["mean_average_precision_at_r"]

        writer.add_scalar("Loss/Train", avg_train_loss, epoch + 1)
        writer.add_scalar("Recall@1/Val", recall_at_1, epoch + 1)
        writer.add_scalar("R-Precision/Val", r_precision, epoch + 1)
        writer.add_scalar("MAP@R/Val", map_at_r, epoch + 1)
        writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], epoch + 1)
        if warmup_scheduler:
            with warmup_scheduler.dampening():
                if warmup_scheduler.last_step + 1 >= warmup_period:
                    scheduler.step()
        else:
            scheduler.step()

        if recall_at_1 > best_val_metric:
            best_val_metric = recall_at_1
            count = 0
            try:
                save_path = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir) / "best_model.pt"  # type: ignore
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "recall_at_1": recall_at_1,
                        "map_at_r": map_at_r,
                        "r_precision": r_precision,
                        "lr": optimizer.param_groups[0]["lr"],
                        "loss": avg_train_loss,
                        "patience": patience,
                        "warmup_period": warmup_period,
                        "patch_size": model.patch_embed.patch_size,
                        "img_size": model.patch_embed.img_size,
                        "margin": loss_fn.margin,
                        "model": model.__class__.__name__,
                        "optimizer": optimizer.__class__.__name__,
                        "loss_fn": loss_fn.__class__.__name__,
                        "miner": miner.__class__.__name__,
                        "scheduler": scheduler.__class__.__name__,
                        "warmup_scheduler": warmup_scheduler.__class__.__name__,
                    },
                    save_path,
                )
                log.info(f"Saved best model with Recall@1={recall_at_1:.4f}")
            except Exception as e:
                log.error(f"Error saving the model: {e}")
        else:
            count += 1

        if count >= patience:
            log.info(f"Early stopping at epoch {epoch+1}")
            break

        log.info(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Recall@1: {recall_at_1:.4f} | "
            f"R-Precision: {r_precision:.4f} | "
            f"MAP@R: {map_at_r:.4f}"
        )

    return model
