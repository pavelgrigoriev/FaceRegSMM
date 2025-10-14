import logging

import torch
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from tqdm import tqdm

log = logging.getLogger(__name__)


def evaluate(model, test_dataloader, device):
    model.eval()
    accuracy_calculator = AccuracyCalculator(
        include=("precision_at_1", "r_precision", "mean_average_precision_at_r"),
        k="max_bin_count",
    )
    test_progress_bar = tqdm(
        test_dataloader, desc=f"Validating on test set...", leave=False
    )
    emb = []
    lbs = []

    with torch.no_grad():
        for images, labels in test_progress_bar:
            images, labels = images.to(device), labels.to(device)

            embeddings = model(images)
            emb.append(embeddings.cpu())
            lbs.append(labels)

    val_embeddings = torch.cat(emb)
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
    log.info(
        f"Recall@1: {recall_at_1:.4f} | "
        f"R-Precision: {r_precision:.4f} | "
        f"MAP@R: {map_at_r:.4f}"
    )
