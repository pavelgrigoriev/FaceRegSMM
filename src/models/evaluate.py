import torch
from tqdm import tqdm


def evaluate(model, test_dataloader, loss_fn, device, miner=None):
    """
    Evaluate model on test set
    """
    model.eval()
    total_test_loss = 0.0

    test_progress_bar = tqdm(
        test_dataloader, desc="Evaluating on test set", leave=False
    )

    with torch.no_grad():
        for images, labels in test_progress_bar:
            images, labels = images.to(device), labels.to(device)

            embeddings = model(images)

            if miner is not None:
                hard_pairs = miner(embeddings, labels)
                loss = loss_fn(embeddings, labels, hard_pairs)
            else:
                loss = loss_fn(embeddings, labels)

            total_test_loss += loss.item()
            test_progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_test_loss = total_test_loss / len(test_dataloader)
    return avg_test_loss


def extract_embeddings(model, dataloader, device):
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting embeddings"):
            images = images.to(device)

            # Получаем embeddings
            embeddings = model(images)

            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return all_embeddings, all_labels
