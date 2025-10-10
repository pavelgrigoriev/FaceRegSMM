import torch
from tqdm import tqdm

def evaluate(model, test_dataloader, loss_fn, device):
        model.eval()
        total_test_loss = 0.0
        test_dataloader = tqdm(test_dataloader, desc=f"Validating on test set...", leave=False)
        with torch.no_grad():    
            for anchor, positive, negative in test_dataloader:
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
                a_emb = model(anchor)
                p_emb = model(positive)
                n_emb = model(negative)
                loss = loss_fn(a_emb, p_emb, n_emb)
                total_test_loss += loss.item()
                test_dataloader.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_test_loss = total_test_loss / len(test_dataloader)

        print(f"Test avg loss: {avg_test_loss:.6f}")

        return avg_test_loss