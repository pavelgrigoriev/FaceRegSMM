import torch

from src.utils.transform import get_transforms


def predict(img, model, img_size, device):
    model.eval()
    _, base_transform = get_transforms(img_size)
    img = base_transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img.to(device))
    return output
