import torch

from src.utils.transform import get_transforms

def predict(img, model, img_size, device):
    model.eval()
    _, base_transorm = get_transforms(img_size)
    img = base_transorm(img).unsqueeze(0)
    with torch.no_grad():    
        output = model(img.to(device))
    return output