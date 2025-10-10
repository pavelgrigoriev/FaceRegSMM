import torch

from src.utils.transform import get_transforms

def predict(img, model, imgsz, device):
    model.eval()
    _, base_transorm = get_transforms(imgsz)
    img = base_transorm(img).unsqueeze(0)
    with torch.no_grad():    
        output = model(img.to(device))
    return output