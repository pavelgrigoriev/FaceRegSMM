import torch

from src.utils.transform import get_transforms


def predict(img, model, img_size, device) -> torch.Tensor:
    """Predict the output for a single image using the trained model.
    Args:
        img (PIL.Image): Input image.
        model (torch.nn.Module): Trained model.
        img_size (int): Size to which the image will be resized.
        device (str): Device to run the model on ('cpu' or 'cuda'). At on 16.10.2025 only 'cuda' is supported.
    Returns:
        torch.Tensor: Model output.
    """
    model.eval()
    _, base_transorm = get_transforms(img_size)
    img = base_transorm(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img.to(device))
    return output
