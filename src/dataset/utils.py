from PIL import Image


def load_image(path, transform):
    try:
        img = Image.open(path).convert("RGB")
        if transform:
            img = transform(img)
    except Exception as e:
        raise ValueError(f"Error loading image {path}: {e}")
    return img
