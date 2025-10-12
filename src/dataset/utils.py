from PIL import Image


def load_image(path, img_size, transform):
    try:
        img = Image.open(path).convert("RGB").resize((img_size, img_size))
        if transform:
            img = transform(img)
    except Exception as e:
        raise ValueError(f"Error loading image {path}: {e}")
    return img
