from PIL import Image


def load_image(path, transform) -> Image.Image:
    """Load an image from the specified path and apply the given transformation.
    Args:
        path (Path): Path to the image file.
        transform (callable): Transformation to be applied to the image.
    Returns:
        PIL.Image: Transformed image.
    """
    try:
        img = Image.open(path).convert("RGB")
        if transform:
            img = transform(img)
    except Exception as e:
        raise ValueError(f"Error loading image {path}: {e}")
    return img
