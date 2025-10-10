from PIL import Image

def load_image(path, imgsz, transform):
    img = Image.open(path).convert("RGB").resize((imgsz, imgsz))
    if transform:
        img = transform(img)
    return img
