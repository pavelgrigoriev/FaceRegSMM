from PIL import Image

def load_image(path, img_size, transform):
    img = Image.open(path).convert("RGB").resize((img_size, img_size))
    if transform:
        img = transform(img)
    return img
