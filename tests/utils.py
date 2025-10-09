from pathlib import Path
import random
from PIL import Image
import numpy as np

def create_images(image_dir, num_img=100, num_trash_img=20):
    extension_list = ["PNG", "png", "jpeg", "JPEG", "jpg","JPG"]
    folder = Path(image_dir)
    if num_img != 0:
        for i in range(num_img):
            array = np.random.randint(0, 256, (np.random.randint(128,2880), np.random.randint(128,2880), 3), dtype=np.uint8)
            image = Image.fromarray(array)
            image.save(f"{folder}/temp_img_{i}.{random.choice(extension_list)}")
    if num_trash_img != 0:        
        extension_list = ["a", "..", "  ", "12", "aa",","]
        for i in range(num_trash_img):
            with open(f"{folder}/temp_trash_{i}.{random.choice(extension_list)}", 'w') as f:
                pass 
    if num_img == 0 and num_trash_img == 0:
        raise ValueError("Error: num_img and num_trash_img == 0!") 