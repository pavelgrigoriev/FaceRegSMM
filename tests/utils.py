import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

project_dir = Path(__file__).resolve().parents[1]
sys.path.append(project_dir.as_posix())

from src.constants import EXTENSION_LIST


def create_images(image_dir, num_img=10, num_trash_img=5):

    person_list = [f"person_{i}" for i in range(10)]
    folder = Path(image_dir)
    if num_img != 0:
        for person in person_list:
            person_fodler = folder / person
            person_fodler.mkdir(parents=True, exist_ok=True)
            for i in range(num_img):
                array = np.random.randint(
                    0,
                    256,
                    (np.random.randint(128, 2880), np.random.randint(128, 2880), 3),
                    dtype=np.uint8,
                )
                image = Image.fromarray(array)
                image.save(
                    f"{person_fodler}/temp_img_{i}.{random.choice(EXTENSION_LIST)}"
                )
    if num_trash_img != 0:
        trash_extension_list = ["a", "..", "  ", "12", "aa", ","]
        for person in person_list:
            person_fodler = folder / person
            person_fodler.mkdir(parents=True, exist_ok=True)
            for i in range(num_trash_img):
                with open(
                    f"{person_fodler}/temp_trash_{i}.{random.choice(trash_extension_list)}",
                    "w",
                ) as f:
                    pass
    if num_img == 0 and num_trash_img == 0:
        raise ValueError("Error: num_img and num_trash_img == 0!")
