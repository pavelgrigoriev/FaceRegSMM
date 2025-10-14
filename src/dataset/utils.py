import random
from pathlib import Path

from PIL import Image


def load_image(path, img_size, transform):
    try:
        img = Image.open(path).convert("RGB").resize((img_size, img_size))
        if transform:
            img = transform(img)
    except Exception as e:
        raise ValueError(f"Error loading image {path}: {e}")
    return img


def create_train_val_test_splits(
    root_dir: str,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    min_images_per_person: int = 5,
    min_images_for_split: int = 3,
    seed: int = 42,
):

    random.seed(seed)
    root = Path(root_dir)

    person_to_paths = {}
    for person_dir in root.iterdir():
        if person_dir.is_dir():
            paths = [p for p in person_dir.glob("*") if p.is_file()]
            if len(paths) >= min_images_per_person:
                person_to_paths[person_dir.name] = paths

    sorted_persons = sorted(person_to_paths.keys())
    person_to_id = {name: i for i, name in enumerate(sorted_persons)}

    train_paths = []
    val_paths = []
    test_paths = []

    for _, paths in person_to_paths.items():
        paths_copy = paths.copy()
        random.shuffle(paths_copy)

        n_total = len(paths_copy)
        n_test = max(min_images_for_split, int(n_total * test_ratio))
        n_val = max(min_images_for_split, int(n_total * val_ratio))
        n_train = n_total - n_test - n_val

        if n_train < min_images_for_split:
            train_paths.extend(paths_copy)
        else:
            test_paths.extend(paths_copy[:n_test])
            val_paths.extend(paths_copy[n_test : n_test + n_val])
            train_paths.extend(paths_copy[n_test + n_val :])

    return train_paths, val_paths, test_paths, person_to_id
