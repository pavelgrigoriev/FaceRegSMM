import sys
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from PIL import Image

project_dir = Path(__file__).resolve().parents[1]
sys.path.append(project_dir.as_posix())

from scripts.utils import load_model
from src.models.predict import predict

device = "cuda" if torch.cuda.is_available() else "cpu"
if device != "cuda":
    raise SystemError("Sorry, ssm only support with gpu/cuda")


@hydra.main(
    version_base=None,
    config_path=str(project_dir / "configs"),
    config_name="predict",
)
def main(cfg: DictConfig) -> None:
    first_img_path = cfg.get("first_img_path")
    if not first_img_path:
        raise ValueError("first_img_path must be specified in the config file.")
    first_img_path = Path(first_img_path)
    if not first_img_path.exists():
        raise FileNotFoundError(f"First image file not found: {first_img_path}")

    second_img_path = cfg.get("second_img_path")
    if not second_img_path:
        raise ValueError("second_img_path must be specified in the config file.")
    second_img_path = Path(second_img_path)
    if not second_img_path.exists():
        raise FileNotFoundError(f"Second image file not found: {second_img_path}")

    model_path = cfg.get("model_path")
    if not model_path:
        raise ValueError("model_path must be specified in the config file.")
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    img_size = cfg.get("img_size")
    similarity_threshold = cfg.get("similarity_threshold")
    first_img = Image.open(first_img_path).convert("RGB")
    second_img = Image.open(second_img_path).convert("RGB")

    model = load_model(model_path, img_size, device)

    embedding1 = predict(first_img, model, img_size, device)
    embedding2 = predict(second_img, model, img_size, device)
    similarity = F.cosine_similarity(embedding1, embedding2)
    if similarity > similarity_threshold:
        print("its ones men")
    else:
        print("different men")
    print(similarity.item())


if __name__ == "__main__":
    main()
