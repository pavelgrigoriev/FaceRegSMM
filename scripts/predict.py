import os
import sys

import hydra
import torch
from PIL import Image
import torch.nn.functional as F
from omegaconf import DictConfig

project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_dir)

from src.models.model import RecSSM
from src.models.predict import predict

device = "cuda" if torch.cuda.is_available() else "cpu"
if device != "cuda":
    raise SystemError("Sorry, ssm only support with gpu/cuda")

@hydra.main(version_base=None, config_path=os.path.join(project_dir, "configs"), config_name="predict")
def main(cfg : DictConfig):
    first_img_path = cfg["first_img_path"]
    second_img_path = cfg["second_img_path"]
    model_path = cfg["model_path"]
    img_size = cfg["img_size"]

    first_img = Image.open(first_img_path).convert("RGB")
    second_img = Image.open(second_img_path).convert("RGB")
    model = RecSSM(img_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    embedding1 = predict(first_img, model, img_size, device)
    embedding2 = predict(second_img, model, img_size, device)
    similarity = F.cosine_similarity(embedding1, embedding2)
    if similarity > 0.5:
        print("its ones men")
    else:
        print("different men")
    print(similarity.item())

if __name__ == "__main__":
    main()

