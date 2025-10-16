import base64
import json
import os
import sys
from io import BytesIO
from pathlib import Path

from scripts.utils import load_model
from src.constants import EXTENSION_LIST, HTML_TEMPLATE

project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_dir)

import logging

import hydra
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from PIL import Image
from regex import T
from sklearn.manifold import TSNE
from tqdm import tqdm

from src.models.model import RecSSM
from src.models.predict import predict

device = "cuda" if torch.cuda.is_available() else "cpu"
if device != "cuda":
    raise SystemError("Sorry, ssm only support with gpu/cuda")

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path=os.path.join(project_dir, "configs"),
    config_name="emb_vizualize",
)
def main(cfg: DictConfig):
    data_path = cfg["data_path"]
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"data_path not found: {data_path}")
    log.info(f"data_path: {data_path}")
    paths = sorted(
        [p for p in Path(data_path).rglob("*") if p.suffix[1:] in EXTENSION_LIST]
    )[:500]

    model_path = cfg["model_path"]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    log.info(f"model_path: {model_path}")
    img_size = cfg["img_size"]
    log.info(f"img_size: {img_size}")

    model = load_model(model_path, img_size, device)

    emb_list = []

    for path in tqdm(paths):
        img = Image.open(path).convert("RGB")
        embedding = predict(img, model, img_size, device)
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        emb_list.append(embedding.cpu().numpy())

    embeded = np.concatenate(emb_list, axis=0)
    log.info("Fitting TSNE")
    model = TSNE(n_components=2, learning_rate="auto", init="random", perplexity=3)
    embeded = model.fit_transform(embeded)
    log.info("TSNE fit done")

    base64_images = []
    for path in tqdm(paths, desc="Encoding images"):
        img = Image.open(path).convert("RGB")
        img.thumbnail((150, 150))
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        base64_images.append(f"data:image/png;base64,{img_base64}")

    filenames = [path.name for path in paths]

    df = pd.DataFrame(
        {
            "x": embeded[:, 0],
            "y": embeded[:, 1],
            "filename": filenames,
            "img_b64": base64_images,
        }
    )

    data_dict = {
        "x": df["x"].tolist(),
        "y": df["y"].tolist(),
        "filename": df["filename"].tolist(),
        "img_b64": df["img_b64"].tolist(),
    }
    data_json = json.dumps(data_dict)
    html_content = HTML_TEMPLATE.format(data_json=data_json)

    with open(
        os.path.join(
            hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,  # type: ignore
            "emb_vizualize.html",
        ),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(html_content)


if __name__ == "__main__":
    main()
