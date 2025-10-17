import sys
import tempfile
from pathlib import Path

from torch.utils.data import DataLoader

from utils import create_images

project_dir = Path(__file__).resolve().parents[1]
sys.path.append(project_dir.as_posix())

from dummy_model import RecSSM

from src.dataset.dataset import PersonDataset
from src.models.evaluate import evaluate
from src.utils.transform import get_transforms


def test_evaluate():
    device = "cpu"
    with tempfile.TemporaryDirectory() as temp_test_dir:
        model = RecSSM(128)
        create_images(temp_test_dir, num_img=10, num_trash_img=0)
        batch_size = 1
        _, base_transform = get_transforms()
        test_dataset = PersonDataset(temp_test_dir, base_transform)

        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        loss = evaluate(model, test_dataloader, device)
