import os
import sys
import tempfile

from torch import nn
from torch.utils.data import DataLoader

project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_dir)

from utils import create_images
from src.models.model import RecSSM
from dummy_blocks import DummySSMBlocks
from src.models.evaluate import evaluate
from src.dataset.dataset import TripletDataset
from src.utils.transform import get_transforms

def test_evaluate():
    device = "cpu"
    with tempfile.TemporaryDirectory() as temp_test_dir:
        model = RecSSM()
        model.ssmblocks = DummySSMBlocks() # type: ignore
        create_images(temp_test_dir, num_img=10, num_trash_img=0)
        batch_size = 1
        _, base_transform = get_transforms()
        test_dataset = TripletDataset(temp_test_dir, base_transform)

        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        loss_fn = nn.TripletMarginLoss(0.2)
        loss = evaluate(model, test_dataloader, loss_fn, device)
