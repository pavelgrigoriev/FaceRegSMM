import os, sys
import tempfile
from torch import nn

from src.dataset.dataset import TripletDataset
from src.models.trainer import train
from src.utils.transform import transform
from torch.utils.data import DataLoader


project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_dir)

import torch
from src.models.model import RecSSM
from dummy_blocks import DummySSMBlocks
from utils import create_images

def test_train():
    device = "cpu"
    with tempfile.TemporaryDirectory() as temp_dir:
        model = RecSSM()
        model.ssmblocks = DummySSMBlocks() # type: ignore
        create_images(temp_dir, num_img=10, num_trash_img=0)
        data_path = temp_dir
        epochs = 1
        batch_size = 1
        dataset = TripletDataset(data_path, transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters())
        loss_fn = nn.TripletMarginLoss(0.2)
        train(epochs, model, dataloader,loss_fn,optimizer,device)
