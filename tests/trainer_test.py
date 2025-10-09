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
    with tempfile.TemporaryDirectory() as temp_train_dir:
        with tempfile.TemporaryDirectory() as temp_val_dir:
            model = RecSSM()
            model.ssmblocks = DummySSMBlocks() # type: ignore
            create_images(temp_train_dir, num_img=10, num_trash_img=0)
            create_images(temp_val_dir, num_img=10, num_trash_img=0)
            batch_size = 1
            train_dataset = TripletDataset(temp_train_dir, transform)
            val_dataset = TripletDataset(temp_val_dir, transform)

            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

            optimizer = torch.optim.AdamW(model.parameters())
            loss_fn = nn.TripletMarginLoss(0.2)
            train(1, model, train_dataloader,val_dataloader,loss_fn,optimizer,device)
