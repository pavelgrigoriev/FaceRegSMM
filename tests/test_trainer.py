import os
import sys
import tempfile

import pytorch_warmup as warmup
import torch
from pytorch_metric_learning import losses, miners
from torch.utils.data import DataLoader

project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_dir)

from unittest import mock

from dummy_blocks import DummySSMBlocks

from src.dataset.dataset import PersonDataset
from src.models.model import RecSSM
from src.models.trainer import train
from src.utils.transform import get_transforms
from utils import create_images


def test_train():
    device = "cpu"
    with tempfile.TemporaryDirectory() as temp_train_dir:
        with tempfile.TemporaryDirectory() as temp_val_dir:
            model = RecSSM(640)
            model.ssm_blocks = DummySSMBlocks()  # type: ignore
            create_images(temp_train_dir, num_img=5, num_trash_img=0)
            create_images(temp_val_dir, num_img=5, num_trash_img=0)
            batch_size = 1
            train_transform, base_transform = get_transforms()
            train_dataset = PersonDataset(temp_train_dir, train_transform)
            val_dataset = PersonDataset(temp_val_dir, base_transform)

            train_dataloader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            val_dataloader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=True
            )

            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 2 - 1, eta_min=1e-6
            )
            warmup_scheduler = warmup.LinearWarmup(optimizer, 1)
            loss_fn = losses.TripletMarginLoss(margin=0.2)
            miner = miners.TripletMarginMiner(margin=0.2, type_of_triplets="all")
            with mock.patch("hydra.core.hydra_config.HydraConfig.get") as mock_hydra:
                mock_hydra.return_value.runtime.output_dir = temp_train_dir
                model = train(
                    2,
                    model,
                    train_dataloader,
                    val_dataloader,
                    loss_fn,
                    miner,
                    optimizer,
                    scheduler,
                    warmup_scheduler,
                    1,
                    device,
                )
