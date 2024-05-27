from typing import Any
from pathlib import Path

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from lightning_classify.transforms import ImageTransform

from lightning import LightningDataModule


class OnepieceImageDataModule(LightningDataModule):
    def __init__(self, root_path: str, batch_size: int = 32, num_workers: int = 11) -> None:
        super().__init__()
        self.root_path = Path(root_path)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.trainset = ImageFolder(
            root=self.root_path.joinpath("train"),
            transform=ImageTransform(is_train=True)
        )

        self.validset = ImageFolder(
            root=self.root_path.joinpath("val"),
            transform=ImageTransform(is_train=False)
        )

        self.class_names = self.trainset.classes
        self.class_to_idx = self.trainset.class_to_idx

    def train_dataloader(self) -> DataLoader:
        loader = DataLoader(
            self.trainset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=True
            )
        return loader
    
    def val_dataloader(self) -> DataLoader:
        loader = DataLoader(
            self.validset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=False
            )
        return loader
