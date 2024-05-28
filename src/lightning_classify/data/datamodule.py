from typing import Any
from pathlib import Path

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from lightning_classify.transforms import ImageTransform

import lightning as L


class OnepieceImageDataModule(L.LightningDataModule):
    def __init__(self, root_path: str, batch_size: int = 32, num_workers: int = 11) -> None:
        super().__init__()
        self.root_path = Path(root_path)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str) -> None:
        self.trainset = self._build_dataset(mode="train")
        self.validset = self._build_dataset(mode="valid")
        
        self.class_names = self.trainset.classes
        self.class_to_idx = self.trainset.class_to_idx

    def _build_dataset(self, mode: str = "train") -> ImageFolder:
        dset = None
        
        if mode == "train":
            trans = ImageTransform(is_train=True)
            _path = self.root_path.joinpath(mode)
        elif mode == "valid":
            trans = ImageTransform(is_train=False)
            _path = self.root_path.joinpath(mode)
        else:
            raise FileNotFoundError("File not found.")
        
        dset = ImageFolder(str(_path), transform=trans)
        return dset
    
    def train_dataloader(self) -> Any:
        loader = DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        return loader
    
    def val_dataloader(self) -> Any:
        loader = DataLoader(
            self.validset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        return loader
