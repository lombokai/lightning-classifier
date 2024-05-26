from typing import Any
from pathlib import Path

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from lightning_classify.transforms import ImageTransform


class OnepieceImageDataLoader:
    def __init__(
            self, 
            root_path: str, 
            batch_size: int = 32, 
            num_workers: int = 11
        ) -> None:
        
        super().__init__()
        self.root_path = Path(root_path)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.trainset = self._build_dataset(mode="train")
        self.validset = self._build_dataset(mode="valid")

        self.class_names = self.trainset.classes
        self.class_to_idx = self.trainset.class_to_idx

        self.train_loader = self._build_dataloader(mode="train")
        self.valid_loader = self._build_dataloader(mode="valid")

    def _build_dataset(self, mode: str = "train") -> ImageFolder:
        dset = None
        
        if mode == "train":
            trans = ImageTransform(is_train=True)
            _path = self.root_path.joinpath(mode)
        elif mode == "valid":
            trans = ImageTransform(is_train=False)
            _path = self.root_path.joinpath(mode)
        else:
            raise FileNotFoundError("File does not exists.")
        
        dset = ImageFolder(str(_path), transform=trans)
        return dset

    def _build_dataloader(self, mode: str = "train") -> DataLoader:
        if mode == "train":
            dset = self.trainset
            shuffle = True
        elif mode == "valid":
            dset = self.validset
            shuffle = False
        else:
            FileNotFoundError("File does not exist.")
        
        loader = DataLoader(
            dset, 
            batch_size=self.batch_size, 
            shuffle=shuffle, 
            pin_memory=True
        )

        return loader
