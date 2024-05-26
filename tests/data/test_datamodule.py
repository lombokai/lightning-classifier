import unittest
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import numpy as np

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from lightning_classify.data import OnepieceImageDataLoader


class TestOnepieceImageDataLoader(unittest.TestCase):
    
    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp()

        # create temporal train and valid directory
        self.train_dir = Path(self.test_dir).joinpath("train")
        self.val_dir = Path(self.test_dir).joinpath("valid")
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.val_dir.mkdir(parents=True, exist_ok=True)

        self.create_dummy_images(self.train_dir, num_images=10, image_size=(234, 451))
        self.create_dummy_images(self.val_dir, num_images=5, image_size=(321, 218))

        self.data_loader = OnepieceImageDataLoader(
            root_path=self.test_dir,
            batch_size=2,
            num_workers=0
        )
    
    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir)

    def create_dummy_images(self, dir_path, num_images, image_size):
        # create subdirectory for class names

        dir_path.joinpath("fake_class").mkdir(parents=True, exist_ok=True)

        # create directory images
        for i in range(num_images):
            fake_image = np.uint8(np.random.rand(*image_size, 3) * 255)
            image = Image.fromarray(fake_image)
            image.save(dir_path / "fake_class" / f"img_{i}.png")
    
    def test_build_dataset(self):
        # build dataset
        fake_trainset = self.data_loader._build_dataset(mode="train")
        fake_valset = self.data_loader._build_dataset(mode="valid")

        # test dataset type
        self.assertIsInstance(fake_trainset, ImageFolder)
        self.assertIsInstance(fake_valset, ImageFolder)

        # test how many dataset
        self.assertTrue(len(fake_trainset), 10)
        self.assertTrue(len(fake_valset), 5)

    def test_build_dataloader(self):
        # build loader
        fake_train_loader = self.data_loader._build_dataloader(mode="train")
        fake_val_loader = self.data_loader._build_dataloader(mode="valid")

        # test loader type
        self.assertIsInstance(fake_train_loader, DataLoader)
        self.assertIsInstance(fake_val_loader, DataLoader)

        # iterate over loader
        train_in, train_out = next(iter(fake_train_loader))
        val_in, val_out = next(iter(fake_val_loader))

        # test batch size
        self.assertTrue(len(train_in), 2)
        self.assertTrue(len(val_in), 2)
    
    def test_class_names_and_class_to_idx(self):
        self.assertTrue(self.data_loader.class_names, ["fake_class"])
        self.assertTrue(self.data_loader.class_to_idx, {"fake_class": 0})
