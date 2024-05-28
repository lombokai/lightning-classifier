import unittest
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from lightning_classify.data import OnepieceImageDataModule


class TestOnepieceImageDataModule(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

        # Create train and val directories
        self.train_dir = Path(self.test_dir).joinpath("train")
        self.val_dir = Path(self.test_dir).joinpath("valid")
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.val_dir.mkdir(parents=True, exist_ok=True)

        # Create dummy images in train and val directories
        self.create_dummy_images(self.train_dir, num_images=10, image_size=(256, 256))
        self.create_dummy_images(self.val_dir, num_images=5, image_size=(256, 256))

        # Create the data loader
        self.data_loader = OnepieceImageDataModule(root_path=self.test_dir, batch_size=2, num_workers=1)

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def create_dummy_images(self, dir_path, num_images, image_size):
        dir_path.joinpath("fake_class").mkdir(parents=True, exist_ok=True)
        for i in range(num_images):
            image = Image.fromarray(np.uint8(np.random.rand(*image_size, 3) * 255))
            image.save(dir_path / "fake_class" / f"img_{i}.jpg")

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
        
        self.assertEqual(len(train_in), 2)
        self.assertEqual(len(val_in), 2)

    def test_class_names_and_class_to_idx(self):
        self.assertEqual(self.data_loader.class_names, ["fake_class"])
        self.assertEqual(self.data_loader.class_to_idx, {"fake_class": 0})
