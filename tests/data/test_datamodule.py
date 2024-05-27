import unittest
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from lightning_classify.data import OnepieceImageDataModule


class TestOnepieceImageDataModule(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

        # Create train and val directories
        self.train_dir = Path(self.test_dir).joinpath("train")
        self.val_dir = Path(self.test_dir).joinpath("val")
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

    def test_train_dataloader(self):
        train_loader = self.data_loader.train_dataloader()
        self.assertIsInstance(train_loader, DataLoader)
        self.assertEqual(len(train_loader.dataset), 10)
        batch = next(iter(train_loader))
        images, labels = batch
        self.assertEqual(images.shape[0], 2)  # batch_size
        self.assertEqual(images.shape[1:], (3, 224, 224))  # transformed image size

    def test_val_dataloader(self):
        val_loader = self.data_loader.val_dataloader()
        self.assertIsInstance(val_loader, DataLoader)
        self.assertEqual(len(val_loader.dataset), 5)
        batch = next(iter(val_loader))
        images, labels = batch
        self.assertEqual(images.shape[0], 2)  # batch_size
        self.assertEqual(images.shape[1:], (3, 224, 224))  # transformed image size

    def test_class_names_and_class_to_idx(self):
        self.assertEqual(self.data_loader.class_names, ["fake_class"])
        self.assertEqual(self.data_loader.class_to_idx, {"fake_class": 0})
