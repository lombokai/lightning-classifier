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
        self.create_dummy_images(self.train_dir, num_images=10, image_size=(321, 251))
        self.create_dummy_images(self.val_dir, num_images=5, image_size=(251, 321))

        # Create the data loader
        self.data_module = OnepieceImageDataModule(root_path=self.test_dir, batch_size=2, num_workers=1)

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def create_dummy_images(self, dir_path, num_images, image_size):
        dir_path.joinpath("fake_class").mkdir(parents=True, exist_ok=True)
        for i in range(num_images):
            image = Image.fromarray(np.uint8(np.random.rand(*image_size, 3) * 255))
            image.save(dir_path / "fake_class" / f"img_{i}.jpg")

    def test_setup(self):
        self.data_module.setup(stage="fit")

        self.assertIsInstance(self.data_module.trainset, ImageFolder)
        self.assertIsInstance(self.data_module.validset, ImageFolder)
        self.assertEqual(self.data_module.class_names, ["fake_class"])
        self.assertEqual(self.data_module.class_to_idx, {"fake_class": 0})
        self.assertEqual(len(self.data_module.trainset), 10)
        self.assertEqual(len(self.data_module.validset), 5)

    def test_train_dataloader(self):
        self.data_module.setup(stage="fit")

        train_loader = self.data_module.train_dataloader()

        self.assertIsInstance(train_loader, DataLoader)
        
        train_in, train_out = next(iter(train_loader))

        self.assertEqual(train_in.shape[0], 2)
        self.assertEqual(train_in.shape[1:], (3, 224, 224))

    def test_val_dataloader(self):
        self.data_module.setup(stage="fit")

        val_loader = self.data_module.val_dataloader()

        self.assertIsInstance(val_loader, DataLoader)
        
        val_in, val_out = next(iter(val_loader))

        self.assertEqual(val_in.shape[0], 2)
        self.assertEqual(val_in.shape[1:], (3, 224, 224))
