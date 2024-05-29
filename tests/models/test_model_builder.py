import unittest
import torch
from torch import nn
from torch.optim import Adam
from torchvision import models
from torchmetrics import Accuracy

from lightning_classify.models import ImageRecogModel


class TestImageRecogModel(unittest.TestCase):
    def setUp(self) -> None:
        self.lrate = 0.001
        self.num_classes = 18

        self.batch_size = 2
        self.image_size = (3, 224, 224)

        self.model = ImageRecogModel(lrate=self.lrate, num_classes=self.num_classes)

        # create dummy data
        self.dummy_images = torch.randn(self.batch_size, *self.image_size)
        self.dummy_labels = torch.randint(0, self.num_classes, (self.batch_size, ))

    def test_model_init(self):
        self.assertEqual(self.model.num_classes, self.num_classes)
        self.assertEqual(self.model.lrate, self.lrate)
        self.assertIsInstance(self.model.criterion, nn.CrossEntropyLoss)

        # backbone
        self.assertEqual(self.model.model.classifier[1].out_features, self.num_classes)

    def test_forward(self):
        output = self.model(self.dummy_images)
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))

    def test_optimizer(self):
        optimizer = self.model.configure_optimizers()
        self.assertIsInstance(optimizer, Adam)
        self.assertEqual(optimizer.defaults["lr"], self.lrate)
