import unittest
from PIL import Image
import numpy as np
import torch
from lightning_classify.transforms import ImageTransform

class TestImageTransform(unittest.TestCase):
    def setUp(self) -> None:
        
        # create fake image to test
        fake_image = np.uint8(np.random.rand(321, 298, 3) * 255)
        self.sample_image = Image.fromarray(fake_image)
    
    def test_train_transform(self):
        
        # instantiate
        trans = ImageTransform(is_train=True)

        # apply transformation
        transformed_image = trans(self.sample_image)

        # check
        expected_output_dim = torch.Size([3, 224, 224])

        self.assertIsInstance(transformed_image, torch.Tensor)
        self.assertEqual(transformed_image.shape, expected_output_dim)

    def test_eval_transform(self):
        # instantiate
        trans = ImageTransform(is_train=False)

        # apply transformation
        transformed_image = trans(self.sample_image)

        # check
        expected_output_dim = torch.Size([3, 224, 224])

        self.assertIsInstance(transformed_image, torch.Tensor)
        self.assertEqual(transformed_image.shape, expected_output_dim)
