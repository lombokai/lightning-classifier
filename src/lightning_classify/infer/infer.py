import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from lightning_classify.utils import downloader
from lightning_classify.models import ImageRecogModelV1
from lightning_classify.transforms import ImageTransform


class ImageRecognition:
    def __init__(self, device: torch.device, model_path=None):
        
        path_to_save = str(self._get_cache_dir()) + "/model.ckpt"
        if model_path is None:
            downloader(path_to_save)
            self.model_path = path_to_save
        else:
            self.model_path = Path(model_path)
            if not self.model_path.exists():
                raise FileNotFoundError("Model does not exist, check your model location and read README for more information")
        

        self.device = torch.device(device)
        # self.class_dict = {
        #     0: "Ace",
        #     1: "Akainu",
        #     2: "Brook",
        #     3: "Chopper",
        #     4: "Crocodile",
        #     5: "Franky",
        #     6: "Jinbei",
        #     7: "Kurohige",
        #     8: "Law",
        #     9: "Luffy",
        #     10: "Mihawk",
        #     11: "Nami",
        #     12: "Rayleigh",
        #     13: "Robin",
        #     14: "Sanji",
        #     15: "Shanks",
        #     16: "Usopp",
        #     17: "Zoro",
        # } 
        self.model = ImageRecogModelV1.load_from_checkpoint(self.model_path, 
                                                            map_location=self.device)
        self.class_dict = self.model.class_dict
        
    def _get_cache_dir(self):
        if sys.platform.startswith("win"):
            cache_dir = Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "OnepieceClassifyCache"
        else:
            cache_dir = Path.home() / ".cache" / "OnepieceClassifyCache"

        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def pre_process(
        self, image: Optional[str | np.ndarray | Image.Image]
    ) -> torch.Tensor:
        trans = ImageTransform(is_train=False)

        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
            img = trans(img)
            img = img.unsqueeze(0)

        elif isinstance(image, Image.Image):
            img = image.convert("RGB")
            img = trans(img).unsqueeze(0)
            img = img.unsqueeze(0)

        elif isinstance(image, np.ndarray):
            img = image.astype(np.uint8)
            img = Image.fromarray(img).convert("RGB")
            img = trans(img)
            img = img.unsqueeze(0)

        else:
            print("Image type not recognized")

        return img

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        self.model.eval()

        result = self.model(image_tensor)
        return result

    def post_process(self, output: torch.Tensor) -> Tuple[str, float]:
        confidence, class_idx = torch.softmax(output, dim=1).max(1) 
        class_names = self.class_dict[class_idx.item()]

        return (class_names, confidence.item())

    def predict(
        self, image: Optional[str | np.ndarray | Image.Image]
    ) -> Dict[str, str]:
        tensor_img = self.pre_process(image=image)
        logits = self.forward(tensor_img)
        class_names, confidence = self.post_process(logits)

        return {"class_names": class_names, "confidence": f"{confidence:.4f}"}
