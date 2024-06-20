import sys
sys.path.append("src")
from argparse import ArgumentParser

import torch
import torch.nn as nn
import lightning as L

from lightning_classify.data import OnepieceImageDataModule
from lightning_classify.models import ImageRecogModelV1

torch.set_float32_matmul_precision('medium')

# parser
parser = ArgumentParser()

parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--accelerator", type=str, default="gpu")
parser.add_argument("--max_epochs", type=int, default=10)
args = parser.parse_args()

loader = OnepieceImageDataModule(
    root_path="data/",
    batch_size=32,
    num_workers=11
)

model = ImageRecogModelV1(
    lrate = args.learning_rate,
    criterion = nn.CrossEntropyLoss(),
)

trainer = L.Trainer(default_root_dir="checkpoint/",
                    max_epochs=args.max_epochs,  
                    accelerator=args.accelerator)

# dict to save
class_dict = {
    0 : "Ace",
    1 : "Akainu",
    2 : "Brook",
    3 : "Chopper",
    4 : "Crocodile",
    5 : "Franky",
    6 : "Jinbei",
    7 : "Kurohige",
    8 : "Law",
    9 : "Luffy",
    10 : "Mihawk",
    11 : "Nami",
    12 : "Rayleigh",
    13 : "Robin",
    14 : "Sanji",
    15 : "Shanks",
    16 : "Usopp",
    17 : "Zoro"
}

if __name__ == "__main__":
    trainer.fit(model, loader)
    torch.save(class_dict, "checkpoint/manifest.pth", )
