import sys
sys.path.append("src")

from torch import nn
import lightning as L

from lightning_classify.data import OnepieceImageDataModule
from lightning_classify.models import ImageRecogModelV1

loader = OnepieceImageDataModule(
    root_path="data/",
    batch_size=32,
    num_workers=11
)

model = ImageRecogModelV1(
    lrate = 1e-3,
    criterion = nn.CrossEntropyLoss(),
)

trainer = L.Trainer(max_epochs=10, 
                    default_root_dir="checkpoint/", 
                    accelerator="cuda")

if __name__ == "__main__":
    trainer.fit(model, loader)
