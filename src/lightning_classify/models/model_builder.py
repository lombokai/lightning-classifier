from typing import Any

import torch
from torch import nn
from torchvision import models
from torch.optim import Adam
import lightning as L
from torchmetrics import Accuracy


class ImageRecogModel(L.LightningModule):
    def __init__(self, lrate: float, num_classes: int = 18) -> Any:
        super().__init__()
        self.num_classes = num_classes
        self.lrate = lrate

        self.model = self._load_backbone()
        self.criterion = nn.CrossEntropyLoss()
        self.metrics = Accuracy(task="multiclass", num_classes=self.num_classes)

    def _load_backbone(self) -> Any:
        mobile_net = models.mobilenet_v3_large(weights="DEFAULT")

        for param in mobile_net.parameters():
            param.requires_grad = False

        in_features = mobile_net.classifier[0].in_features
        mobile_net.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, out_features=self.num_classes)
        )

        return mobile_net

    def forward(self, x) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)

        y_class = torch.argmax(torch.softmax(logits, dim=1), dim=1)
        acc = self.metrics(y, y_class)

        self.log_dict({"train/loss": loss, "train/acc": acc}, prog_bar=True)

        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)

        y_class = torch.argmax(torch.softmax(logits, dim=1), dim=1)
        acc = self.metrics(y, y_class)

        self.log_dict({"val/loss": loss, "val/acc": acc}, prog_bar=True)
    
    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=self.lrate)
