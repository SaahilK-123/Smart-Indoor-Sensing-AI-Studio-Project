# model_rd.py
import torch
import torch.nn as nn
import torchvision.models as models


class RDResNet18(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        # no pretraining
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Linear(512, n_classes)

    def forward(self, x):
        # x: [B,1,T,R] -> [B,3,T,R]
        x = x.repeat(1, 3, 1, 1)
        return self.backbone(x)
