from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class CIFAR10AlexNet(nn.Module):
    """AlexNet-style CNN adapted for 32x32 CIFAR-10 images."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self._initialize_weights()
        num_flat_features = self._get_num_flat_features()
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_flat_features, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        flattened = features.view(features.size(0), -1)
        return self.classifier(flattened)

    def _get_num_flat_features(self) -> int:
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 32, 32)
            output_features = self.features(dummy_input)
        return output_features.view(1, -1).size(1)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)


def build_resnet18(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """Build a CIFAR-10 ResNet-18, optionally fine-tuning ImageNet weights."""

    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    nn.init.xavier_normal_(model.fc.weight)
    nn.init.constant_(model.fc.bias, 0)
    return model


def build_model(name: str, pretrained_resnet: bool = True, num_classes: int = 10) -> nn.Module:
    normalized = name.lower().replace("-", "").replace("_", "")
    if normalized in {"alexnet", "alexnetstyle"}:
        return CIFAR10AlexNet(num_classes=num_classes)
    if normalized in {"resnet18", "resnet"}:
        return build_resnet18(num_classes=num_classes, pretrained=pretrained_resnet)
    raise ValueError(f"Unknown model: {name}")


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
