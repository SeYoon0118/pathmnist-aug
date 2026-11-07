"""PathMNIST 9-class CNN model ."""

import torch.nn as nn


class PathMNISTCNN(nn.Module):
    """28x28 RGB images -> 9 class logits."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 9),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
