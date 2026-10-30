import torch
from torch import nn


# Define the custom neural network
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        # Define layers of the neural network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2)             # dimezza altezza e larghezza
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # B x 256 x H x W -> B x 256 x 1 x 1

        self.fc1 = nn.Linear(256, 200)  # 200 is the number of classes in TinyImageNet

    def forward(self, x):
        # B x 3 x 224 x 224
        x = self.pool(self.conv1(x).relu())   # B x 64 x 112 x 112
        x = self.pool(self.conv2(x).relu())   # B x 128 x 56 x 56
        x = self.pool(self.conv3(x).relu())   # B x 256 x 28 x 28
        x = self.avgpool(x)                   # B x 256 x 1 x 1
        x = x.flatten(1)                      # B x 256
        x = self.fc1(x)                       # B x 200 (logits)

        return x
