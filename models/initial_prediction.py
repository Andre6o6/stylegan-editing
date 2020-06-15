import torch
import torch.nn as nn
from torchvision.models import resnet50

class InitialLatentPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.ELU()

        # 3, 256, 256 ->
        self.resnet = list(resnet50(pretrained=True).children())[:-2]
        self.resnet = nn.Sequential(*self.resnet)
        # -> 2048, 8, 8
        self.conv2d = nn.Conv2d(2048, 256, kernel_size=1)
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(256*8*8, 512)
        self.dense2 = nn.Linear(512, (18 * 512))

    def forward(self, image):
        x = self.resnet(image)
        x = self.conv2d(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = x.view((-1, 18, 512))
        return x