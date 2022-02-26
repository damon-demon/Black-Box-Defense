import torch.nn as nn
import torchvision.models as models


class TinyResNet50(nn.Module):
    def __init__(self, in_channels=1):
        super(TinyResNet50, self).__init__()

        # Load a pretrained resnet model from torchvision.models in Pytorch
        self.model = models.resnet50(pretrained=True)

        self.model.avgpool = nn.AdaptiveAvgPool2d(1)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 200)

        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.model.maxpool = nn.Sequential()

    def forward(self, x):
        return self.model(x)


class TinyResNet18(nn.Module):
    def __init__(self, in_channels=1):
        super(TinyResNet18, self).__init__()

        # Load a pretrained resnet model from torchvision.models
        self.model = models.resnet18(pretrained=True)

        self.model.avgpool = nn.AdaptiveAvgPool2d(1)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 200)

        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.model.maxpool = nn.Sequential()

    def forward(self, x):
        return self.model(x)