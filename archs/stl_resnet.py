#Code from https://jovian.ai/venkatesh-vran/stl10-resnet

import torch.nn as nn

class SimpleResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        return self.relu2(out) + x  # ReLU can be applied before or after adding the input


def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class STL10_ResNet18(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.conv1_1 = conv_block(in_channels, 64, pool=True)
        self.conv1_2 = conv_block(64, 64, pool=False)

        self.conv2_1 = conv_block(64, 128, pool=True)  # output: 128 x 24 x 24
        self.conv2_2 = conv_block(128, 128, pool=False)  # output: 128 x 24 x 24


        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.conv3_1 = conv_block(128, 256, pool=True)  # output: 256 x 12 x 12
        self.conv3_2 = conv_block(256, 256, pool=False)  # output: 256 x 12 x 12

        self.conv4_1 = conv_block(256, 512, pool=True)  # output: 512 x 6 x 6
        self.conv4_2 = conv_block(512, 512, pool=False)  # output: 512 x 6 x 6

        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

        self.classifier = nn.Sequential(nn.MaxPool2d(6),
                                        nn.Flatten(),
                                        nn.Dropout(0.2),
                                        nn.Linear(512, num_classes))

    def forward(self, xb):
        out = self.conv1_1(xb)
        out = self.conv1_2(out)

        out = self.conv2_1(out)
        out = self.conv2_2(out)

        out = self.res1(out) + out

        out = self.conv3_1(out)
        out = self.conv3_2(out)

        out = self.conv4_1(out)
        out = self.conv4_2(out)

        out = self.res2(out) + out
        out = self.classifier(out)
        return out
