import torch.nn as nn
import torch.nn.functional as F


class CifarNet(nn.Module):
    def __init__(self):
        super(CifarNet, self).__init__()
        self.conv2d = nn.Conv2d(3, 64, 3, 1)
        self.conv2d_1 = nn.Conv2d(64, 64, 3, 1)
        self.conv2d_2 = nn.Conv2d(64, 128, 3, 1)
        self.conv2d_3 = nn.Conv2d(128, 128, 3, 1)

        self.dense = nn.Linear(3200, 1024)
        self.dense_1 = nn.Linear(1024, 256)
        self.dense_2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv2d(x))
        x = F.relu(self.conv2d_1(x))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv2d_2(x))
        x = F.relu(self.conv2d_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.contiguous().view(-1, 3200)

        x = F.relu(self.dense(x))
        x = F.dropout(x, p=0.5)
        x = F.relu(self.dense_1(x))
        x = F.dropout(x, p=0.5)
        x = self.dense_2(x)
   #     x = F.log_softmax(x, dim=1)
        return x
