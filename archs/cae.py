
from torch import nn
from torchvision.datasets import MNIST
import os

# img_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5), (0.5))
# ])
#
# dataset = MNIST('./data', transform=img_transform)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class MNIST_CAE(nn.Module):
    def __init__(self):
        super(MNIST_CAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=1),  # b, 16, 28, 28
            nn.ReLU(True),

            nn.Conv2d(16, 32, 3, stride=3, padding=1), # b, 32, 10, 10
            nn.ReLU(True),

            nn.Conv2d(32, 16, 3, stride=3, padding=1), # b, 16, 4, 4
            nn.ReLU(True),

            nn.Conv2d(16, 8, 3, stride=3, padding=1),  # b, 8, 2, 2
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class MNIST_Dim_Encoder(nn.Module):
    def __init__(self):
        super(MNIST_Dim_Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 3, 4, stride=2, padding=1),      #(3, 14, 14)
            nn.ReLU(),
            nn.Conv2d(3, 12, 5, stride=3, padding=0),     #(12, 4, 4)
            nn.ReLU(),
        )
    def forward(self, x):
        encoded = self.encoder(x)    # output size  [batch, 192, 1, 1]
        return encoded


class MNIST_Dim_Decoder(nn.Module):
    def __init__(self):
        super(MNIST_Dim_Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(12, 3, 5, stride=3, padding=0),   #(3, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(3, 1, 4, stride=2, padding=1),   #(1, 28, 28)
            nn.Sigmoid(),
        )
    def forward(self, x):
        decoded = self.decoder(x)
        return decoded


class CelebA_CAE(nn.Module):
    def __init__(self):
        super(CelebA_CAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class CIFAR_CAE(nn.Module):
    def __init__(self):
        super(CIFAR_CAE, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=4, padding=0),            # [batch, 32, 16, 16]
            nn.ReLU(),
            nn.Conv2d(32, 48, 3, stride=3, padding=2),           # [batch, 48, 8, 8]
            nn.ReLU(),
			# nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            # nn.ReLU(),
 			nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
             nn.ReLU(),
        )
        self.decoder = nn.Sequential(
             nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
             nn.ReLU(),
			#  nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            # nn.ReLU(),
			nn.ConvTranspose2d(48, 32, 3, stride=3 , padding=2),  # [batch, 32, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=4, padding=0),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class STL_Encoder(nn.Module):
    def __init__(self):
        super(STL_Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 18, 5, stride=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(18, 72, 4, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(72, 144, 3, stride=3, padding=2),           # [batch, 24, 8, 8]
            nn.ReLU(),
			# nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            # nn.ReLU(),
 			nn.Conv2d(144, 288, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
             nn.ReLU(),
            nn.Conv2d(288, 576, 2, stride=2, padding=0),  # [batch, 576, 1, 1]
            nn.ReLU(),
        )
    def forward(self, x):
        encoded = self.encoder(x)    # output size  [batch, 192, 1, 1]
        return encoded


class STL_Decoder(nn.Module):
    def __init__(self):
        super(STL_Decoder, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(576, 288, 2, stride=2, padding=0),  # [batch, 96, 2, 2]
            nn.ReLU(),
             nn.ConvTranspose2d(288, 144, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
             nn.ReLU(),
			# nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            # nn.ReLU(),
			nn.ConvTranspose2d(144, 72, 3, stride=3 , padding=2),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(72, 18, 4, stride=4, padding=0),   # [batch, 3, 32, 32]
            nn.ReLU(),
            nn.ConvTranspose2d(18, 3, 5, stride=3, padding=1),  # [batch, 12, 16, 16]
            nn.Sigmoid(),
        )
    def forward(self, x):
        decoded = self.decoder(x)
        return decoded


class ImageNet_Encoder_15552(nn.Module):
    def __init__(self):
        super(ImageNet_Encoder_15552, self).__init__()
        # Input size: [batch, 3, 32, 32]  3072
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 72, 8, stride=6, padding=0),            # [batch, 18, 55, 55]
            nn.ReLU(),
            nn.Conv2d(72, 144, 3, stride=2, padding=0),           # [batch, 72, 13, 13]
            nn.ReLU(),
            nn.Conv2d(144, 432, 3, stride=3, padding=0),           # [batch, 144, 6, 6]
            nn.ReLU(),
            nn.Conv2d(432, 864, 2, stride=2, padding=1),          # [batch, 288, 3, 3]
            nn.ReLU(),
            nn.Conv2d(864, 1728, 2, stride=2, padding=1),         # [batch, 864, 1, 1]
            nn.ReLU()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded


class ImageNet_Decoder_15552(nn.Module):
    def __init__(self):
        super(ImageNet_Decoder_15552, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1728, 864, 2, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(864, 432, 2, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(432, 144, 3, stride=3, padding=0),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(144, 72, 3, stride=2, padding=0),  # [batch, 48, 4, 4]
            nn.ReLU(),
			# nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            # nn.ReLU(),
			#nn.ConvTranspose2d(48, 32, 3, stride=3 , padding=2),  # [batch, 12, 16, 16]
            #nn.ReLU(),
            nn.ConvTranspose2d(72, 3, 8, stride=6, padding=0),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        decoded = self.decoder(x)
        return decoded




class ImageNet_Encoder_1152(nn.Module):
    def __init__(self):
        super(ImageNet_Encoder_1152, self).__init__()
        # Input size: [batch, 3, 32, 32]  3072
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 24, 8, stride=6, padding=0),            # [batch, 18, 55, 55]
            nn.ReLU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=0),           # [batch, 72, 13, 13]
            nn.ReLU(),
            nn.Conv2d(48, 96, 3, stride=3, padding=0),           # [batch, 144, 6, 6]
            nn.ReLU(),
            nn.Conv2d(96, 192, 2, stride=2, padding=1),          # [batch, 288, 3, 3]
            nn.ReLU(),
            nn.Conv2d(192, 384, 2, stride=2, padding=1),         # [batch, 864, 1, 1]
            nn.ReLU(),
            nn.Conv2d(384, 1152, 3, stride=3, padding=0),        # [batch, 864, 1, 1]
            nn.ReLU(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded


class ImageNet_Decoder_1152(nn.Module):
    def __init__(self):
        super(ImageNet_Decoder_1152, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1152, 384, 3, stride=3, padding=0),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(384, 192, 2, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(192, 96, 2, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(96,48, 3, stride=3, padding=0),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(48, 24, 3, stride=2, padding=0),  # [batch, 48, 4, 4]
            nn.ReLU(),
			# nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            # nn.ReLU(),
			#nn.ConvTranspose2d(48, 32, 3, stride=3 , padding=2),  # [batch, 12, 16, 16]
            #nn.ReLU(),
            nn.ConvTranspose2d(24, 3, 8, stride=6, padding=0),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        decoded = self.decoder(x)
        return decoded


class ImageNet_Encoder_1728(nn.Module):
    def __init__(self):
        super(ImageNet_Encoder_1728, self).__init__()
        # Input size: [batch, 3, 32, 32]  3072
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 36, 8, stride=6, padding=0),            # [batch, 18, 55, 55]
            nn.ReLU(),
            nn.Conv2d(36, 72, 3, stride=2, padding=0),           # [batch, 72, 13, 13]
            nn.ReLU(),
            nn.Conv2d(72, 144, 3, stride=3, padding=0),           # [batch, 144, 6, 6]
            nn.ReLU(),
            nn.Conv2d(144, 288, 2, stride=2, padding=1),          # [batch, 288, 3, 3]
            nn.ReLU(),
            nn.Conv2d(288, 576, 2, stride=2, padding=1),         # [batch, 864, 1, 1]
            nn.ReLU(),
            nn.Conv2d(576, 1728, 3, stride=3, padding=0),        # [batch, 864, 1, 1]
            nn.ReLU(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded


class ImageNet_Decoder_1728(nn.Module):
    def __init__(self):
        super(ImageNet_Decoder_1728, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1728, 576, 3, stride=3, padding=0),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(576, 288, 2, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(288, 144, 2, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(144,72, 3, stride=3, padding=0),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(72, 36, 3, stride=2, padding=0),  # [batch, 48, 4, 4]
            nn.ReLU(),
			# nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            # nn.ReLU(),
			#nn.ConvTranspose2d(48, 32, 3, stride=3 , padding=2),  # [batch, 12, 16, 16]
            #nn.ReLU(),
            nn.ConvTranspose2d(36, 3, 8, stride=6, padding=0),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        decoded = self.decoder(x)
        return decoded


class ImageNet_Encoder_2304(nn.Module):
    def __init__(self):
        super(ImageNet_Encoder_2304, self).__init__()
        # Input size: [batch, 3, 32, 32]  3072
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 48, 8, stride=6, padding=0),            # [batch, 18, 55, 55]
            nn.ReLU(),
            nn.Conv2d(48, 96, 3, stride=2, padding=0),           # [batch, 72, 13, 13]
            nn.ReLU(),
            nn.Conv2d(96, 192, 3, stride=3, padding=0),           # [batch, 144, 6, 6]
            nn.ReLU(),
            nn.Conv2d(192, 384, 2, stride=2, padding=1),          # [batch, 288, 3, 3]
            nn.ReLU(),
            nn.Conv2d(384, 768, 2, stride=2, padding=1),         # [batch, 864, 1, 1]
            nn.ReLU(),
            nn.Conv2d(768, 2304, 3, stride=3, padding=0),        # [batch, 864, 1, 1]
            nn.ReLU(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded


class ImageNet_Decoder_2304(nn.Module):
    def __init__(self):
        super(ImageNet_Decoder_2304, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2304, 768, 3, stride=3, padding=0),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(768, 384, 2, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(384, 192, 2, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(192,96, 3, stride=3, padding=0),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(96, 48, 3, stride=2, padding=0),  # [batch, 48, 4, 4]
            nn.ReLU(),
			# nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            # nn.ReLU(),
			#nn.ConvTranspose2d(48, 32, 3, stride=3 , padding=2),  # [batch, 12, 16, 16]
            #nn.ReLU(),
            nn.ConvTranspose2d(48, 3, 8, stride=6, padding=0),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        decoded = self.decoder(x)
        return decoded


class ImageNet_Encoder_3456(nn.Module):
    def __init__(self):
        super(ImageNet_Encoder_3456, self).__init__()
        # Input size: [batch, 3, 32, 32]  3072
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 48, 8, stride=6, padding=0),            # [batch, 18, 55, 55]
            nn.ReLU(),
            nn.Conv2d(48, 96, 3, stride=2, padding=0),           # [batch, 72, 13, 13]
            nn.ReLU(),
            nn.Conv2d(96, 288, 3, stride=3, padding=0),           # [batch, 144, 6, 6]
            nn.ReLU(),
            nn.Conv2d(288, 576, 2, stride=2, padding=1),          # [batch, 288, 3, 3]
            nn.ReLU(),
            nn.Conv2d(576, 1152, 2, stride=2, padding=1),         # [batch, 864, 1, 1]
            nn.ReLU(),
            nn.Conv2d(1152, 3456, 3, stride=3, padding=0),        # [batch, 864, 1, 1]
            nn.ReLU(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded


class ImageNet_Decoder_3456(nn.Module):
    def __init__(self):
        super(ImageNet_Decoder_3456, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(3456, 1152, 3, stride=3, padding=0),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(1152, 576, 2, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(576, 288, 2, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(288,96, 3, stride=3, padding=0),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(96, 48, 3, stride=2, padding=0),  # [batch, 48, 4, 4]
            nn.ReLU(),
			# nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            # nn.ReLU(),
			#nn.ConvTranspose2d(48, 32, 3, stride=3 , padding=2),  # [batch, 12, 16, 16]
            #nn.ReLU(),
            nn.ConvTranspose2d(48, 3, 8, stride=6, padding=0),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        decoded = self.decoder(x)
        return decoded


class TinyImageNet_Encoder(nn.Module):
    def __init__(self):
        super(TinyImageNet_Encoder, self).__init__()
        # Input size: [batch, 3, 64, 64]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, 2, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(6, 24, 4, stride=4, padding=0),            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(24, 48, 3, stride=3, padding=2),           # [batch, 24, 8, 8]
            nn.ReLU(),
			# nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            # nn.ReLU(),
 			nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
             nn.ReLU(),

        )
    def forward(self, x):
        encoded = self.encoder(x)    # output size  [batch, 192, 1, 1]
        return encoded


class TinyImageNet_Decoder(nn.Module):
    def __init__(self):
        super(TinyImageNet_Decoder, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.decoder = nn.Sequential(
             nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
             nn.ReLU(),
			# nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            # nn.ReLU(),
			nn.ConvTranspose2d(48, 24, 3, stride=3 , padding=2),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(24, 6, 4, stride=4, padding=0),   # [batch, 3, 32, 32]
            nn.ReLU(),
            nn.ConvTranspose2d(6, 3, 2, stride=2, padding= 0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        decoded = self.decoder(x)
        return decoded

class TinyImageNet_Encoder_768(nn.Module):
    def __init__(self):
        super(TinyImageNet_Encoder_768, self).__init__()
        # Input size: [batch, 3, 64, 64]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 24, 4, stride=4, padding=0), # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.Conv2d(24, 48, 2, stride=2, padding=0),  # [batch, 48, 4, 4]
            nn.ReLU(),
            # nn.Conv2d(24, 48, 4, stride=2, padding=1),
            # nn.ReLU(),
            nn.Conv2d(48, 96, 4, stride=2, padding=1),  # [batch, 96, 2, 2]
            nn.ReLU(),
            nn.Conv2d(96, 192, 2, stride=2, padding=0),  # [batch, 192, 1, 1]
            nn.ReLU(),

        )
    def forward(self, x):
        encoded = self.encoder(x)    # output size  [batch, 192, 1, 1]
        return encoded


class TinyImageNet_Decoder_768(nn.Module):
    def __init__(self):
        super(TinyImageNet_Decoder_768, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(192, 96, 2, stride=2, padding=0),  # [batch, 96, 2, 2]
            nn.ReLU(),
            nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
            # nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            # nn.ReLU(),
            nn.ConvTranspose2d(48, 24, 2, stride=2, padding=0),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(24, 3, 4, stride=4, padding=0),  # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        decoded = self.decoder(x)
        return decoded



class Cifar_Encoder_48(nn.Module):
    def __init__(self):
        super(Cifar_Encoder_48, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, 4, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(6, 12, 3, stride=3, padding=2),           # [batch, 24, 8, 8]
            nn.ReLU(),
			# nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            # nn.ReLU(),
 			nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
             nn.ReLU(),
            nn.Conv2d(24, 48, 2, stride=2, padding=0),  # [batch, 192, 1, 1]
            nn.ReLU(),
        )
    def forward(self, x):
        encoded = self.encoder(x)    # output size  [batch, 192, 1, 1]
        return encoded


class Cifar_Decoder_48(nn.Module):
    def __init__(self):
        super(Cifar_Decoder_48, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(48, 24, 2, stride=2, padding=0),  # [batch, 96, 2, 2]
            nn.ReLU(),
             nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
             nn.ReLU(),
			# nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            # nn.ReLU(),
			nn.ConvTranspose2d(12, 6, 3, stride=3 , padding=2),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(6, 3, 4, stride=4, padding=0),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )
    def forward(self, x):
        decoded = self.decoder(x)
        return decoded




class Cifar_Encoder_96(nn.Module):
    def __init__(self):
        super(Cifar_Encoder_96, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(12, 24, 3, stride=3, padding=2),           # [batch, 24, 8, 8]
            nn.ReLU(),
			# nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            # nn.ReLU(),
 			nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
             nn.ReLU(),
            nn.Conv2d(48, 96, 2, stride=2, padding=0),  # [batch, 192, 1, 1]
            nn.ReLU(),
        )
    def forward(self, x):
        encoded = self.encoder(x)    # output size  [batch, 192, 1, 1]
        return encoded


class Cifar_Decoder_96(nn.Module):
    def __init__(self):
        super(Cifar_Decoder_96, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(96, 48, 2, stride=2, padding=0),  # [batch, 96, 2, 2]
            nn.ReLU(),
             nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
             nn.ReLU(),
			# nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            # nn.ReLU(),
			nn.ConvTranspose2d(24, 12, 3, stride=3 , padding=2),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=4, padding=0),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )
    def forward(self, x):
        decoded = self.decoder(x)
        return decoded




class Cifar_Encoder_192_24(nn.Module):
    def __init__(self):
        super(Cifar_Encoder_192_24, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 24, 4, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(24, 48, 3, stride=3, padding=2),           # [batch, 24, 8, 8]
            nn.ReLU(),
			# nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            # nn.ReLU(),
 			nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
             nn.ReLU(),
            nn.Conv2d(96, 192, 2, stride=2, padding=0),  # [batch, 192, 1, 1]
            nn.ReLU(),
        )
    def forward(self, x):
        encoded = self.encoder(x)    # output size  [batch, 192, 1, 1]
        return encoded


class Cifar_Decoder_192_24(nn.Module):
    def __init__(self):
        super(Cifar_Decoder_192_24, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(192, 96, 2, stride=2, padding=0),  # [batch, 96, 2, 2]
            nn.ReLU(),
             nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
             nn.ReLU(),
			# nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            # nn.ReLU(),
			nn.ConvTranspose2d(48, 24, 3, stride=3 , padding=2),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(24, 3, 4, stride=4, padding=0),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        decoded = self.decoder(x)
        return decoded




class Cifar_Encoder_192(nn.Module):
    def __init__(self):
        super(Cifar_Encoder_192, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=4, padding=0),            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(32, 48, 3, stride=3, padding=2),           # [batch, 24, 8, 8]
            nn.ReLU(),
			# nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            # nn.ReLU(),
 			nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
             nn.ReLU(),
            nn.Conv2d(96, 192, 2, stride=2, padding=0),  # [batch, 192, 1, 1]
            nn.ReLU(),
        )
    def forward(self, x):
        encoded = self.encoder(x)    # output size  [batch, 192, 1, 1]
        return encoded


class Cifar_Decoder_192(nn.Module):
    def __init__(self):
        super(Cifar_Decoder_192, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(192, 96, 2, stride=2, padding=0),  # [batch, 96, 2, 2]
            nn.ReLU(),
             nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
             nn.ReLU(),
			# nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            # nn.ReLU(),
			nn.ConvTranspose2d(48, 32, 3, stride=3 , padding=2),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=4, padding=0),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        decoded = self.decoder(x)
        return decoded



class Cifar_Encoder_384(nn.Module):
    def __init__(self):
        super(Cifar_Encoder_384, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 24, 4, stride=4, padding=0),            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(24, 48, 3, stride=3, padding=2),           # [batch, 24, 8, 8]
            nn.ReLU(),
			# nn.Conv2d(24, 48, 4, stride=2, padding=1),          # [batch, 48, 4, 4]
            # nn.ReLU(),
 			nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
             nn.ReLU(),
        )
    def forward(self, x):
        encoded = self.encoder(x)    # output size  [batch, 96, 2, 2]
        return encoded


class Cifar_Decoder_384(nn.Module):
    def __init__(self):
        super(Cifar_Decoder_384, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.decoder = nn.Sequential(
             nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
             nn.ReLU(),
			# nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            # nn.ReLU(),
			nn.ConvTranspose2d(48, 24, 3, stride=3 , padding=2),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(24, 3, 4, stride=4, padding=0),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        decoded = self.decoder(x)
        return decoded


class Cifar_Encoder_768_32(nn.Module):
    def __init__(self):
        super(Cifar_Encoder_768_32, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=4, padding=0),            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(32, 48, 3, stride=3, padding=2),           # [batch, 24, 8, 8]
            nn.ReLU(),
			#nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            #nn.ReLU(),
 			#nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
             #nn.ReLU(),
        )
    def forward(self, x):
        encoded = self.encoder(x)          # Output Size: [batch, 48, 4, 4]
        return encoded


class Cifar_Decoder_768_32(nn.Module):
    def __init__(self):
        super(Cifar_Decoder_768_32, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.decoder = nn.Sequential(
             #nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
             #nn.ReLU(),
			#nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            #nn.ReLU(),
			nn.ConvTranspose2d(48, 32, 3, stride=3 , padding=2),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=4, padding=0),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        decoded = self.decoder(x)
        return decoded


class Cifar_Encoder_768_24(nn.Module):
    def __init__(self):
        super(Cifar_Encoder_768_24, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 24, 4, stride=4, padding=0),            # [batch, 12, 16, 16]
            nn.ReLU(),
            #nn.Conv2d(32, 48, 3, stride=3, padding=2),           # [batch, 24, 8, 8]
            #nn.ReLU(),
			nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            nn.ReLU(),
 			#nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
             #nn.ReLU(),
        )
    def forward(self, x):
        encoded = self.encoder(x)          # Output Size: [batch, 48, 4, 4]
        return encoded


class Cifar_Decoder_768_24(nn.Module):
    def __init__(self):
        super(Cifar_Decoder_768_24, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.decoder = nn.Sequential(
             #nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
             #nn.ReLU(),
			nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
			#nn.ConvTranspose2d(48, 32, 3, stride=3 , padding=2),  # [batch, 12, 16, 16]
            #nn.ReLU(),
            nn.ConvTranspose2d(24, 3, 4, stride=4, padding=0),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        decoded = self.decoder(x)
        return decoded


class Cifar_Encoder_1536(nn.Module):
    def __init__(self):
        super(Cifar_Encoder_1536, self).__init__()
        # Input size: [batch, 3, 32, 32]  3072
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 24, 4, stride=4, padding=0),            # [batch, 12, 16, 16]
            nn.ReLU(),
            #nn.Conv2d(32, 48, 3, stride=3, padding=2),           # [batch, 24, 8, 8]
            #nn.ReLU(),
			# nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            # nn.ReLU(),
 			#nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
             #nn.ReLU(),
        )
    def forward(self, x):
        encoded = self.encoder(x)
        return encoded


class Cifar_Decoder_1536(nn.Module):
    def __init__(self):
        super(Cifar_Decoder_1536, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.decoder = nn.Sequential(
             #nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
             #nn.ReLU(),
			# nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            # nn.ReLU(),
			#nn.ConvTranspose2d(48, 32, 3, stride=3 , padding=2),  # [batch, 12, 16, 16]
            #nn.ReLU(),
            nn.ConvTranspose2d(24, 3, 4, stride=4, padding=0),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        decoded = self.decoder(x)
        return decoded


class Cifar_Encoder_2048(nn.Module):
    def __init__(self):
        super(Cifar_Encoder_2048, self).__init__()
        # Input size: [batch, 3, 32, 32]  3072
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=4, padding=0),            # [batch, 12, 16, 16]
            nn.ReLU(),
            #nn.Conv2d(32, 48, 3, stride=3, padding=2),           # [batch, 24, 8, 8]
            #nn.ReLU(),
			# nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            # nn.ReLU(),
 			#nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
             #nn.ReLU(),
        )
    def forward(self, x):
        encoded = self.encoder(x)
        return encoded


class Cifar_Decoder_2048(nn.Module):
    def __init__(self):
        super(Cifar_Decoder_2048, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.decoder = nn.Sequential(
             #nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
             #nn.ReLU(),
			# nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            # nn.ReLU(),
			#nn.ConvTranspose2d(48, 32, 3, stride=3 , padding=2),  # [batch, 12, 16, 16]
            #nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=4, padding=0),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        decoded = self.decoder(x)
        return decoded
