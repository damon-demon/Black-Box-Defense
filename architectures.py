# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from archs.cifarnet import CifarNet
from archs.tiny_resnet import TinyResNet50, TinyResNet18
#from archs.cae import MNIST_CAE, CelebA_CAE, CIFAR_CAE, Cifar_Decoder_384, Cifar_Encoder_384, Cifar_Decoder_768_24, Cifar_Encoder_768_24, Cifar_Decoder_768_32, Cifar_Encoder_768_32, Cifar_Decoder_1536, Cifar_Encoder_1536, Cifar_Decoder_2048, Cifar_Encoder_2048, Cifar_Decoder_192, Cifar_Encoder_192, Cifar_Decoder_192_24, Cifar_Encoder_192_24, TinyImageNet_Decoder, TinyImageNet_Encoder, TinyImageNet_Decoder_768, TinyImageNet_Encoder_768, STL_Encoder, STL_Decoder, MNIST_Dim_Encoder, MNIST_Dim_Decoder, ImageNet_Encoder_1152, ImageNet_Decoder_1152, ImageNet_Encoder_1728, ImageNet_Decoder_1728
from archs.cae import *
from archs.dncnn import DnCNN
from archs.cifar_resnet import resnet as resnet_cifar
from archs.mnist_resnet import MnistResNet101 as resnet101_mnist
from archs.memnet import MemNet
from archs.wrn import WideResNet
from archs.stl_resnet import STL10_ResNet18
from datasets import get_normalize_layer
from torchvision.models.resnet import resnet18, resnet34, resnet50
from archs.resnet import ResNet50, ResNet18
#from archs.vrnet import VariationalNetwork

import torch
import torch.backends.cudnn as cudnn

IMAGENET_CLASSIFIERS = [
                        'resnet18', 
                        'resnet34', 
                        'resnet50',
                        'tiny_resnet50',
                        'tiny_resnet18', 'resnet50_restricted', 'resnet18_restricted'
                        ]

CIFAR10_CLASSIFIERS = [
                        'cifarnet', 'cifar_resnet110', 'mnist_resnet101', 'MNIST_CAE', 'MNIST_CAE_NoNorm', 'CelebA_CAE', 'CIFAR_CAE',
                        'cifar_wrn', 'cifar_wrn40', 'cifar_dncnn_recon',
                        'VGG16', 'VGG19', 'ResNet18','PreActResNet18','GoogLeNet',
                        'DenseNet121','ResNeXt29_2x64d','MobileNet','MobileNetV2',
                        'SENet18','ShuffleNetV2','EfficientNetB0'
                        'imagenet32_resnet110', 'imagenet32_wrn',
                        'stl10_resnet18', 'cifar_wrn_28_4'
                        ]

MRI_RECON = ['vrnet']

CLASSIFIERS_ARCHITECTURES = IMAGENET_CLASSIFIERS + CIFAR10_CLASSIFIERS + MRI_RECON

DENOISERS_ARCHITECTURES = ["cifar_dncnn", "mnist_dncnn", "cifar_dncnn_wide", "memnet", # cifar10 denoisers
                            'tiny_imagenet_dncnn', 'imagenet_dncnn', 'imagenet_memnet', # imagenet denoisers
                           'stl10_dncnn'
                        ]

AUTOENCODER_ARCHITECTURES = ['cifar_encoder_48','cifar_decoder_48',
                             'cifar_encoder_96','cifar_decoder_96',
                             'cifar_encoder_192','cifar_decoder_192',
                             'cifar_encoder_192_24','cifar_decoder_192_24',
                             'cifar_encoder_384','cifar_decoder_384',
                             'cifar_encoder_768_32','cifar_decoder_768_32',
                             'cifar_encoder_768_24', 'cifar_decoder_768_24',
                             'cifar_encoder_1536','cifar_decoder_1536',
                             'cifar_encoder_2048','cifar_decoder_2048',
                             'TinyImageNet_encoder', 'TinyImageNet_decoder',
                             'TinyImageNet_encoder_768', 'TinyImageNet_decoder_768',
                             'restricted_imagenet_encoder_1152', 'restricted_imagenet_decoder_1152',
                             'restricted_imagenet_encoder_1728', 'restricted_imagenet_decoder_1728',
                             'restricted_imagenet_encoder_2304', 'restricted_imagenet_decoder_2304',
                             'restricted_imagenet_encoder_3456', 'restricted_imagenet_decoder_3456',
                             'restricted_imagenet_encoder_15552', 'restricted_imagenet_decoder_15552',
                             'stl_encoder', 'stl_decoder',
                             'mnist_dim_encoder', 'mnist_dim_decoder'
                             ]


def get_architecture(arch: str, dataset: str, pytorch_pretrained: bool=False) -> torch.nn.Module:
    """ Return a neural network (with random weights)

    :param arch: the architecture - should be in the ARCHITECTURES list above
    :param dataset: the dataset - should be in the datasets.DATASETS list
    :return: a Pytorch module
    """
    ## ImageNet classifiers
    if arch == "resnet18" and dataset == "imagenet":
        model = torch.nn.DataParallel(resnet18(pretrained=pytorch_pretrained)).cuda()
        cudnn.benchmark = True
    elif arch == "resnet34" and dataset == "imagenet":
        model = torch.nn.DataParallel(resnet34(pretrained=pytorch_pretrained)).cuda()
        cudnn.benchmark = True
    elif arch == "resnet50" and dataset == "imagenet":
        model = torch.nn.DataParallel(resnet50(pretrained=pytorch_pretrained)).cuda()
        cudnn.benchmark = True
    elif arch == 'resnet50_restricted':
        model = ResNet50().cuda()
        cudnn.benchmark = True
        return model
    elif arch == 'resnet18_restricted':
        model = ResNet18().cuda()
        cudnn.benchmark = True
        return model

    elif arch == "tiny_resnet50" and dataset == "tinyimagenet":
        model = TinyResNet50().cuda()
        cudnn.benchmark = True

    elif arch == "tiny_resnet18" and dataset == "tinyimagenet":
        model = TinyResNet18().cuda()
        cudnn.benchmark = True

    ## Cifar classifiers
    elif arch == "cifarnet":
        model = CifarNet().cuda()
    elif arch == "cifar_resnet20":
        model = resnet_cifar(depth=20, num_classes=10).cuda()
    elif arch == "cifar_resnet110":
        model = resnet_cifar(depth=110, num_classes=10).cuda()
    elif arch == "stl10_resnet18":
        model = STL10_ResNet18(3, 10).cuda()
    elif arch == "mnist_resnet101":
        model = resnet101_mnist().cuda()
    elif arch == "imagenet32_resnet110":
        model = resnet_cifar(depth=110, num_classes=1000).cuda()
    elif arch == "imagenet32_wrn":
        model = WideResNet(depth=28, num_classes=1000, widen_factor=10).cuda()
    elif arch == 'cifar_wrn_28_4':
        model = WideResNet(depth=28, num_classes=10, widen_factor=4).cuda()

    # Cifar10 Models from https://github.com/kuangliu/pytorch-cifar
    # The 14 models we use in the paper as surrogate models
    elif arch == "cifar_wrn":
        model = WideResNet(depth=28, num_classes=10, widen_factor=10).cuda()
    elif arch == "cifar_wrn40":
        model = WideResNet(depth=40, num_classes=10, widen_factor=10).cuda()
    elif arch == "VGG16":
        model = VGG('VGG16').cuda()
    elif arch == "VGG19":
        model = VGG('VGG19').cuda()
    elif arch == "ResNet18":
        model = ResNet18().cuda()
    elif arch == "PreActResNet18":
        model = PreActResNet18().cuda()
    elif arch == "GoogLeNet":
        model = GoogLeNet().cuda()
    elif arch == "DenseNet121":
        model = DenseNet121().cuda()
    elif arch == "ResNeXt29_2x64d":
        model = ResNeXt29_2x64d().cuda()
    elif arch == "MobileNet":
        model = MobileNet().cuda()
    elif arch == "MobileNetV2":
        model = MobileNetV2().cuda()
    elif arch == "SENet18":
        model = SENet18().cuda()
    elif arch == "ShuffleNetV2":
        model = ShuffleNetV2(1).cuda()
    elif arch == "EfficientNetB0":
        model = EfficientNetB0().cuda()

    ## Image Reconstruction Network
    elif arch == "MNIST_CAE":
        model = MNIST_CAE().cuda()
    elif arch == "MNIST_CAE_NoNorm":
        model = MNIST_CAE().cuda()
        return model
    elif arch == "CelebA_CAE":
        model = CelebA_CAE().cuda()
    elif arch == "CIFAR_CAE":
        model = CIFAR_CAE().cuda()
    elif arch == "cifar_dncnn_recon":
        model = DnCNN(image_channels=3, depth=17, n_channels=64).cuda()

    # Encoder and Decoders
    elif arch =='restricted_imagenet_encoder_1152':
        model = ImageNet_Encoder_1152().cuda()
        return model
    elif arch =='restricted_imagenet_decoder_1152':
        model = ImageNet_Decoder_1152().cuda()
        return model
    elif arch =='restricted_imagenet_encoder_1728':
        model = ImageNet_Encoder_1728().cuda()
        return model
    elif arch =='restricted_imagenet_decoder_1728':
        model = ImageNet_Decoder_1728().cuda()
        return model
    elif arch =='restricted_imagenet_encoder_2304':
        model = ImageNet_Encoder_2304().cuda()
        return model
    elif arch =='restricted_imagenet_decoder_2304':
        model = ImageNet_Decoder_2304().cuda()
        return model
    elif arch =='restricted_imagenet_encoder_3456':
        model = ImageNet_Encoder_3456().cuda()
        return model
    elif arch =='restricted_imagenet_decoder_3456':
        model = ImageNet_Decoder_3456().cuda()
        return model
    elif arch =='restricted_imagenet_encoder_15552':
        model = ImageNet_Encoder_15552().cuda()
        return model
    elif arch =='restricted_imagenet_decoder_15552':
        model = ImageNet_Decoder_15552().cuda()
        return model

    elif arch == "TinyImageNet_encoder":
        model = TinyImageNet_Encoder().cuda()
        return model
    elif arch == "TinyImageNet_decoder":
        model = TinyImageNet_Decoder().cuda()
        return model
    elif arch == "TinyImageNet_encoder_768":
        model = TinyImageNet_Encoder_768().cuda()
        return model
    elif arch == "TinyImageNet_decoder_768":
        model = TinyImageNet_Decoder_768().cuda()
        return model

    elif arch == "mnist_dim_encoder":
        model = MNIST_Dim_Encoder().cuda()
        return model
    elif arch == "mnist_dim_decoder":
        model = MNIST_Dim_Decoder().cuda()
        return model

    elif arch == "stl_encoder":
        model = STL_Encoder().cuda()
        return model
    elif arch == "stl_decoder":
        model = STL_Decoder().cuda()
        return model

    elif arch == "cifar_encoder_48":
        model = Cifar_Encoder_48().cuda()
        return model
    elif arch == "cifar_decoder_48":
        model = Cifar_Decoder_48().cuda()
        return model

    elif arch == "cifar_encoder_96":
        model = Cifar_Encoder_96().cuda()
        return model
    elif arch == "cifar_decoder_96":
        model = Cifar_Decoder_96().cuda()
        return model

    elif arch == "cifar_encoder_192":
        model = Cifar_Encoder_192().cuda()
        return model
    elif arch == "cifar_decoder_192":
        model = Cifar_Decoder_192().cuda()
        return model

    elif arch == "cifar_encoder_192_24":
        model = Cifar_Encoder_192_24().cuda()
        return model
    elif arch == "cifar_decoder_192_24":
        model = Cifar_Decoder_192_24().cuda()
        return model

    elif arch == "cifar_encoder_384":
        model = Cifar_Encoder_384().cuda()
        return model
    elif arch == "cifar_decoder_384":
        model = Cifar_Decoder_384().cuda()
        return model

    elif arch == "cifar_encoder_768_32":
        model = Cifar_Encoder_768_32().cuda()
        return model
    elif arch == "cifar_decoder_768_32":
        model = Cifar_Decoder_768_32().cuda()
        return model

    elif arch == "cifar_encoder_768_24":
        model = Cifar_Encoder_768_24().cuda()
        return model
    elif arch == "cifar_decoder_768_24":
        model = Cifar_Decoder_768_24().cuda()
        return model

    elif arch == "cifar_encoder_1536":
        model = Cifar_Encoder_1536().cuda()
        return model
    elif arch == "cifar_decoder_1536":
        model = Cifar_Decoder_1536().cuda()
        return model

    elif arch == "cifar_encoder_2048":
        model = Cifar_Encoder_2048().cuda()
        return model
    elif arch == "cifar_decoder_2048":
        model = Cifar_Decoder_2048().cuda()
        return model
    #elif arch == "vrnet":
        #model = VariationalNetwork().cuda()


    ## Image Denoising Architectures
    elif arch == "cifar_dncnn":
        model = DnCNN(image_channels=3, depth=17, n_channels=64).cuda()
        #model = DnCNN(image_channels=3, depth=17, n_channels=64)
        return model
    elif arch == "mnist_dncnn":
        model = DnCNN(image_channels=1, depth=17, n_channels=64).cuda()
        #model = DnCNN(image_channels=3, depth=17, n_channels=64)
        return model
    elif arch == "cifar_dncnn_wide":
        model = DnCNN(image_channels=3, depth=17, n_channels=128).cuda()
        return model
    elif arch == 'memnet':
        model = MemNet(in_channels=3, channels=64, num_memblock=3, num_resblock=6).cuda()
        return model
    elif arch == "imagenet_dncnn":
        model = DnCNN(image_channels=3, depth=17, n_channels=64).cuda()
        cudnn.benchmark = True
        return model
    elif arch == "tiny_imagenet_dncnn":
        model = DnCNN(image_channels=3, depth=17, n_channels=64).cuda()
        cudnn.benchmark = True
        return model
    elif arch == "stl10_dncnn":
        model = DnCNN(image_channels=3, depth=17, n_channels=64).cuda()
        return model
    elif arch == 'imagenet_memnet':
        model = torch.nn.DataParallel(MemNet(in_channels=3, channels=64, num_memblock=3, num_resblock=6)).cuda()
        cudnn.benchmark = True
        return model
    else:
        raise Exception('Unknown architecture.')

    normalize_layer = get_normalize_layer(dataset)
    return torch.nn.Sequential(normalize_layer, model)
