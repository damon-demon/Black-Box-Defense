# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from architectures import DENOISERS_ARCHITECTURES, get_architecture, IMAGENET_CLASSIFIERS
from datasets import get_dataset, DATASETS
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
import argparse
import os
import torch


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', type=str, choices=DATASETS)
parser.add_argument('--arch', type=str, choices=DENOISERS_ARCHITECTURES)
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--batch', default=1, type=int, metavar='N',
                    help='batchsize (default: 256)')
parser.add_argument('--gpu', default=None, type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--noise_sd', default=0.0, type=float,
                    help="standard deviation of noise distribution for data augmentation")
parser.add_argument('--classifier', default='', type=str,
                    help='path to the classifier used with the `classificaiton`'
                         'or `stability` objectives of the denoiser.')
parser.add_argument('--pretrained-denoiser', default='', type=str,
                    help='path to a pretrained denoiser')



args = parser.parse_args()

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

toPilImage = ToPILImage()


def main():
    ## This is used to test the performance of the denoiser attached to a cifar10 classifier
    cifar10_test_loader = DataLoader(get_dataset('cifar10', 'test'), shuffle=False, batch_size=args.batch,
                                     num_workers=args.workers)

    # Denoiser Loading
    if args.pretrained_denoiser:
        checkpoint = torch.load(args.pretrained_denoiser)
        assert checkpoint['arch'] == args.arch
        denoiser = get_architecture(checkpoint['arch'], args.dataset)
        denoiser.load_state_dict(checkpoint['state_dict'])
    else:
        denoiser = get_architecture(args.arch, args.dataset)
    denoiser.eval()


    # Classifier Loading
    if args.classifier in IMAGENET_CLASSIFIERS:
        assert args.dataset == 'imagenet'
        # loading pretrained imagenet architectures
        clf = get_architecture(args.classifier, args.dataset, pytorch_pretrained=True)
    else:
        checkpoint = torch.load(args.classifier)
        clf = get_architecture(checkpoint['arch'], 'cifar10')
        clf.load_state_dict(checkpoint['state_dict'])
    clf.cuda().eval()

    num = visualize(cifar10_test_loader, denoiser, args.noise_sd, clf)

    print(num)
    print("Finished!")


def tensor_to_PIL(tensor):

    unloader = ToPILImage()
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image


def visualize(loader: DataLoader, denoiser: torch.nn.Module, noise_sd: float, classifier: torch.nn.Module):
    """
    A function to test the classification performance of a denoiser when attached to a given classifier
        :param loader:DataLoader: test dataloader
        :param denoiser:torch.nn.Module: the denoiser
        :param noise_sd:float: the std-dev of the Guassian noise perturbation of the input
        :param classifier:torch.nn.Module: the classifier to which the denoiser is attached
    """

    # switch to eval mode
    classifier.eval()
    denoiser.eval()

    k = 0

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            k = k + 1

            inputs = inputs.cuda()
            targets = targets.cuda()
            noise = torch.randn_like(inputs, device='cuda') * noise_sd

            # augment inputs with noise
            noisy_inputs = inputs + noise
            pre_original = classifier(noisy_inputs).argmax(1).detach().clone()

            recon = denoiser(noisy_inputs)
            pre_real = classifier(recon).argmax(1).detach().clone()

            if pre_original != targets and pre_real == targets and k > 1:
                break

        inputs = tensor_to_PIL(inputs)
        inputs.save("input.jpg")

        noise = tensor_to_PIL(noise)
        noise.save("noise.jpg")

        noisy_inputs = tensor_to_PIL(noisy_inputs)
        noisy_inputs.save("noisy_input.jpg")

        recon = tensor_to_PIL(recon)
        recon.save("recon.jpg")

        print("Original Prediction")
        print(pre_original)

        print("Denoised Prediction")
        print(pre_real)

    return k


if __name__ == "__main__":
    main()
