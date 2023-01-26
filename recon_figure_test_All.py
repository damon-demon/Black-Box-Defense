# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# File for training denoisers with at most one classifier attached to

from architectures import DENOISERS_ARCHITECTURES, get_architecture, IMAGENET_CLASSIFIERS, CLASSIFIERS_ARCHITECTURES, AUTOENCODER_ARCHITECTURES
from datasets import get_dataset, DATASETS
from test_denoiser_recon import test, recon_visual
from torch.nn import MSELoss, CrossEntropyLoss, L1Loss
import torch.nn as nn
from torch.optim import SGD, Optimizer, Adam
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from train_utils import AverageMeter, accuracy, init_logfile, log, copy_code, requires_grad_, measurement

import argparse
from datetime import datetime
import numpy as np
import os
import time
import torch
import torchvision
import itertools
from torchvision.utils import save_image
from recon_attacks import Attacker, recon_PGD_L2
import pytorch_ssim
#from skimage.metrics import structural_similarity as ssim

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', type=str, default='mnist', choices=DATASETS)
parser.add_argument('--outdir', type=str, default='MNIST_Recon_Visual_SSIM', help='folder to save denoiser and training log)')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--batch', default=1, type=int, metavar='N',
                    help='batchsize (default: 256)')
parser.add_argument('--gpu', default=None, type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--noise_sd', default=0.25, type=float,
                    help="standard deviation of noise distribution for data augmentation")
parser.add_argument('--test_time', default=5, type=int, metavar='N',
                    help='number of total epochs to run')

# Fixed
parser.add_argument('--classifier', default='MNIST_ReconNet_NoNorm/best.pth.tar', type=str,
                    help='path to the classifier used with the `classificaiton`'
                         'or `stability` objectives of the denoiser.')
parser.add_argument('--pretrained-decoder', default='MNIST_AE_Dim_Norm/decoder.pth.tar', type=str,
                    help='path to a pretrained decoder')

# FO-DS
parser.add_argument('--denoiser_DS_FO', default='DS_FO_lr-3_Adam200_MNIST_recon_NoNorm_step50_epsilon0.75/denoiser.pth.tar', type=str,
                    help='path to a pretrained denoiser')
# ZO-DS
parser.add_argument('--denoiser_DS_ZO', default='DS_ZO_lr-3_Adam200_MNIST_recon_Norm_step50_epsilon0.75/denoiser.pth.tar', type=str,
                    help='path to a pretrained denoiser')

#FO-AE-DS
parser.add_argument('--denoiser_AE_DS_FO', default='AE_DS_FO_lr-3_Adam200_MNIST_recon_Norm_step50_epsilon0.75/denoiser.pth.tar', type=str,
                    help='path to a pretrained denoiser')
parser.add_argument('--encoder_AE_DS_FO', default='AE_DS_FO_lr-3_Adam200_MNIST_recon_Norm_step50_epsilon0.75/encoder.pth.tar', type=str,
                    help='path to a pretrained encoder')

#ZO-AE-DS
parser.add_argument('--denoiser_AE_DS_ZO', default='AE_DS_ZO_lr-3_Adam200_MNIST_recon_Norm_step50_epsilon0.75/denoiser.pth.tar', type=str,
                    help='path to a pretrained denoiser')
parser.add_argument('--encoder_AE_DS_ZO', default='AE_DS_ZO_lr-3_Adam200_MNIST_recon_Norm_step50_epsilon0.75/encoder.pth.tar', type=str,
                    help='path to a pretrained encoder')


parser.add_argument('--arch', type=str, default='mnist_dncnn', choices = DENOISERS_ARCHITECTURES)
parser.add_argument('--clf_arch', type=str, default= 'MNIST_CAE_NoNorm', choices=CLASSIFIERS_ARCHITECTURES)
parser.add_argument('--encoder_arch', type=str, default='mnist_dim_encoder', choices=AUTOENCODER_ARCHITECTURES)
parser.add_argument('--decoder_arch', type=str, default='mnist_dim_decoder', choices=AUTOENCODER_ARCHITECTURES)


parser.add_argument('--noise_num', default=10, type=int,
                    help='number of noise for smoothing')
# Parameters for adv examples generation
parser.add_argument('--num_steps', default=40, type=int,
                    help='Number of steps for attack')


args = parser.parse_args()

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

toPilImage = ToPILImage()


def main():
    initial_ponit = 0
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    test_dataset = get_dataset(args.dataset, 'test')

    #Dataloaders
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch,
                             num_workers=args.workers)

    #Fixed
    if args.classifier:
        checkpoint = torch.load(args.classifier)
        clf = get_architecture(args.clf_arch, args.dataset)
        clf.load_state_dict(checkpoint['state_dict'])
        clf.cuda().eval()

    if args.pretrained_decoder:
        checkpoint = torch.load(args.pretrained_decoder)
        assert checkpoint['arch'] == args.decoder_arch
        decoder = get_architecture(checkpoint['arch'], args.dataset)
        decoder.load_state_dict(checkpoint['state_dict'])

    #FO-DS
    checkpoint = torch.load(args.denoiser_DS_FO)
    assert checkpoint['arch'] == args.arch
    denoiser_fo_ds = get_architecture(checkpoint['arch'], args.dataset)
    denoiser_fo_ds.load_state_dict(checkpoint['state_dict'])

    #ZO-DS
    checkpoint = torch.load(args.denoiser_DS_ZO)
    assert checkpoint['arch'] == args.arch
    denoiser_zo_ds = get_architecture(checkpoint['arch'], args.dataset)
    denoiser_zo_ds.load_state_dict(checkpoint['state_dict'])


    #FO-AE-DS
    checkpoint = torch.load(args.denoiser_AE_DS_FO)
    assert checkpoint['arch'] == args.arch
    denoiser_fo_ae_ds = get_architecture(checkpoint['arch'], args.dataset)
    denoiser_fo_ae_ds.load_state_dict(checkpoint['state_dict'])

    checkpoint = torch.load(args.encoder_AE_DS_FO)
    assert checkpoint['arch'] == args.encoder_arch
    encoder_fo_ae_ds = get_architecture(checkpoint['arch'], args.dataset)
    encoder_fo_ae_ds.load_state_dict(checkpoint['state_dict'])


    # ZO-AE-DS
    checkpoint = torch.load(args.denoiser_AE_DS_ZO)
    assert checkpoint['arch'] == args.arch
    denoiser_zo_ae_ds = get_architecture(checkpoint['arch'], args.dataset)
    denoiser_zo_ae_ds.load_state_dict(checkpoint['state_dict'])

    checkpoint = torch.load(args.encoder_AE_DS_ZO)
    assert checkpoint['arch'] == args.encoder_arch
    encoder_zo_ae_ds = get_architecture(checkpoint['arch'], args.dataset)
    encoder_zo_ae_ds.load_state_dict(checkpoint['state_dict'])


    logfilename = os.path.join(args.outdir, 'log.txt')
    init_logfile(logfilename,
                 "epsilon\tRecon\tRecon_Fx\tFO_DS\tFO_DS_Fx\tZO_DS\tZO_DS_Fx\tFO_AE_DS\tFO_AE_DS_Fx\tZO_AE_DS\tZO_AE_DS_Fx")

    for i in range(0, args.test_time):
        epsilon = i * 1.0
        recon, recon_fx, recon_ssim, fo_ds, fo_ds_fx, fo_ds_ssim, zo_ds, zo_ds_fx, zo_ds_ssim, fo_ae_ds, fo_ae_ds_fx, fo_ae_ds_ssim, zo_ae_ds, zo_ae_ds_fx, zo_ae_ds_ssim= test_with_recon_ae(test_loader, clf, decoder, denoiser_fo_ds, denoiser_zo_ds, denoiser_fo_ae_ds, encoder_fo_ae_ds, denoiser_zo_ae_ds, encoder_zo_ae_ds, epsilon)


        log(logfilename, "{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}".format(epsilon,recon, recon_fx, recon_ssim, fo_ds, fo_ds_fx, fo_ds_ssim, zo_ds, zo_ds_fx, zo_ds_ssim, fo_ae_ds, fo_ae_ds_fx, fo_ae_ds_ssim, zo_ae_ds, zo_ae_ds_fx, zo_ae_ds_ssim))


def test_with_recon_ae(loader: DataLoader, recon_net: torch.nn.Module, decoder: torch.nn.Module, de_fo_ds: torch.nn.Module, de_zo_ds: torch.nn.Module, de_fo_ae_ds: torch.nn.Module, enc_fo_ae_ds: torch.nn.Module, de_zo_ae_ds: torch.nn.Module, enc_zo_ae_ds: torch.nn.Module, epsilon: float):
    """
    A function to test the classification performance of a denoiser when attached to a given classifier
        :param loader:DataLoader: test dataloader
        :param denoiser:torch.nn.Module: the denoiser
        :param criterion: the loss function (e.g. MAE)
        :param noise_sd:float: the std-dev of the Guassian noise perturbation of the input
        :param print_freq:int: the frequency of logging
        :param recon_net:torch.nn.Module: the reconstruction network to which the denoiser is attached
    """
    outdir = args.outdir
    noise_sd = args.noise_sd
    noise_num = args.noise_num
    num_steps = args.num_steps

    criterion = RMSELoss().cuda()

    # switch to eval mode
    recon_net.eval()
    decoder.eval()

    de_fo_ds.eval()
    de_zo_ds.eval()

    de_fo_ae_ds.eval()
    enc_fo_ae_ds.eval()

    de_zo_ae_ds.eval()
    enc_zo_ae_ds.eval()

    n_measurement = 576
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    recon_losses = AverageMeter()
    fo_ds_losses = AverageMeter()
    zo_ds_losses = AverageMeter()
    fo_ae_ds_losses = AverageMeter()
    zo_ae_ds_losses = AverageMeter()

    recon_losses_fx = AverageMeter()
    fo_ds_losses_fx = AverageMeter()
    zo_ds_losses_fx = AverageMeter()
    fo_ae_ds_losses_fx = AverageMeter()
    zo_ae_ds_losses_fx = AverageMeter()

    recon_ssims = AverageMeter()
    fo_ds_ssims = AverageMeter()
    zo_ds_ssims = AverageMeter()
    fo_ae_ds_ssims = AverageMeter()
    zo_ae_ds_ssims = AverageMeter()

    mm_min = torch.tensor(-2.5090184, dtype=torch.long).cuda()
    mm_max = torch.tensor(3.3369503, dtype=torch.long).cuda()
    mm_dis = mm_max - mm_min

    mark = 39

    attacker = recon_PGD_L2(steps=num_steps, device='cuda', max_norm=epsilon)  # remember epsilon/256


    for i, (img_original, _) in enumerate(loader):
        img_original = img_original.cuda()

        # Obtain the Shape of Inputs (Batch_size x Channel x H x W)
        batch_size = img_original.size()[0]
        channel = img_original.size()[1]
        h = img_original.size()[2]
        w = img_original.size()[3]

        # Flatten the Reconstructed Images for Gradient Estimation
        d = channel * h * w

        img_original = img_original.cuda()  # input x (batch,  channel, h, w)
        img = img_original.view(batch_size, d).cuda()

        if i == 0:
            a = measurement(n_measurement, d)

        img = torch.mm(img, a)
        img = torch.mm(img, a.t())
        img = img.view(batch_size, channel, h, w)
        img = img.float()
        img = (img - mm_min) / mm_dis

        if epsilon == 0:
            with torch.no_grad():
                # ----------Clean Reconstruction Acc [Recon]------------
                fx = recon_net(img)
                recon_loss = criterion(fx, img_original)
                recon_loss_fx = criterion(fx, fx)
                recon_ssim = pytorch_ssim.ssim(fx, img_original)

                recon_losses.update(recon_loss.item(), img.size(0))
                recon_losses_fx.update(recon_loss_fx.item(), img.size(0))
                recon_ssims.update(recon_ssim.item(), img.size(0))

                if i == mark:
                    pic = to_img(fx.cpu().data)
                    save_image(pic,
                               os.path.join(outdir,
                                            'Epsilon_{epsilon}_ReconOnly_Loss_{loss:.3f}_LossFx_{loss_fx:.3f}_SSIM_{ssim:.3f}.png').format(
                                   epsilon=epsilon,
                                   loss=recon_losses.avg, loss_fx=recon_losses_fx.avg, ssim = recon_ssims.avg))

                # ----------Clean Reconstruction Acc [FO-DS]------------
                outputs = img.repeat((1, noise_num, 1, 1)).view(-1, channel, h, w)
                outputs = outputs + torch.randn_like(outputs, device='cuda') * noise_sd
                outputs = de_fo_ds(outputs)
                outputs = recon_net(outputs)
                outputs = outputs.view(-1, batch_size, noise_num, channel, h, w).mean(2, keepdim=True).reshape(
                    batch_size,
                    channel, h,
                    w)
                fo_ds_loss = criterion(outputs, img_original)
                fo_ds_loss_fx = criterion(outputs, fx)
                fo_ds_ssim = pytorch_ssim.ssim(outputs, img_original)

                fo_ds_losses.update(fo_ds_loss.item(), img.size(0))
                fo_ds_losses_fx.update(fo_ds_loss_fx.item(), img.size(0))
                fo_ds_ssims.update(fo_ds_ssim.item(), img.size(0))
                if i == mark:
                    pic = to_img(outputs.cpu().data)
                    save_image(pic, os.path.join(outdir,
                                                 'Epsilon_{epsilon}_FO_DS_Loss_{loss:.3f}_LossFx_{loss_fx:.3f}_SSIM_{ssim:.3f}.png').format(
                        epsilon=epsilon,
                        loss=fo_ds_losses.avg,
                        loss_fx=fo_ds_losses_fx.avg, ssim = fo_ds_ssims.avg))

                # ----------Clean Reconstruction Acc [ZO-DS]------------
                outputs = img.repeat((1, noise_num, 1, 1)).view(-1, channel, h, w)
                outputs = outputs + torch.randn_like(outputs, device='cuda') * noise_sd
                outputs = de_zo_ds(outputs)
                outputs = recon_net(outputs)
                outputs = outputs.view(-1, batch_size, noise_num, channel, h, w).mean(2, keepdim=True).reshape(
                    batch_size,
                    channel, h,
                    w)

                zo_ds_loss = criterion(outputs, img_original)
                zo_ds_loss_fx = criterion(outputs, fx)
                zo_ds_ssim = pytorch_ssim.ssim(outputs, img_original)

                zo_ds_losses.update(zo_ds_loss.item(), img.size(0))
                zo_ds_losses_fx.update(zo_ds_loss_fx.item(), img.size(0))
                zo_ds_ssims.update(zo_ds_ssim.item(), img.size(0))

                if i == mark:
                    pic = to_img(outputs.cpu().data)
                    save_image(pic, os.path.join(outdir,
                                                 'Epsilon_{epsilon}_ZO_DS_Loss_{loss:.3f}_LossFx_{loss_fx:.3f}_SSIM_{ssim:.3f}.png').format(
                        epsilon=epsilon,
                        loss=zo_ds_losses.avg,
                        loss_fx=zo_ds_losses_fx.avg, ssim = zo_ds_ssims.avg))

                # ------------- clean Reconstruction Acc  [FO-AE-DS] --------------
                outputs = img.repeat((1, noise_num, 1, 1)).view(-1, channel, h, w)
                outputs = outputs + torch.randn_like(outputs, device='cuda') * noise_sd
                outputs = enc_fo_ae_ds(de_fo_ae_ds(outputs))
                outputs = recon_net(decoder(outputs))
                outputs = outputs.view(-1, batch_size, noise_num, channel, h, w).mean(2, keepdim=True).reshape(
                    batch_size,
                    channel, h,
                    w)

                fo_ae_ds_loss = criterion(outputs, img_original)
                fo_ae_ds_loss_fx = criterion(outputs, fx)
                fo_ae_ds_ssim = pytorch_ssim.ssim(outputs, img_original)

                fo_ae_ds_losses.update(fo_ae_ds_loss.item(), img.size(0))
                fo_ae_ds_losses_fx.update(fo_ae_ds_loss_fx.item(), img.size(0))
                fo_ae_ds_ssims.update(fo_ae_ds_ssim.item(), img.size(0))

                if i == mark:
                    pic = to_img(outputs.cpu().data)
                    save_image(pic,
                               os.path.join(outdir,
                                            'Epsilon_{epsilon}_FO_AE_DS_Loss_{loss:.3f}_LossFx_{loss_fx:.3f}_SSIM_{ssim:.3f}.png').format(
                                   epsilon=epsilon,
                                   loss=fo_ae_ds_losses.avg, loss_fx=fo_ae_ds_losses_fx.avg, ssim = fo_ae_ds_ssims.avg))

                # ------------- clean Reconstruction Acc  [ZO-AE-DS] --------------
                outputs = img.repeat((1, noise_num, 1, 1)).view(-1, channel, h, w)
                outputs = outputs + torch.randn_like(outputs, device='cuda') * noise_sd
                outputs = enc_zo_ae_ds(de_zo_ae_ds(outputs))
                outputs = recon_net(decoder(outputs))
                outputs = outputs.view(-1, batch_size, noise_num, channel, h, w).mean(2, keepdim=True).reshape(
                    batch_size,
                    channel, h,
                    w)
                zo_ae_ds_loss = criterion(outputs, img_original)
                zo_ae_ds_loss_fx = criterion(outputs, fx)
                zo_ae_ds_ssim = pytorch_ssim.ssim(outputs, img_original)

                zo_ae_ds_losses.update(zo_ae_ds_loss.item(), img.size(0))
                zo_ae_ds_losses_fx.update(zo_ae_ds_loss_fx.item(), img.size(0))
                zo_ae_ds_ssims.update(zo_ae_ds_ssim.item(), img.size(0))

                if i == mark:
                    pic = to_img(outputs.cpu().data)
                    save_image(pic,
                               os.path.join(outdir,
                                            'Epsilon_{epsilon}_ZO_AE_DS_Loss_{loss:.3f}_LossFx_{loss_fx:.3f}_SSIM_{ssim:.3f}.png').format(
                                   epsilon=epsilon,
                                   loss=zo_ae_ds_losses.avg, loss_fx=zo_ae_ds_losses_fx.avg, ssim = zo_ae_ds_ssims.avg))

        else:
            adv_img = attacker.attack(recon_net, img, img_original, criterion)

            with torch.no_grad():
                # ----------Clean Reconstruction Acc [Recon]  --> Get Fx------------
                fx = recon_net(img)

                # ----------Clean Reconstruction Acc [Recon] ------------
                outputs = recon_net(adv_img)

                recon_loss = criterion(outputs, img_original)
                recon_loss_fx = criterion(outputs, fx)
                recon_ssim = pytorch_ssim.ssim(outputs, img_original)

                recon_losses.update(recon_loss.item(), img.size(0))
                recon_losses_fx.update(recon_loss_fx.item(), img.size(0))
                recon_ssims.update(recon_ssim.item(), img.size(0))

                if i == mark:
                    pic = to_img(outputs.cpu().data)
                    save_image(pic,
                               os.path.join(outdir,
                                            'Epsilon_{epsilon}_ReconOnly_Loss_{loss:.3f}_LossFx_{loss_fx:.3f}_SSIM_{ssim:.3f}.png').format(
                                   epsilon=epsilon,
                                   loss=recon_losses.avg, loss_fx=recon_losses_fx.avg, ssim = zo_ae_ds_ssims.avg))

                # ----------Clean Reconstruction Acc [FO-DS]------------
                outputs = adv_img.repeat((1, noise_num, 1, 1)).view(-1, channel, h, w)
                outputs = outputs + torch.randn_like(outputs, device='cuda') * noise_sd
                outputs = de_fo_ds(outputs)
                outputs = recon_net(outputs)
                outputs = outputs.view(-1, batch_size, noise_num, channel, h, w).mean(2, keepdim=True).reshape(
                    batch_size,
                    channel, h,
                    w)
                fo_ds_loss = criterion(outputs, img_original)
                fo_ds_loss_fx = criterion(outputs, fx)
                fo_ds_ssim = pytorch_ssim.ssim(outputs, img_original)

                fo_ds_losses.update(fo_ds_loss.item(), img.size(0))
                fo_ds_losses_fx.update(fo_ds_loss_fx.item(), img.size(0))
                fo_ds_ssims.update(fo_ds_ssim.item(), img.size(0))
                if i == mark:
                    pic = to_img(outputs.cpu().data)
                    save_image(pic, os.path.join(outdir,
                                                 'Epsilon_{epsilon}_FO_DS_Loss_{loss:.3f}_LossFx_{loss_fx:.3f}_SSIM_{ssim:.3f}.png').format(
                        epsilon=epsilon,
                        loss=fo_ds_losses.avg,
                        loss_fx=fo_ds_losses_fx.avg, ssim = fo_ds_ssims.avg))

                # ----------Clean Reconstruction Acc [ZO-DS]------------
                outputs = adv_img.repeat((1, noise_num, 1, 1)).view(-1, channel, h, w)
                outputs = outputs + torch.randn_like(outputs, device='cuda') * noise_sd
                outputs = de_zo_ds(outputs)
                outputs = recon_net(outputs)
                outputs = outputs.view(-1, batch_size, noise_num, channel, h, w).mean(2, keepdim=True).reshape(
                    batch_size,
                    channel, h,
                    w)
                zo_ds_loss = criterion(outputs, img_original)
                zo_ds_loss_fx = criterion(outputs, fx)
                zo_ds_ssim = pytorch_ssim.ssim(outputs, img_original)

                zo_ds_losses.update(zo_ds_loss.item(), img.size(0))
                zo_ds_losses_fx.update(zo_ds_loss_fx.item(), img.size(0))
                zo_ds_ssims.update(zo_ds_ssim.item(), img.size(0))
                if i == mark:
                    pic = to_img(outputs.cpu().data)
                    save_image(pic, os.path.join(outdir,
                                                 'Epsilon_{epsilon}_FO_DS_Loss_{loss:.3f}_LossFx_{loss_fx:.3f}_SSIM_{ssim:.3f}.png').format(
                        epsilon=epsilon,
                        loss=zo_ds_losses.avg,
                        loss_fx=zo_ds_losses_fx.avg, ssim = zo_ds_ssims.avg))

                # ------------- clean Reconstruction Acc  [FO-AE-DS] --------------
                outputs = adv_img.repeat((1, noise_num, 1, 1)).view(-1, channel, h, w)
                outputs = outputs + torch.randn_like(outputs, device='cuda') * noise_sd
                outputs = enc_fo_ae_ds(de_fo_ae_ds(outputs))
                outputs = recon_net(decoder(outputs))
                outputs = outputs.view(-1, batch_size, noise_num, channel, h, w).mean(2, keepdim=True).reshape(
                    batch_size,
                    channel, h,
                    w)
                fo_ae_ds_loss = criterion(outputs, img_original)
                fo_ae_ds_loss_fx = criterion(outputs, fx)
                fo_ae_ds_ssim = pytorch_ssim.ssim(outputs, img_original)

                fo_ae_ds_losses.update(fo_ae_ds_loss.item(), img.size(0))
                fo_ae_ds_losses_fx.update(fo_ae_ds_loss_fx.item(), img.size(0))
                fo_ae_ds_ssims.update(fo_ae_ds_ssim.item(), img.size(0))

                if i == mark:
                    pic = to_img(outputs.cpu().data)
                    save_image(pic,
                               os.path.join(outdir,
                                            'Epsilon_{epsilon}_FO_AE_DS_Loss_{loss:.3f}_LossFx_{loss_fx:.3f}_SSIM_{ssim:.3f}.png').format(
                                   epsilon=epsilon,
                                   loss=fo_ae_ds_losses.avg, loss_fx=fo_ae_ds_losses_fx.avg, ssim = fo_ae_ds_ssims.avg))

                # ------------- clean Reconstruction Acc  [ZO-AE-DS] --------------
                outputs = adv_img.repeat((1, noise_num, 1, 1)).view(-1, channel, h, w)
                outputs = outputs + torch.randn_like(outputs, device='cuda') * noise_sd
                outputs = enc_zo_ae_ds(de_zo_ae_ds(outputs))
                outputs = recon_net(decoder(outputs))
                outputs = outputs.view(-1, batch_size, noise_num, channel, h, w).mean(2, keepdim=True).reshape(
                    batch_size,
                    channel, h,
                    w)
                zo_ae_ds_loss = criterion(outputs, img_original)
                zo_ae_ds_loss_fx = criterion(outputs, fx)
                zo_ae_ds_ssim = pytorch_ssim.ssim(outputs, img_original)

                zo_ae_ds_losses.update(zo_ae_ds_loss.item(), img.size(0))
                zo_ae_ds_losses_fx.update(zo_ae_ds_loss_fx.item(), img.size(0))
                zo_ae_ds_ssims.update(zo_ae_ds_ssim.item(), img.size(0))

                if i == mark:
                    pic = to_img(outputs.cpu().data)
                    save_image(pic,
                               os.path.join(outdir,
                                            'Epsilon_{epsilon}_ZO_AE_DS_Loss_{loss:.3f}_LossFx_{loss_fx:.3f}_SSIM_{ssim:.3f}.png').format(
                                   epsilon=epsilon,
                                   loss=zo_ae_ds_losses.avg, loss_fx=zo_ae_ds_losses_fx.avg, ssim = zo_ae_ds_ssims.avg))



        log = 'Test: [{0}/{1}]\t'' \
        ''Recon {recon_loss.avg:.4f} ({recon_loss_fx.avg:.4f})\t'' \
        ''FO-DS {fo_ds_loss.avg:.4f} ({fo_ds_loss_fx.avg:.4f})\t'' \
        ''ZO-DS {zo_ds_loss.avg:.4f} ({zo_ds_loss_fx.avg:.4f})\t'' \
        ''FO-AE-DS {fo_ae_ds_loss.avg:.4f} ({fo_ae_ds_loss_fx.avg:.4f})\t'' \
        ''ZO-AE-DS {zo_ae_ds_loss.avg:.4f} ({zo_ae_ds_loss_fx.avg:.4f})\n'.format(
            i, len(loader), recon_loss =recon_losses, recon_loss_fx =recon_losses_fx, fo_ds_loss =fo_ds_losses, fo_ds_loss_fx = fo_ds_losses_fx, zo_ds_loss=zo_ds_losses, zo_ds_loss_fx = zo_ds_losses_fx, fo_ae_ds_loss = fo_ae_ds_losses, fo_ae_ds_loss_fx= fo_ae_ds_losses_fx, zo_ae_ds_loss= zo_ae_ds_losses, zo_ae_ds_loss_fx=zo_ae_ds_losses_fx)

        print(log)

    return recon_losses.avg, recon_losses_fx.avg, recon_ssims.avg, fo_ds_losses.avg, fo_ds_losses_fx.avg, fo_ds_ssims.avg, zo_ds_losses.avg, zo_ds_losses_fx.avg, zo_ds_ssims.avg, fo_ae_ds_losses.avg, fo_ae_ds_losses_fx.avg, fo_ae_ds_ssims.avg, zo_ae_ds_losses.avg, zo_ae_ds_losses_fx.avg, zo_ae_ds_ssims.avg
    # Return Empericial Robustness Recon acc (MSE)

def test_with_recon_ae_visual(loader: DataLoader, recon_net: torch.nn.Module, decoder: torch.nn.Module, de_fo_ds: torch.nn.Module, de_zo_ds: torch.nn.Module, de_fo_ae_ds: torch.nn.Module, enc_fo_ae_ds: torch.nn.Module, de_zo_ae_ds: torch.nn.Module, enc_zo_ae_ds: torch.nn.Module, epsilon: float):
    """
    A function to test the classification performance of a denoiser when attached to a given classifier
        :param loader:DataLoader: test dataloader
        :param denoiser:torch.nn.Module: the denoiser
        :param criterion: the loss function (e.g. MAE)
        :param noise_sd:float: the std-dev of the Guassian noise perturbation of the input
        :param print_freq:int: the frequency of logging
        :param recon_net:torch.nn.Module: the reconstruction network to which the denoiser is attached
    """
    outdir = args.outdir
    noise_sd = args.noise_sd
    noise_num = args.noise_num
    num_steps = args.num_steps

    criterion = RMSELoss().cuda()

    # switch to eval mode
    recon_net.eval()
    decoder.eval()

    de_fo_ds.eval()
    de_zo_ds.eval()

    de_fo_ae_ds.eval()
    enc_fo_ae_ds.eval()

    de_zo_ae_ds.eval()
    enc_zo_ae_ds.eval()

    n_measurement = 576
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    recon_losses = AverageMeter()
    fo_ds_losses = AverageMeter()
    zo_ds_losses = AverageMeter()
    fo_ae_ds_losses = AverageMeter()
    zo_ae_ds_losses = AverageMeter()

    recon_losses_fx = AverageMeter()
    fo_ds_losses_fx = AverageMeter()
    zo_ds_losses_fx = AverageMeter()
    fo_ae_ds_losses_fx = AverageMeter()
    zo_ae_ds_losses_fx = AverageMeter()

    mm_min = torch.tensor(-2.5090184, dtype=torch.long).cuda()
    mm_max = torch.tensor(3.3369503, dtype=torch.long).cuda()
    mm_dis = mm_max - mm_min

    mark = 39

    attacker = recon_PGD_L2(steps=num_steps, device='cuda', max_norm=epsilon)  # remember epsilon/256


    for i, (img_original, _) in enumerate(loader):
        img_original = img_original.cuda()

        # Obtain the Shape of Inputs (Batch_size x Channel x H x W)
        batch_size = img_original.size()[0]
        channel = img_original.size()[1]
        h = img_original.size()[2]
        w = img_original.size()[3]

        # Flatten the Reconstructed Images for Gradient Estimation
        d = channel * h * w

        img_original = img_original.cuda()  # input x (batch,  channel, h, w)
        img = img_original.view(batch_size, d).cuda()
        # adv_img = attacker.attack(recon_net, img.view(batch_size, channel, h, w), img_original, criterion)

        if i == 0:
            a = measurement(n_measurement, d)

        img = torch.mm(img, a)
        img = torch.mm(img, a.t())
        img = img.view(batch_size, channel, h, w)
        img = img.float()
        img = (img - mm_min) / mm_dis

        if epsilon == 0:
            with torch.no_grad():
                # ----------Clean Reconstruction Acc [Recon]------------
                fx = recon_net(img)
                recon_loss = criterion(fx, img_original)
                recon_loss_fx = criterion(fx, fx)
                recon_losses.update(recon_loss.item(), img.size(0))
                recon_losses_fx.update(recon_loss_fx.item(), img.size(0))
                if i == mark:
                    pic = to_img(fx.cpu().data)
                    save_image(pic,
                               os.path.join(outdir,
                                            'Epsilon_{epsilon}_ReconOnly_Loss_{loss:.3f}_LossFx_{loss_fx:.3f}.png').format(
                                   epsilon=epsilon,
                                   loss=recon_losses.avg, loss_fx=recon_losses_fx.avg))

                # ----------Clean Reconstruction Acc [FO-DS]------------
                outputs = img.repeat((1, noise_num, 1, 1)).view(-1, channel, h, w)
                outputs = outputs + torch.randn_like(outputs, device='cuda') * noise_sd
                outputs = de_fo_ds(outputs)
                outputs = recon_net(outputs)
                outputs = outputs.view(-1, batch_size, noise_num, channel, h, w).mean(2, keepdim=True).reshape(
                    batch_size,
                    channel, h,
                    w)
                fo_ds_loss = criterion(outputs, img_original)
                fo_ds_loss_fx = criterion(outputs, fx)
                fo_ds_losses.update(fo_ds_loss.item(), img.size(0))
                fo_ds_losses_fx.update(fo_ds_loss_fx.item(), img.size(0))
                if i == mark:
                    pic = to_img(outputs.cpu().data)
                    save_image(pic, os.path.join(outdir,
                                                 'Epsilon_{epsilon}_FO_DS_Loss_{loss:.3f}_LossFx_{loss_fx:.3f}.png').format(
                        epsilon=epsilon,
                        loss=fo_ds_losses.avg,
                        loss_fx=fo_ds_losses_fx.avg))

                # ----------Clean Reconstruction Acc [ZO-DS]------------
                outputs = img.repeat((1, noise_num, 1, 1)).view(-1, channel, h, w)
                outputs = outputs + torch.randn_like(outputs, device='cuda') * noise_sd
                outputs = de_zo_ds(outputs)
                outputs = recon_net(outputs)
                outputs = outputs.view(-1, batch_size, noise_num, channel, h, w).mean(2, keepdim=True).reshape(
                    batch_size,
                    channel, h,
                    w)
                zo_ds_loss = criterion(outputs, img_original)
                zo_ds_loss_fx = criterion(outputs, fx)
                zo_ds_losses.update(zo_ds_loss.item(), img.size(0))
                zo_ds_losses_fx.update(zo_ds_loss_fx.item(), img.size(0))
                if i == mark:
                    pic = to_img(outputs.cpu().data)
                    save_image(pic, os.path.join(outdir,
                                                 'Epsilon_{epsilon}_ZO_DS_Loss_{loss:.3f}_LossFx_{loss_fx:.3f}.png').format(
                        epsilon=epsilon,
                        loss=zo_ds_losses.avg,
                        loss_fx=zo_ds_losses_fx.avg))

                # ------------- clean Reconstruction Acc  [FO-AE-DS] --------------
                outputs = img.repeat((1, noise_num, 1, 1)).view(-1, channel, h, w)
                outputs = outputs + torch.randn_like(outputs, device='cuda') * noise_sd
                outputs = enc_fo_ae_ds(de_fo_ae_ds(outputs))
                outputs = recon_net(decoder(outputs))
                outputs = outputs.view(-1, batch_size, noise_num, channel, h, w).mean(2, keepdim=True).reshape(
                    batch_size,
                    channel, h,
                    w)
                fo_ae_ds_loss = criterion(outputs, img_original)
                fo_ae_ds_loss_fx = criterion(outputs, fx)
                fo_ae_ds_losses.update(fo_ae_ds_loss.item(), img.size(0))
                fo_ae_ds_losses_fx.update(fo_ae_ds_loss_fx.item(), img.size(0))

                if i == mark:
                    pic = to_img(outputs.cpu().data)
                    save_image(pic,
                               os.path.join(outdir,
                                            'Epsilon_{epsilon}_FO_AE_DS_Loss_{loss:.3f}_LossFx_{loss_fx:.3f}.png').format(
                                   epsilon=epsilon,
                                   loss=fo_ae_ds_losses.avg, loss_fx=fo_ae_ds_losses_fx.avg))

                # ------------- clean Reconstruction Acc  [ZO-AE-DS] --------------
                outputs = img.repeat((1, noise_num, 1, 1)).view(-1, channel, h, w)
                outputs = outputs + torch.randn_like(outputs, device='cuda') * noise_sd
                outputs = enc_zo_ae_ds(de_zo_ae_ds(outputs))
                outputs = recon_net(decoder(outputs))
                outputs = outputs.view(-1, batch_size, noise_num, channel, h, w).mean(2, keepdim=True).reshape(
                    batch_size,
                    channel, h,
                    w)
                zo_ae_ds_loss = criterion(outputs, img_original)
                zo_ae_ds_loss_fx = criterion(outputs, fx)
                zo_ae_ds_losses.update(zo_ae_ds_loss.item(), img.size(0))
                zo_ae_ds_losses_fx.update(zo_ae_ds_loss_fx.item(), img.size(0))

                if i == mark:
                    pic = to_img(outputs.cpu().data)
                    save_image(pic,
                               os.path.join(outdir,
                                            'Epsilon_{epsilon}_ZO_AE_DS_Loss_{loss:.3f}_LossFx_{loss_fx:.3f}.png').format(
                                   epsilon=epsilon,
                                   loss=zo_ae_ds_losses.avg, loss_fx=zo_ae_ds_losses_fx.avg))

        else:
            adv_img = attacker.attack(recon_net, img, img_original, criterion)

            with torch.no_grad():
                if i == mark:
                    pic = to_img(img_original.cpu().data)
                    save_image(pic,
                               os.path.join(outdir,
                                            'original.png'))

                if i == mark:
                    # ----------Clean Reconstruction Acc [Recon]  --> Get Fx------------
                    fx = recon_net(img)

                    # ----------Clean Reconstruction Acc [Recon] ------------
                    outputs = recon_net(adv_img)
                    recon_loss = criterion(outputs, img_original)
                    recon_loss_fx = criterion(outputs, fx)
                    recon_losses.update(recon_loss.item(), img.size(0))
                    recon_losses_fx.update(recon_loss_fx.item(), img.size(0))

                    if i == mark:
                        pic = to_img(outputs.cpu().data)
                        save_image(pic,
                                   os.path.join(outdir,
                                                'Epsilon_{epsilon}_ReconOnly_Loss_{loss:.3f}_LossFx_{loss_fx:.3f}.png').format(
                                       epsilon=epsilon,
                                       loss=recon_losses.avg, loss_fx=recon_losses_fx.avg))

                    # ----------Clean Reconstruction Acc [FO-DS]------------
                    outputs = adv_img.repeat((1, noise_num, 1, 1)).view(-1, channel, h, w)
                    outputs = outputs + torch.randn_like(outputs, device='cuda') * noise_sd
                    outputs = de_fo_ds(outputs)
                    outputs = recon_net(outputs)
                    outputs = outputs.view(-1, batch_size, noise_num, channel, h, w).mean(2, keepdim=True).reshape(
                        batch_size,
                        channel, h,
                        w)
                    fo_ds_loss = criterion(outputs, img_original)
                    fo_ds_loss_fx = criterion(outputs, fx)
                    fo_ds_losses.update(fo_ds_loss.item(), img.size(0))
                    fo_ds_losses_fx.update(fo_ds_loss_fx.item(), img.size(0))
                    if i == mark:
                        pic = to_img(outputs.cpu().data)
                        save_image(pic, os.path.join(outdir,
                                                     'Epsilon_{epsilon}_FO_DS_Loss_{loss:.3f}_LossFx_{loss_fx:.3f}.png').format(
                            epsilon=epsilon,
                            loss=fo_ds_losses.avg,
                            loss_fx=fo_ds_losses_fx.avg))

                    # ----------Clean Reconstruction Acc [ZO-DS]------------
                    outputs = adv_img.repeat((1, noise_num, 1, 1)).view(-1, channel, h, w)
                    outputs = outputs + torch.randn_like(outputs, device='cuda') * noise_sd
                    outputs = de_zo_ds(outputs)
                    outputs = recon_net(outputs)
                    outputs = outputs.view(-1, batch_size, noise_num, channel, h, w).mean(2, keepdim=True).reshape(
                        batch_size,
                        channel, h,
                        w)
                    zo_ds_loss = criterion(outputs, img_original)
                    zo_ds_loss_fx = criterion(outputs, fx)
                    zo_ds_losses.update(zo_ds_loss.item(), img.size(0))
                    zo_ds_losses_fx.update(zo_ds_loss_fx.item(), img.size(0))
                    if i == mark:
                        pic = to_img(outputs.cpu().data)
                        save_image(pic, os.path.join(outdir,
                                                     'Epsilon_{epsilon}_FO_DS_Loss_{loss:.3f}_LossFx_{loss_fx:.3f}.png').format(
                            epsilon=epsilon,
                            loss=zo_ds_losses.avg,
                            loss_fx=zo_ds_losses_fx.avg))

                    # ------------- clean Reconstruction Acc  [FO-AE-DS] --------------
                    outputs = adv_img.repeat((1, noise_num, 1, 1)).view(-1, channel, h, w)
                    outputs = outputs + torch.randn_like(outputs, device='cuda') * noise_sd
                    outputs = enc_fo_ae_ds(de_fo_ae_ds(outputs))
                    outputs = recon_net(decoder(outputs))
                    outputs = outputs.view(-1, batch_size, noise_num, channel, h, w).mean(2, keepdim=True).reshape(
                        batch_size,
                        channel, h,
                        w)
                    fo_ae_ds_loss = criterion(outputs, img_original)
                    fo_ae_ds_loss_fx = criterion(outputs, fx)
                    fo_ae_ds_losses.update(fo_ae_ds_loss.item(), img.size(0))
                    fo_ae_ds_losses_fx.update(fo_ae_ds_loss_fx.item(), img.size(0))

                    if i == mark:
                        pic = to_img(outputs.cpu().data)
                        save_image(pic,
                                   os.path.join(outdir,
                                                'Epsilon_{epsilon}_FO_AE_DS_Loss_{loss:.3f}_LossFx_{loss_fx:.3f}.png').format(
                                       epsilon=epsilon,
                                       loss=fo_ae_ds_losses.avg, loss_fx=fo_ae_ds_losses_fx.avg))

                    # ------------- clean Reconstruction Acc  [ZO-AE-DS] --------------
                    outputs = adv_img.repeat((1, noise_num, 1, 1)).view(-1, channel, h, w)
                    outputs = outputs + torch.randn_like(outputs, device='cuda') * noise_sd
                    outputs = enc_zo_ae_ds(de_zo_ae_ds(outputs))
                    outputs = recon_net(decoder(outputs))
                    outputs = outputs.view(-1, batch_size, noise_num, channel, h, w).mean(2, keepdim=True).reshape(
                        batch_size,
                        channel, h,
                        w)
                    zo_ae_ds_loss = criterion(outputs, img_original)
                    zo_ae_ds_loss_fx = criterion(outputs, fx)
                    zo_ae_ds_losses.update(zo_ae_ds_loss.item(), img.size(0))
                    zo_ae_ds_losses_fx.update(zo_ae_ds_loss_fx.item(), img.size(0))

                    if i == mark:
                        pic = to_img(outputs.cpu().data)
                        save_image(pic,
                                   os.path.join(outdir,
                                                'Epsilon_{epsilon}_ZO_AE_DS_Loss_{loss:.3f}_LossFx_{loss_fx:.3f}.png').format(
                                       epsilon=epsilon,
                                       loss=zo_ae_ds_losses.avg, loss_fx=zo_ae_ds_losses_fx.avg))




        log = 'Test: [{0}/{1}]\t'' \
        ''Recon {recon_loss.avg:.4f} ({recon_loss_fx.avg:.4f})\t'' \
        ''FO-DS {fo_ds_loss.avg:.4f} ({fo_ds_loss_fx.avg:.4f})\t'' \
        ''ZO-DS {zo_ds_loss.avg:.4f} ({zo_ds_loss_fx.avg:.4f})\t'' \
        ''FO-AE-DS {fo_ae_ds_loss.avg:.4f} ({fo_ae_ds_loss_fx.avg:.4f})\t'' \
        ''ZO-AE-DS {zo_ae_ds_loss.avg:.4f} ({zo_ae_ds_loss_fx.avg:.4f})\n'.format(
            i, len(loader), recon_loss =recon_losses, recon_loss_fx =recon_losses_fx, fo_ds_loss =fo_ds_losses, fo_ds_loss_fx = fo_ds_losses_fx, zo_ds_loss=zo_ds_losses, zo_ds_loss_fx = zo_ds_losses_fx, fo_ae_ds_loss = fo_ae_ds_losses, fo_ae_ds_loss_fx= fo_ae_ds_losses_fx, zo_ae_ds_loss= zo_ae_ds_losses, zo_ae_ds_loss_fx=zo_ae_ds_losses_fx)

        print(log)

    return recon_losses.avg, recon_losses_fx.avg, fo_ds_losses.avg, fo_ds_losses_fx.avg, zo_ds_losses.avg, zo_ds_losses_fx.avg, fo_ae_ds_losses.avg, fo_ae_ds_losses_fx.avg, zo_ae_ds_losses.avg, zo_ae_ds_losses_fx.avg
    # Return Empericial Robustness Recon acc (MSE)


def norm_loss(x, y):
    l1 = L1Loss(size_average=None, reduce=None, reduction='none').cuda()
    l2 = MSELoss(size_average=None, reduce=None, reduction='none').cuda()
    zero = torch.zeros_like(x)
    loss = l1(x, y)/l1(y, zero) + l2(x,y)/l2(y, zero)
    return loss

def to_img(x):
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss

if __name__ == "__main__":
    main()
