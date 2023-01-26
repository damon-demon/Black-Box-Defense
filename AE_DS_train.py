from architectures import DENOISERS_ARCHITECTURES, get_architecture, IMAGENET_CLASSIFIERS, AUTOENCODER_ARCHITECTURES
from datasets import get_dataset, DATASETS
from torch.nn import MSELoss, CrossEntropyLoss
from torch.optim import SGD, Optimizer, Adam
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from train_utils import AverageMeter, accuracy, init_logfile, log, copy_code, requires_grad_, measurement

import argparse
from datetime import datetime
import os
import time
import torch
import itertools
from robustness import datasets as dataset_r
from robustness.tools.imagenet_helpers import common_superclass_wnid, ImageNetHierarchy
from torchvision.utils import save_image
from recon_attacks import Attacker, recon_PGD_L2

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Training Objective
parser.add_argument('--train_objective', default='classification', type=str,
                    help="The whole model is built for classificaiton / reconstruction",
                    choices=['classification', 'reconstruction'])
parser.add_argument('--ground_truth', default='original_output', type=str,
                    help="The choice of groundtruth",
                    choices=['original_output', 'labels'])

# Dataset
parser.add_argument('--dataset', type=str, choices=DATASETS)
parser.add_argument('--data_min', default=-2.5090184, type=float, help='minimum value of training data')
parser.add_argument('--data_max', default=3.3369503, type=float, help='maximum value of training data')

parser.add_argument('--batch', default=256, type=int, metavar='N', help='batchsize (default: 256)')
parser.add_argument('--measurement', default=576, type=int, metavar='N', help='the size of measurement for image reconstruction')


# Optimization Method
parser.add_argument('--optimization_method', default='FO', type=str,
                    help="FO: First-Order (White-Box), ZO: Zeroth-Order (Black-box)",
                    choices=['FO', 'ZO'])
parser.add_argument('--zo_method', default='RGE', type=str,
                    help="Random Gradient Estimation: RGE, Coordinate-Wise Gradient Estimation: CGE",
                    choices=['RGE', 'CGE', 'CGE_sim'])
parser.add_argument('--q', default=192, type=int, metavar='N',
                    help='query direction (default: 20)')
parser.add_argument('--mu', default=0.005, type=float, metavar='N',
                    help='Smoothing Parameter')

# Model type
parser.add_argument('--model_type', default='AE_DS', type=str,
                    help="Denoiser + (AutoEncoder) + classifier/reconstructor",
                    choices=['DS', 'AE_DS'])
parser.add_argument('--arch', type=str, choices=DENOISERS_ARCHITECTURES)
parser.add_argument('--encoder_arch', type=str, default='cifar_encoder', choices=AUTOENCODER_ARCHITECTURES)
parser.add_argument('--decoder_arch', type=str, default='cifar_decoder', choices=AUTOENCODER_ARCHITECTURES)
parser.add_argument('--classifier', default='', type=str,
                    help='path to the classifier used with the `classificaiton`'
                         'or `stability` objectives of the denoiser.')
parser.add_argument('--pretrained-denoiser', default='', type=str, help='path to a pretrained denoiser')
parser.add_argument('--pretrained-encoder', default='', type=str, help='path to a pretrained encoder')
parser.add_argument('--pretrained-decoder', default='', type=str, help='path to a pretrained decoder')

# Model to be trained
parser.add_argument('--train_method', default='whole', type=str,
                    help="*part*: only denoiser parameters would be optimized; *whole*: denoiser and encoder parameters would be optimized, *whole_plus*: denoiser and auto-encoder parameters would be optimized",
                    choices=['part', 'whole', 'whole_plus'])

# Training Setting
parser.add_argument('--outdir', type=str, help='folder to save denoiser and training log)')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--optimizer', default='Adam', type=str,
                    help='SGD, Adam', choices=['SGD', 'Adam'])
parser.add_argument('--epochs', default=600, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--lr_step_size', type=int, default=100,
                    help='How often to decrease learning by gamma.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--gpu', default=None, type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--noise_sd', default=0.0, type=float,
                    help="standard deviation of noise distribution for data augmentation")
parser.add_argument('--visual_freq', default=1, type=int,
                    metavar='N', help='visualization frequency (default: 5)')

# Parameters for adv examples generation
parser.add_argument('--noise_num', default=10, type=int,
                    help='number of noise for smoothing')
parser.add_argument('--num_steps', default=40, type=int,
                    help='Number of steps for attack')
parser.add_argument('--epsilon', default=512, type=float)

args = parser.parse_args()
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

toPilImage = ToPILImage()


def main():
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # Copy code to output directory
    copy_code(args.outdir)
    pin_memory = (args.dataset == "imagenet")

    # --------------------- Dataset Loading ----------------------
    if args.dataset == 'cifar10' or args.dataset == 'stl10' or args.dataset == 'mnist':
        train_dataset = get_dataset(args.dataset, 'train')
        test_dataset = get_dataset(args.dataset, 'test')

        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch,
                                  num_workers=args.workers, pin_memory=pin_memory)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch,
                                 num_workers=args.workers, pin_memory=pin_memory)

    elif args.dataset == 'restricted_imagenet':
        in_path = '/localscratch2/damondemon/datasets/imagenet'
        in_info_path = '/localscratch2/damondemon/datasets/imagenet_info'
        in_hier = ImageNetHierarchy(in_path, in_info_path)

        superclass_wnid = ['n02084071', 'n02120997', 'n01639765', 'n01662784', 'n02401031', 'n02131653', 'n02484322',
                           'n01976957', 'n02159955', 'n01482330']

        class_ranges, label_map = in_hier.get_subclasses(superclass_wnid, balanced=True)
        custom_dataset = dataset_r.CustomImageNet(in_path, class_ranges)
        train_loader, test_loader = custom_dataset.make_loaders(workers=4, batch_size=args.batch)

    # --------------------- Model Loading -------------------------
    # a) Denoiser
    if args.pretrained_denoiser:
        checkpoint = torch.load(args.pretrained_denoiser)
        assert checkpoint['arch'] == args.arch
        denoiser = get_architecture(checkpoint['arch'], args.dataset)
        denoiser.load_state_dict(checkpoint['state_dict'])
    else:
        denoiser = get_architecture(args.arch, args.dataset)

    # b) AutoEncoder
    if args.model_type == 'AE_DS':
        if args.pretrained_encoder:
            checkpoint = torch.load(args.pretrained_encoder)
            assert checkpoint['arch'] == args.encoder_arch
            encoder = get_architecture(checkpoint['arch'], args.dataset)
            encoder.load_state_dict(checkpoint['state_dict'])
        else:
            encoder = get_architecture(args.encoder_arch, args.dataset)

        if args.pretrained_decoder:
            checkpoint = torch.load(args.pretrained_decoder)
            assert checkpoint['arch'] == args.decoder_arch
            decoder = get_architecture(checkpoint['arch'], args.dataset)
            decoder.load_state_dict(checkpoint['state_dict'])
        else:
            decoder = get_architecture(args.decoder_arch, args.dataset)

    # c) Classifier / Reconstructor
    checkpoint = torch.load(args.classifier)
    clf = get_architecture(checkpoint['arch'], args.dataset)
    clf.load_state_dict(checkpoint['state_dict'])
    clf.cuda().eval()
    requires_grad_(clf, False)

    # --------------------- Model to be trained ------------------------
    if args.optimizer == 'Adam':
        if args.train_method =='part':
            optimizer = Adam(denoiser.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if args.train_method =='whole':
            optimizer = Adam(itertools.chain(denoiser.parameters(), encoder.parameters()), lr=args.lr, weight_decay=args.weight_decay)
        if args.train_method =='whole_plus':
            optimizer = Adam(itertools.chain(denoiser.parameters(), encoder.parameters(), decoder.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        if args.train_method =='part':
            optimizer = SGD(denoiser.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        if args.train_method =='whole':
            optimizer = SGD(itertools.chain(denoiser.parameters(), encoder.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        if args.train_method =='whole_plus':
            optimizer = SGD(itertools.chain(denoiser.parameters(), encoder.parameters(), decoder.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)

    # --------------------- Log file initialization ---------------------
    starting_epoch = 0
    logfilename = os.path.join(args.outdir, 'log.txt')
    if args.train_objective == 'classification':
        init_logfile(logfilename, "epoch\ttime\tlr\ttrainloss\ttestloss\ttestAcc")
    elif args.train_objective == 'reconstruction':
        init_logfile(logfilename,
                     "epoch\ttime\tlr\ttrain_stab_loss\tClean_TestLoss_NoDenoiser\tSmoothed_Clean_TestLoss_NoDenoiser\tClean_TestLoss\tSmoothed_Clean_TestLoss\tNoDenoiser_AdvLoss\tSmoothed_NoDenoiser_AdvLoss\tAdv_Loss\tSmoothed_AdvLoss")

    # --------------------- Objective function ---------------------
    if args.train_objective == 'classification':
        criterion = CrossEntropyLoss(size_average=None, reduce=False, reduction='none').cuda()
    elif args.train_objective == 'reconstruction':
        criterion = MSELoss(size_average=None, reduce=None, reduction='none').cuda()

    # --------------------- Start Training -------------------------------
    best_acc = 0
    for epoch in range(starting_epoch, args.epochs):
        before = time.time()

        # classificaiton / reconstruction
        if args.train_objective == 'classification':
            if args.model_type == 'AE_DS':
                train_loss = train_ae(train_loader, encoder, decoder, denoiser, criterion, optimizer, epoch,
                                      args.noise_sd,
                                      clf)
                _, train_acc = test_with_classifier_ae(train_loader, encoder, decoder, denoiser, criterion,
                                                       args.noise_sd,
                                                       args.print_freq, clf)
                test_loss, test_acc = test_with_classifier_ae(test_loader, encoder, decoder, denoiser, criterion,
                                                              args.noise_sd,
                                                              args.print_freq, clf)
            elif args.model_type == 'DS':
                train_loss = train(train_loader, denoiser, criterion, optimizer, epoch, args.noise_sd,
                                   clf)
                _, train_acc = test_with_classifier(train_loader, denoiser, criterion, args.noise_sd,
                                                    args.print_freq, clf)
                test_loss, test_acc = test_with_classifier(test_loader, denoiser, criterion,
                                                           args.noise_sd,
                                                           args.print_freq, clf)
            after = time.time()

            log(logfilename, "{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}".format(
                epoch, after - before,
                args.lr, train_loss, test_loss, train_acc, test_acc))

        elif args.train_objective == 'reconstruction':
            if args.model_type == 'AE_DS':
                stab_train_loss = recon_train_ae(train_loader, encoder, decoder, denoiser, criterion, optimizer, epoch,
                                        args.noise_sd, clf)
                test_no_loss, test_no_loss_smooth, test_loss, test_loss_smooth, recon_loss, recon_loss_smooth, adv_loss, smooth_loss = test_with_recon_ae(
                    test_loader, encoder, decoder, denoiser, criterion, args.outdir, args.noise_sd, epoch,
                    args.visual_freq, args.noise_num,
                    args.num_steps, args.epsilon, args.print_freq, clf)

                log(logfilename,
                    "{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}".format(
                        epoch, after - before,
                        args.lr, stab_train_loss, test_no_loss, test_no_loss_smooth, test_loss, test_loss_smooth,
                        recon_loss, recon_loss_smooth, adv_loss, smooth_loss))

            elif args.model_type == 'DS':
                stab_train_loss = recon_train(train_loader, denoiser, criterion, optimizer, epoch,
                                                 args.noise_sd, clf)
                test_no_loss, test_no_loss_smooth, test_loss, test_loss_smooth, recon_loss, recon_loss_smooth, adv_loss, smooth_loss = test_with_recon(
                    test_loader, denoiser, criterion, args.outdir, args.noise_sd, epoch,
                    args.visual_freq, args.noise_num,
                    args.num_steps, args.epsilon, args.print_freq, clf)

                log(logfilename,
                    "{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}".format(
                        epoch, after - before,
                        args.lr, stab_train_loss, test_no_loss, test_no_loss_smooth, test_loss, test_loss_smooth,
                        recon_loss, recon_loss_smooth, adv_loss, smooth_loss))

        scheduler.step(epoch)
        args.lr = scheduler.get_lr()[0]

        # -----------------  Save the latest model  -------------------
        torch.save({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': denoiser.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(args.outdir, 'denoiser.pth.tar'))

        if args.model_type == 'AE_DS':
            torch.save({
                'epoch': epoch + 1,
                'arch': args.encoder_arch,
                'state_dict': encoder.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(args.outdir, 'encoder.pth.tar'))

            torch.save({
                'epoch': epoch + 1,
                'arch': args.decoder_arch,
                'state_dict': decoder.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(args.outdir, 'decoder.pth.tar'))

        # ----------------- Save the best model according to acc -----------------
        if test_acc > best_acc:
            best_acc = test_acc
        else:
            continue

        torch.save({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': denoiser.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(args.outdir, 'best_denoiser.pth.tar'))

        if args.model_type == 'AE_DS':
            torch.save({
                'epoch': epoch + 1,
                'arch': args.encoder_arch,
                'state_dict': encoder.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(args.outdir, 'best_encoder.pth.tar'))

            torch.save({
                'epoch': epoch + 1,
                'arch': args.decoder_arch,
                'state_dict': decoder.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(args.outdir, 'best_decoder.pth.tar'))


def train(loader: DataLoader, denoiser: torch.nn.Module, criterion, optimizer: Optimizer, epoch: int, noise_sd: float,
          classifier: torch.nn.Module = None):
    """
    Function for training denoiser for one epoch
        :param loader:DataLoader: training dataloader
        :param denoiser:torch.nn.Module: the denoiser being trained
        :param criterion: loss function
        :param optimizer:Optimizer: optimizer used during trainined
        :param epoch:int: the current epoch (for logging)
        :param noise_sd:float: the std-dev of the Guassian noise perturbation of the input
        :param classifier:torch.nn.Module=None: a ``freezed'' classifier attached to the denoiser
                                                (required classifciation/stability objectives), None for denoising objective
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    # switch to train mode
    denoiser.train()
    classifier.eval()

    for i, (inputs, targets) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs = inputs.cuda()
        targets = targets.cuda()
        if args.ground_truth == 'original_output':
            with torch.no_grad():
                targets = classifier(inputs)
                targets = targets.argmax(1).detach().clone()

        noise = torch.randn_like(inputs, device='cuda') * noise_sd
        recon = denoiser(inputs + noise)

        if args.optimization_method == 'FO':
            recon = classifier(recon)
            loss = criterion(recon, targets)

            # record loss
            losses.update(loss.item(), inputs.size(0))

        elif args.optimization_method == 'ZO':
            recon.requires_grad_(True)
            recon.retain_grad()

            # Obtain the Shape of Inputs (Batch_size x Channel x H x W)
            batch_size = recon.size()[0]
            channel = recon.size()[1]
            h = recon.size()[2]
            w = recon.size()[3]
            d = channel * h * w

            # For DS model, only RGE could be exploited for ZO gradient estimation
            if args.zo_method == 'RGE':
                with torch.no_grad():
                    m, sigma = 0, 100  # mean and standard deviation
                    mu = torch.tensor(args.mu).cuda()
                    q = torch.tensor(args.q).cuda()

                    # Forward Inference (Original)
                    original_pre = classifier(inputs).argmax(1).detach().clone()

                    recon_pre = classifier(recon)
                    loss_0 = criterion(recon_pre, original_pre)

                    # record original loss
                    loss_0_mean = loss_0.mean()
                    losses.update(loss_0_mean.item(), inputs.size(0))

                    recon_flat_no_grad = torch.flatten(recon, start_dim=1).cuda()
                    grad_est = torch.zeros(batch_size, d).cuda()

                    # ZO Gradient Estimation
                    for k in range(args.q):
                        # Obtain a random direction vector
                        u = torch.normal(m, sigma, size=(batch_size, d))
                        u_norm = torch.norm(u, p=2, dim=1).reshape(batch_size, 1).expand(batch_size, d)    # dim -- careful
                        u = torch.div(u, u_norm).cuda()       # (batch_size, d)

                        # Forward Inference (reconstructed image + random direction vector)
                        recon_q = recon_flat_no_grad + mu * u
                        recon_q = recon_q.view(batch_size, channel, h, w)
                        recon_q_pre = classifier(recon_q)

                        # Loss Calculation and Gradient Estimation
                        loss_tmp = criterion(recon_q_pre, original_pre)
                        loss_diff = torch.tensor(loss_tmp - loss_0)
                        grad_est = grad_est + (d / q) * u * loss_diff.reshape(batch_size, 1).expand_as(u) / mu

                recon_flat = torch.flatten(recon, start_dim=1).cuda()
                grad_est_no_grad = grad_est.detach()

                # reconstructed image * gradient estimation   <--   g(x) * a
                loss = torch.sum(recon_flat * grad_est_no_grad, dim=-1).mean()

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                epoch, i, len(loader), batch_time=batch_time,
                data_time=data_time, loss=losses))

    return losses.avg


def test_with_classifier(loader: DataLoader, denoiser: torch.nn.Module, criterion, noise_sd: float, print_freq: int, classifier: torch.nn.Module):
    """
    A function to test the classification performance of a denoiser when attached to a given classifier
        :param loader:DataLoader: test dataloader
        :param denoiser:torch.nn.Module: the denoiser
        :param criterion: the loss function (e.g. CE)
        :param noise_sd:float: the std-dev of the Guassian noise perturbation of the input
        :param print_freq:int: the frequency of logging
        :param classifier:torch.nn.Module: the classifier to which the denoiser is attached
    """

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # switch to eval mode
    classifier.eval()
    if denoiser:
        denoiser.eval()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            # measure data loading time
            data_time.update(time.time() - end)

            inputs = inputs.cuda()
            targets = targets.cuda()

            # augment inputs with noise
            inputs = inputs + torch.randn_like(inputs, device='cuda') * noise_sd

            if denoiser is not None:
                inputs = denoiser(inputs)
            # compute output
            outputs = classifier(inputs)
            loss = criterion(outputs, targets)
            loss_mean = loss.mean()

            # measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss_mean.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                log = 'Test: [{0}/{1}]\t'' \
                ''Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'' \
                ''Data {data_time.val:.3f} ({data_time.avg:.3f})\t'' \
                ''Loss {loss.val:.4f} ({loss.avg:.4f})\t'' \
                ''Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'' \
                ''Acc@5 {top5.val:.3f} ({top5.avg:.3f})\n'.format(
                    i, len(loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5)

                print(log)
        return (losses.avg, top1.avg)


def train_ae(loader: DataLoader, encoder: torch.nn.Module, decoder: torch.nn.Module, denoiser: torch.nn.Module, criterion,
          optimizer: Optimizer, epoch: int, noise_sd: float,
          classifier: torch.nn.Module = None):
    """
    Function for training denoiser for one epoch
        :param loader:DataLoader: training dataloader
        :param denoiser:torch.nn.Module: the denoiser being trained
        :param criterion: loss function
        :param optimizer:Optimizer: optimizer used during trainined
        :param epoch:int: the current epoch (for logging)
        :param noise_sd:float: the std-dev of the Guassian noise perturbation of the input
        :param classifier:torch.nn.Module=None: a ``freezed'' classifier attached to the denoiser
                                                (required classifciation/stability objectives), None for denoising objective
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    # switch to train mode
    denoiser.train()

    if args.train_method == 'part':
        encoder.eval()
        decoder.eval()
    if args.train_method == 'whole':
        encoder.train()
        decoder.eval()
    if args.train_method == 'whole_plus':
        encoder.train()
        decoder.train()

    if classifier:
        classifier.eval()

    for i, (inputs, targets) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs = inputs.cuda()
        targets = targets.cuda()
        if args.ground_truth == 'original_output':
            with torch.no_grad():
                targets = classifier(inputs)
                targets = targets.argmax(1).detach().clone()

        # augment inputs with noise
        noise = torch.randn_like(inputs, device='cuda') * noise_sd

        recon = denoiser(inputs + noise)
        recon = encoder(recon)

        if args.optimization_method == 'FO':
            recon = decoder(recon)
            recon = classifier(recon)
            loss = criterion(recon, targets)

            # record loss
            losses.update(loss.item(), inputs.size(0))

        elif args.optimization_method == 'ZO':
            recon.requires_grad_(True)
            recon.retain_grad()

            # Obtain the Shape of Inputs (Batch_size x Channel x H x W)
            batch_size = recon.size()[0]
            channel = recon.size()[1]
            h = recon.size()[2]
            w = recon.size()[3]
            d = channel * h * w

            if args.zo_method =='RGE':
                with torch.no_grad():
                    m, sigma = 0, 100  # mean and standard deviation
                    mu = torch.tensor(args.mu).cuda()
                    q = torch.tensor(args.q).cuda()

                    # Forward Inference (Original)
                    original_pre = classifier(inputs).argmax(1).detach().clone()

                    recon_pre = classifier(decoder(recon))
                    loss_0 = criterion(recon_pre, original_pre)

                    # record original loss
                    loss_0_mean = loss_0.mean()
                    losses.update(loss_0_mean.item(), inputs.size(0))

                    recon_flat_no_grad = torch.flatten(recon, start_dim=1).cuda()
                    grad_est = torch.zeros(batch_size, d).cuda()

                    # ZO Gradient Estimation
                    for k in range(args.q):
                        # Obtain a random direction vector
                        u = torch.normal(m, sigma, size=(batch_size, d))
                        u_norm = torch.norm(u, p=2, dim=1).reshape(batch_size, 1).expand(batch_size, d)    # dim -- careful
                        u = torch.div(u, u_norm).cuda()       # (batch_size, d)

                        # Forward Inference (reconstructed image + random direction vector)
                        recon_q = recon_flat_no_grad + mu * u
                        recon_q = recon_q.view(batch_size, channel, h, w)
                        recon_q_pre = classifier(decoder(recon_q))

                        # Loss Calculation and Gradient Estimation
                        loss_tmp = criterion(recon_q_pre, original_pre)
                        loss_diff = torch.tensor(loss_tmp - loss_0)
                        grad_est = grad_est + (d / q) * u * loss_diff.reshape(batch_size, 1).expand_as(u) / mu

                recon_flat = torch.flatten(recon, start_dim=1).cuda()
                grad_est_no_grad = grad_est.detach()

                # reconstructed image * gradient estimation   <--   g(x) * a
                loss = torch.sum(recon_flat * grad_est_no_grad, dim=-1).mean()

            elif args.zo_method =='CGE':
                with torch.no_grad():
                    mu = torch.tensor(args.mu).cuda()
                    q = torch.tensor(args.q).cuda()

                    # Forward Inference (Original)
                    original_pre = classifier(inputs).argmax(1).detach().clone()

                    recon_pre = classifier(decoder(recon))
                    loss_0 = criterion(recon_pre, original_pre)

                    # record original loss
                    loss_0_mean = loss_0.mean()
                    losses.update(loss_0_mean.item(), inputs.size(0))

                    recon_flat_no_grad = torch.flatten(recon, start_dim=1).cuda()
                    grad_est = torch.zeros(batch_size, d).cuda()

                    # ZO Gradient Estimation
                    for k in range(d):
                        # Obtain a direction vector (1-0)
                        u = torch.zeros(batch_size, d).cuda()
                        u[:, k] = 1

                        # Forward Inference (reconstructed image + random direction vector)
                        recon_q_plus = recon_flat_no_grad + mu * u
                        recon_q_minus = recon_flat_no_grad - mu * u

                        recon_q_plus = recon_q_plus.view(batch_size, channel, h, w)
                        recon_q_minus = recon_q_minus.view(batch_size, channel, h, w)
                        recon_q_pre_plus = classifier(decoder(recon_q_plus))
                        recon_q_pre_minus = classifier(decoder(recon_q_minus))

                        # Loss Calculation and Gradient Estimation
                        loss_tmp_plus = criterion(recon_q_pre_plus, original_pre)
                        loss_tmp_minus = criterion(recon_q_pre_minus, original_pre)

                        loss_diff = torch.tensor(loss_tmp_plus - loss_tmp_minus)
                        grad_est = grad_est + u * loss_diff.reshape(batch_size, 1).expand_as(u) / (2 * mu)

                recon_flat = torch.flatten(recon, start_dim=1).cuda()
                grad_est_no_grad = grad_est.detach()

                # reconstructed image * gradient estimation   <--   g(x) * a
                loss = torch.sum(recon_flat * grad_est_no_grad, dim=-1).mean()

            elif args.zo_method =='CGE_sim':
                # Generate Coordinate-wise Query Matrix
                u_flat = torch.zeros(1, args.q, d).cuda()
                for k in range(d):
                    u_flat[:, k, k] = 1
                u_flat = u_flat.repeat(1, batch_size, 1).view(batch_size * args.q, d)
                u = u_flat.view(-1, channel, h, w)

                with torch.no_grad():
                    mu = torch.tensor(args.mu).cuda()

                    recon_pre = classifier(decoder(recon))  # (batch_size, 10)

                    loss_0 = criterion(recon_pre, targets)  # (batch_size )
                    loss_0_mean = loss_0.mean()
                    losses.update(loss_0_mean.item(), inputs.size(0))

                    # Repeat q times
                    targets = targets.view(batch_size, 1).repeat(1, args.q).view(batch_size * args.q)  # (batch_size * q, )

                    recon_q = recon.repeat((1, args.q, 1, 1)).view(-1, channel, h, w)
                    recon_q_plus = recon_q + mu * u
                    recon_q_minus = recon_q - mu * u

                    # Black-Box Query
                    recon_q_pre_plus = classifier(decoder(recon_q_plus))
                    recon_q_pre_minus = classifier(decoder(recon_q_minus))
                    loss_tmp_plus = criterion(recon_q_pre_plus, targets)
                    loss_tmp_minus = criterion(recon_q_pre_minus, targets)

                    loss_diff = torch.tensor(loss_tmp_plus - loss_tmp_minus)
                    grad_est = u_flat * loss_diff.reshape(batch_size * args.q, 1).expand_as(u_flat) / (2 * mu)
                    grad_est = grad_est.view(batch_size, args.q, d).mean(1, keepdim=True).view(batch_size,d)

                recon_flat = torch.flatten(recon, start_dim=1).cuda()
                grad_est_no_grad = grad_est.detach()

                # reconstructed image * gradient estimation   <--   g(x) * a
                loss = torch.sum(recon_flat * grad_est_no_grad, dim=-1).mean()  # l_mean

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                epoch, i, len(loader), batch_time=batch_time,
                data_time=data_time, loss=losses))

    return losses.avg


def test_with_classifier_ae(loader: DataLoader, encoder: torch.nn.Module, decoder: torch.nn.Module,denoiser: torch.nn.Module, criterion, noise_sd: float, print_freq: int, classifier: torch.nn.Module):
    """
    A function to test the classification performance of a denoiser when attached to a given classifier
        :param loader:DataLoader: test dataloader
        :param denoiser:torch.nn.Module: the denoiser
        :param criterion: the loss function (e.g. CE)
        :param noise_sd:float: the std-dev of the Guassian noise perturbation of the input
        :param print_freq:int: the frequency of logging
        :param classifier:torch.nn.Module: the classifier to which the denoiser is attached
    """

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # switch to eval mode
    classifier.eval()
    encoder.eval()
    decoder.eval()
    if denoiser:
        denoiser.eval()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            # measure data loading time
            data_time.update(time.time() - end)

            inputs = inputs.cuda()
            targets = targets.cuda()

            # augment inputs with noise
            inputs = inputs + torch.randn_like(inputs, device='cuda') * noise_sd

            if denoiser is not None:
                inputs = denoiser(inputs)
            # compute output
            outputs = encoder(inputs)
            outputs = decoder(outputs)
            outputs = classifier(outputs)
            loss = criterion(outputs, targets)
            loss_mean = loss.mean()

            # measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss_mean.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                log = 'Test: [{0}/{1}]\t'' \
                ''Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'' \
                ''Data {data_time.val:.3f} ({data_time.avg:.3f})\t'' \
                ''Loss {loss.val:.4f} ({loss.avg:.4f})\t'' \
                ''Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'' \
                ''Acc@5 {top5.val:.3f} ({top5.avg:.3f})\n'.format(
                    i, len(loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5)

                print(log)

        return (losses.avg, top1.avg)


def recon_train(loader: DataLoader, denoiser: torch.nn.Module, criterion, optimizer: Optimizer, epoch: int, noise_sd: float,
          recon_net: torch.nn.Module = None):
    """
    Function for training denoiser for one epoch
        :param loader:DataLoader: training dataloader
        :param denoiser:torch.nn.Module: the denoiser being trained
        :param criterion: loss function
        :param optimizer:Optimizer: optimizer used during trainined
        :param epoch:int: the current epoch (for logging)
        :param noise_sd:float: the std-dev of the Guassian noise perturbation of the input
        :param classifier:torch.nn.Module=None: a ``freezed'' classifier attached to the denoiser
                                                (required classifciation/stability objectives), None for denoising objective
    """
    n_measurement = args.measurement
    mm_min = torch.tensor(args.data_min, dtype=torch.long).cuda()
    mm_max = torch.tensor(args.data_max, dtype=torch.long).cuda()
    mm_dis = mm_max - mm_min

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()

    # switch to train mode
    denoiser.train()
    if recon_net:
        recon_net.eval()

    for i, (img_original, _) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # Obtain the Shape of Inputs (Batch_size x Channel x H x W)
        batch_size = img_original.size()[0]
        channel = img_original.size()[1]
        h = img_original.size()[2]
        w = img_original.size()[3]
        d = channel * h * w

        img_original = img_original.cuda()           # input x (batch,  channel, h, w)
        img = img_original.view(batch_size, d).cuda()

        if i ==0:
            a = measurement(n_measurement, d)    # Measurement Matrix
            print("-----------------------training--------------------")
            print(a[0, :])

        img = torch.mm(img, a)     # y = A^T x
        img = torch.mm(img, a.t())
        img = img.view(batch_size, channel, h, w)
        img = img.float()
        img = (img - mm_min) / mm_dis

        #augment inputs with noise
        noise = torch.randn_like(img, device='cuda') * noise_sd
        recon = denoiser(img + noise)

        if args.optimization_method == 'FO':
            recon = recon_net(recon)
            stab_loss = criterion(recon, img_original)

            # record original loss
            stab_loss_mean = stab_loss.mean()
            losses.update(stab_loss_mean.item(), img_original.size(0))

            # compute gradient and do step
            optimizer.zero_grad()
            stab_loss_mean.backward()
            optimizer.step()

        elif args.optimization_method == 'ZO':
            recon.requires_grad_(True)
            recon.retain_grad()

            with torch.no_grad():
                mu = torch.tensor(args.mu).cuda()
                q = torch.tensor(args.q).cuda()

                # Forward Inference (Original)
                original_recon = recon_net(img)

                recon_test = recon_net(recon)
                loss_0 = criterion(recon_test, original_recon)
                # record original loss
                loss_0_mean = loss_0.mean()
                losses.update(loss_0_mean.item(), img_original.size(0))

                recon_flat_no_grad = torch.flatten(recon, start_dim=1).cuda()
                grad_est = torch.zeros(batch_size, d).cuda()

                # ZO Gradient Estimation
                for k in range(d):
                    # Obtain a direction vector (1-0)
                    u = torch.zeros(batch_size, d).cuda()
                    u[:, k] = 1

                    # Forward Inference (reconstructed image + random direction vector)
                    recon_q_plus = recon_flat_no_grad + mu * u
                    recon_q_minus = recon_flat_no_grad - mu * u

                    recon_q_plus = recon_q_plus.view(batch_size, channel, h, w)
                    recon_q_minus = recon_q_minus.view(batch_size, channel, h, w)
                    recon_q_pre_plus = recon_net(recon_q_plus)
                    recon_q_pre_minus = recon_net(recon_q_minus)

                    # Loss Calculation and Gradient Estimation
                    loss_tmp_plus = criterion(recon_q_pre_plus, original_recon)
                    loss_tmp_minus = criterion(recon_q_pre_minus, original_recon)

                    loss_diff = torch.tensor(loss_tmp_plus - loss_tmp_minus)
                    loss_diff = loss_diff.mean(3, keepdim=True).mean(2, keepdim=True)
                    grad_est = grad_est + u * loss_diff.reshape(batch_size, 1).expand_as(u) / (2 * mu)

            recon_flat = torch.flatten(recon, start_dim=1).cuda()
            grad_est_no_grad = grad_est.detach()

            # reconstructed image * gradient estimation   <--   g(x) * a
            loss = torch.sum(recon_flat * grad_est_no_grad, dim=-1).mean()  # l_mean
            # compute gradient and do step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            log = 'Epoch: [{0}][{1}/{2}]\t'' \
            ''Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'' \
            ''Data {data_time.val:.3f} ({data_time.avg:.3f})\t'' \
            ''Stab_Loss {stab_loss.val:.4f} ({stab_loss.avg:.4f})''\n'.format(
                epoch, i, len(loader), batch_time=batch_time,
                data_time=data_time, stab_loss = losses)

            print(log)

    return losses.avg


def test_with_recon(loader: DataLoader, denoiser: torch.nn.Module, criterion, outdir:str, noise_sd: float, epoch: int, visual_freq: int, noise_num: int, num_steps: int, epsilon: int, print_freq: int, recon_net: torch.nn.Module):
    """
    A function to test the classification performance of a denoiser when attached to a given classifier
        :param loader:DataLoader: test dataloader
        :param denoiser:torch.nn.Module: the denoiser
        :param criterion: the loss function (e.g. MAE)
        :param noise_sd:float: the std-dev of the Guassian noise perturbation of the input
        :param print_freq:int: the frequency of logging
        :param recon_net:torch.nn.Module: the reconstruction network to which the denoiser is attached
    """

    # switch to eval mode
    recon_net.eval()
    if denoiser:
        denoiser.eval()

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_no = AverageMeter()
    losses = AverageMeter()
    recon_losses = AverageMeter()
    adv_losses = AverageMeter()
    smooth_losses = AverageMeter()
    smooth_losses_no_clean = AverageMeter()
    smooth_losses_clean = AverageMeter()
    smooth_losses_no_adv = AverageMeter()
    end = time.time()

    n_measurement = args.measurement
    mm_min = torch.tensor(args.data_min, dtype=torch.long).cuda()
    mm_max = torch.tensor(args.data_max, dtype=torch.long).cuda()
    mm_dis = mm_max - mm_min

    mark = 39

    attacker = recon_PGD_L2(steps=num_steps, device='cuda', max_norm=epsilon)  # remember epsilon/256

    # switch to eval mode
    recon_net.eval()
    if denoiser:
        denoiser.eval()

    for i, (img_original, _) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        img_original = img_original.cuda()

        # Obtain the Shape of Inputs (Batch_size x Channel x H x W)
        batch_size = img_original.size()[0]
        channel = img_original.size()[1]
        h = img_original.size()[2]
        w = img_original.size()[3]
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


        # adv_img = attacker.attack(model, img, img_original, criterion)
        adv_img = attacker.attack(recon_net, img, img_original, criterion)

        # augment inputs with noise
        # inputs = img + torch.randn_like(img, device='cuda') * noise_sd

        with torch.no_grad():
            # ----------Clean Reconstruction Acc without denoiser------------
            outputs = recon_net(img)
            loss_no = criterion(outputs, img_original)
            loss_no_mean = loss_no.mean()
            losses_no.update(loss_no_mean.item(), img.size(0))

            if i == mark and epoch % visual_freq == 0:
                pic = to_img(outputs.cpu().data)
                save_image(pic,
                           os.path.join(outdir, 'Epoch_{epoch}_clean_NoDenoiser_Loss_{loss:.3f}.png').format(
                               epoch=epoch,
                               loss=losses_no.avg))

            # ------------- clean Reconstruction Acc using smoothed reconstruction network --------------
            outputs = img.repeat((1, noise_num, 1, 1)).view(-1, channel, h, w)
            outputs = outputs + torch.randn_like(outputs, device='cuda') * noise_sd
            outputs = recon_net(outputs)
            outputs = outputs.view(-1, batch_size, noise_num, channel, h, w).mean(2, keepdim=True).reshape(batch_size,
                                                                                                           channel, h,
                                                                                                           w)
            smooth_loss_no_clean = criterion(outputs, img_original)
            smooth_loss_no_clean_mean = smooth_loss_no_clean.mean()
            smooth_losses_no_clean.update(smooth_loss_no_clean_mean.item(), img.size(0))

            if i == mark and epoch % visual_freq == 0:
                pic = to_img(outputs.cpu().data)
                save_image(pic,
                           os.path.join(outdir, 'Epoch_{epoch}_clean_NoDenoiser_smooth_Loss_{loss:.3f}.png').format(
                               epoch=epoch,
                               loss=smooth_losses_no_clean.avg))

            # ----------Clean Reconstruction Acc with DS and AE------------
            outputs = denoiser(img)
            outputs = recon_net(outputs)
            loss = criterion(outputs, img_original)
            loss_mean = loss.mean()
            losses.update(loss_mean.item(), img.size(0))

            if i == mark and epoch % visual_freq == 0:
                pic = to_img(outputs.cpu().data)
                save_image(pic, os.path.join(outdir, 'Epoch_{epoch}_clean_Loss_{loss:.3f}.png').format(epoch=epoch,
                                                                                                       loss=losses.avg))

            # ------------- clean Reconstruction Acc with DS and AE using smoothed model --------------
            outputs = img.repeat((1, noise_num, 1, 1)).view(-1, channel, h, w)
            outputs = outputs + torch.randn_like(outputs, device='cuda') * noise_sd
            outputs = denoiser(outputs)
            outputs = recon_net(outputs)
            outputs = outputs.view(-1, batch_size, noise_num, channel, h, w).mean(2, keepdim=True).reshape(batch_size,
                                                                                                           channel, h,
                                                                                                           w)
            smooth_loss_clean = criterion(outputs, img_original)
            smooth_loss_clean_mean = smooth_loss_clean.mean()
            smooth_losses_clean.update(smooth_loss_clean_mean.item(), img.size(0))

            if i == mark and epoch % visual_freq == 0:
                pic = to_img(outputs.cpu().data)
                save_image(pic,
                           os.path.join(outdir, 'Epoch_{epoch}_clean_smooth_Loss_{loss:.3f}.png').format(
                               epoch=epoch,
                               loss=smooth_losses_clean.avg))

            # ----------Adversarial Reconstruction Acc using recon-net only------------
            outputs = recon_net(adv_img)
            recon_loss = criterion(outputs, img_original)
            recon_loss_mean = recon_loss.mean()
            recon_losses.update(recon_loss_mean.item(), img.size(0))

            if i == mark and epoch % visual_freq == 0:
                pic = to_img(outputs.cpu().data)
                save_image(pic,
                           os.path.join(outdir, 'Epoch_{epoch}_adv_NoDenoiser_Loss_{loss:.3f}.png').format(
                               epoch=epoch,
                               loss=recon_losses.avg))

            # ------------- Adversarial Reconstruction Acc using smoothed reconstruction network --------------
            outputs = adv_img.repeat((1, noise_num, 1, 1)).view(-1, channel, h, w)
            outputs = outputs + torch.randn_like(outputs, device='cuda') * noise_sd
            outputs = recon_net(outputs)
            outputs = outputs.view(-1, batch_size, noise_num, channel, h, w).mean(2, keepdim=True).reshape(
                batch_size,
                channel, h,
                w)
            smooth_loss_no_adv = criterion(outputs, img_original)
            smooth_loss_no_adv_mean = smooth_loss_no_adv.mean()
            smooth_losses_no_adv.update(smooth_loss_no_adv_mean.item(), img.size(0))

            if i == mark and epoch % visual_freq == 0:
                pic = to_img(outputs.cpu().data)
                save_image(pic,
                           os.path.join(outdir, 'Epoch_{epoch}_adv_NoDenoiser_smooth_Loss_{loss:.3f}.png').format(
                               epoch=epoch,
                               loss=smooth_losses_no_adv.avg))

            # -----------Adversarial Reconstruction Acc using recon-net + denoiser--------------
            outputs = denoiser(adv_img)
            outputs = recon_net(outputs)
            adv_loss = criterion(outputs, img_original)
            adv_loss_mean = adv_loss.mean()
            adv_losses.update(adv_loss_mean.item(), img.size(0))

            if i == mark and epoch % visual_freq == 0:
                pic = to_img(outputs.cpu().data)
                save_image(pic, os.path.join(outdir, 'Epoch_{epoch}_adv_Loss_{loss:.3f}.png').format(epoch=epoch,
                                                                                                     loss=adv_losses.avg))

            # ------------- Adversarial Reconstruction Acc with DS + AE using smoothed model --------------
            outputs = adv_img.repeat((1, noise_num, 1, 1)).view(-1, channel, h, w)
            outputs = outputs + torch.randn_like(outputs, device='cuda') * noise_sd
            outputs = denoiser(outputs)
            outputs = recon_net(outputs)
            outputs = outputs.view(-1, batch_size, noise_num, channel, h, w).mean(2, keepdim=True).reshape(
                batch_size,
                channel, h,
                w)
            smooth_loss = criterion(outputs, img_original)
            smooth_loss_mean = smooth_loss.mean()
            smooth_losses.update(smooth_loss_mean.item(), img.size(0))

            if i == mark and epoch % visual_freq == 0:
                pic = to_img(outputs.cpu().data)
                save_image(pic,
                           os.path.join(outdir, 'Epoch_{epoch}_adv_smooth_Loss_{loss:.3f}.png').format(epoch=epoch,
                                                                                                       loss=smooth_losses.avg))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            log = 'Test: [{0}/{1}]\t'' \
            ''Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'' \
            ''Data {data_time.val:.3f} ({data_time.avg:.3f})\t'' \
            ''Clean Loss without Denoiser {loss_no.val:.4f} ({loss_no.avg:.4f})\t'' \
            ''Clean Loss {loss.val:.4f} ({loss.avg:.4f})\t'' \
            ''Recon_Loss {recon_loss.val:.4f} ({recon_loss.avg:.4f})\t'' \
            ''Adv_Loss {adv_loss.val:.4f} ({adv_loss.avg:.4f})\t'' \
            ''Smooth_Loss {smooth_loss.val:.4f} ({smooth_loss.avg:.4f})\n'.format(
                i, len(loader), batch_time=batch_time,
                data_time=data_time, loss_no=losses_no, loss=losses, recon_loss=recon_losses, adv_loss=adv_losses,
                smooth_loss=smooth_losses)

            print(log)

    return losses_no.avg, smooth_losses_no_clean.avg, losses.avg, smooth_losses_clean.avg, recon_losses.avg, smooth_losses_no_adv.avg, adv_losses.avg, smooth_losses.avg


def recon_train_ae(loader: DataLoader, encoder: torch.nn.Module, decoder: torch.nn.Module, denoiser: torch.nn.Module, criterion, optimizer: Optimizer, epoch: int, noise_sd: float,
          recon_net: torch.nn.Module = None):
    """
    Function for training denoiser for one epoch
        :param loader:DataLoader: training dataloader
        :param denoiser:torch.nn.Module: the denoiser being trained
        :param criterion: loss function
        :param optimizer:Optimizer: optimizer used during trainined
        :param epoch:int: the current epoch (for logging)
        :param noise_sd:float: the std-dev of the Guassian noise perturbation of the input
        :param classifier:torch.nn.Module=None: a ``freezed'' classifier attached to the denoiser
                                                (required classifciation/stability objectives), None for denoising objective
    """
    n_measurement = args.measurement
    mm_min = torch.tensor(args.data_min, dtype=torch.long).cuda()
    mm_max = torch.tensor(args.data_max, dtype=torch.long).cuda()
    mm_dis = mm_max - mm_min

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()

    # switch to train mode
    denoiser.train()

    if args.train_method == 'part':
        encoder.eval()
        decoder.eval()
    if args.train_method == 'whole':
        encoder.train()
        decoder.eval()
    if args.train_method == 'whole_plus':
        encoder.train()
        decoder.train()

    if recon_net:
        recon_net.eval()

    for i, (img_original, _) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # Obtain the Shape of Inputs (Batch_size x Channel x H x W)
        batch_size = img_original.size()[0]
        channel = img_original.size()[1]
        h = img_original.size()[2]
        w = img_original.size()[3]
        d = channel * h * w

        img_original = img_original.cuda()           # input x (batch,  channel, h, w)
        img = img_original.view(batch_size, d).cuda()

        if i ==0:
            a = measurement(n_measurement, d)    # Measurement Matrix
            print("-----------------------training--------------------")
            print(a[0, :])

        img = torch.mm(img, a)     # y = A^T x
        img = torch.mm(img, a.t())
        img = img.view(batch_size, channel, h, w)
        img = img.float()
        img = (img - mm_min) / mm_dis

        #augment inputs with noise
        noise = torch.randn_like(img, device='cuda') * noise_sd
        recon = encoder(denoiser(img + noise))

        if args.optimization_method == 'FO':
            recon = recon_net(decoder(recon))
            stab_loss = criterion(recon, img_original)

            # record original loss
            stab_loss_mean = stab_loss.mean()
            losses.update(stab_loss_mean.item(), img_original.size(0))

            # compute gradient and do step
            optimizer.zero_grad()
            stab_loss_mean.backward()
            optimizer.step()

        elif args.optimization_method == 'ZO':
            recon.requires_grad_(True)
            recon.retain_grad()

            with torch.no_grad():
                mu = torch.tensor(args.mu).cuda()
                q = torch.tensor(args.q).cuda()

                # Forward Inference (Original)
                original_recon = recon_net(img)

                recon_test = recon_net(decoder(recon))
                loss_0 = criterion(recon_test, original_recon)
                # record original loss
                loss_0_mean = loss_0.mean()
                losses.update(loss_0_mean.item(), img_original.size(0))

                recon_flat_no_grad = torch.flatten(recon, start_dim=1).cuda()
                grad_est = torch.zeros(batch_size, d).cuda()

                # ZO Gradient Estimation
                for k in range(d):
                    # Obtain a direction vector (1-0)
                    u = torch.zeros(batch_size, d).cuda()
                    u[:, k] = 1

                    # Forward Inference (reconstructed image + random direction vector)
                    recon_q_plus = recon_flat_no_grad + mu * u
                    recon_q_minus = recon_flat_no_grad - mu * u

                    recon_q_plus = recon_q_plus.view(batch_size, channel, h, w)
                    recon_q_minus = recon_q_minus.view(batch_size, channel, h, w)
                    recon_q_pre_plus = recon_net(decoder(recon_q_plus))
                    recon_q_pre_minus = recon_net(decoder(recon_q_minus))

                    # Loss Calculation and Gradient Estimation
                    loss_tmp_plus = criterion(recon_q_pre_plus, original_recon)
                    loss_tmp_minus = criterion(recon_q_pre_minus, original_recon)

                    loss_diff = torch.tensor(loss_tmp_plus - loss_tmp_minus)
                    loss_diff = loss_diff.mean(3, keepdim=True).mean(2, keepdim=True)
                    grad_est = grad_est + u * loss_diff.reshape(batch_size, 1).expand_as(u) / (2 * mu)

            recon_flat = torch.flatten(recon, start_dim=1).cuda()
            grad_est_no_grad = grad_est.detach()

            # reconstructed image * gradient estimation   <--   g(x) * a
            loss = torch.sum(recon_flat * grad_est_no_grad, dim=-1).mean()  # l_mean
            # compute gradient and do step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            log = 'Epoch: [{0}][{1}/{2}]\t'' \
            ''Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'' \
            ''Data {data_time.val:.3f} ({data_time.avg:.3f})\t'' \
            ''Stab_Loss {stab_loss.val:.4f} ({stab_loss.avg:.4f})''\n'.format(
                epoch, i, len(loader), batch_time=batch_time,
                data_time=data_time, stab_loss = losses)

            print(log)

    return losses.avg


def test_with_recon_ae(loader: DataLoader, encoder: torch.nn.Module, decoder: torch.nn.Module, denoiser: torch.nn.Module, criterion, outdir:str, noise_sd: float, epoch: int, visual_freq: int, noise_num: int, num_steps: int, epsilon: int, print_freq: int, recon_net: torch.nn.Module):
    """
    A function to test the classification performance of a denoiser when attached to a given classifier
        :param loader:DataLoader: test dataloader
        :param denoiser:torch.nn.Module: the denoiser
        :param criterion: the loss function (e.g. MAE)
        :param noise_sd:float: the std-dev of the Guassian noise perturbation of the input
        :param print_freq:int: the frequency of logging
        :param recon_net:torch.nn.Module: the reconstruction network to which the denoiser is attached
    """

    # switch to eval mode
    recon_net.eval()
    encoder.eval()
    decoder.eval()
    if denoiser:
        denoiser.eval()

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_no = AverageMeter()
    losses = AverageMeter()
    recon_losses = AverageMeter()
    adv_losses = AverageMeter()
    smooth_losses = AverageMeter()
    smooth_losses_no_clean = AverageMeter()
    smooth_losses_clean = AverageMeter()
    smooth_losses_no_adv = AverageMeter()
    end = time.time()

    n_measurement = args.measurement
    mm_min = torch.tensor(args.data_min, dtype=torch.long).cuda()
    mm_max = torch.tensor(args.data_max, dtype=torch.long).cuda()
    mm_dis = mm_max - mm_min

    mark = 39

    attacker = recon_PGD_L2(steps=num_steps, device='cuda', max_norm=epsilon)  # remember epsilon/256

    # switch to eval mode
    recon_net.eval()
    if denoiser:
        denoiser.eval()

    for i, (img_original, _) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        img_original = img_original.cuda()

        # Obtain the Shape of Inputs (Batch_size x Channel x H x W)
        batch_size = img_original.size()[0]
        channel = img_original.size()[1]
        h = img_original.size()[2]
        w = img_original.size()[3]
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


        # adv_img = attacker.attack(model, img, img_original, criterion)
        adv_img = attacker.attack(recon_net, img, img_original, criterion)

        # augment inputs with noise
        # inputs = img + torch.randn_like(img, device='cuda') * noise_sd

        with torch.no_grad():
            # ----------Clean Reconstruction Acc without denoiser------------
            outputs = recon_net(img)
            loss_no = criterion(outputs, img_original)
            loss_no_mean = loss_no.mean()
            losses_no.update(loss_no_mean.item(), img.size(0))

            if i == mark and epoch % visual_freq == 0:
                pic = to_img(outputs.cpu().data)
                save_image(pic,
                           os.path.join(outdir, 'Epoch_{epoch}_clean_NoDenoiser_Loss_{loss:.3f}.png').format(
                               epoch=epoch,
                               loss=losses_no.avg))

            # ------------- clean Reconstruction Acc using smoothed reconstruction network --------------
            outputs = img.repeat((1, noise_num, 1, 1)).view(-1, channel, h, w)
            outputs = outputs + torch.randn_like(outputs, device='cuda') * noise_sd
            outputs = recon_net(outputs)
            outputs = outputs.view(-1, batch_size, noise_num, channel, h, w).mean(2, keepdim=True).reshape(batch_size,
                                                                                                           channel, h,
                                                                                                           w)
            smooth_loss_no_clean = criterion(outputs, img_original)
            smooth_loss_no_clean_mean = smooth_loss_no_clean.mean()
            smooth_losses_no_clean.update(smooth_loss_no_clean_mean.item(), img.size(0))

            if i == mark and epoch % visual_freq == 0:
                pic = to_img(outputs.cpu().data)
                save_image(pic,
                           os.path.join(outdir, 'Epoch_{epoch}_clean_NoDenoiser_smooth_Loss_{loss:.3f}.png').format(
                               epoch=epoch,
                               loss=smooth_losses_no_clean.avg))

            # ----------Clean Reconstruction Acc with DS and AE------------
            outputs = encoder(denoiser(img))
            outputs = recon_net(decoder(outputs))
            loss = criterion(outputs, img_original)
            loss_mean = loss.mean()
            losses.update(loss_mean.item(), img.size(0))

            if i == mark and epoch % visual_freq == 0:
                pic = to_img(outputs.cpu().data)
                save_image(pic, os.path.join(outdir, 'Epoch_{epoch}_clean_Loss_{loss:.3f}.png').format(epoch=epoch,
                                                                                                       loss=losses.avg))

            # ------------- clean Reconstruction Acc with DS and AE using smoothed model --------------
            outputs = img.repeat((1, noise_num, 1, 1)).view(-1, channel, h, w)
            outputs = outputs + torch.randn_like(outputs, device='cuda') * noise_sd
            outputs = encoder(denoiser(outputs))
            outputs = recon_net(decoder(outputs))
            outputs = outputs.view(-1, batch_size, noise_num, channel, h, w).mean(2, keepdim=True).reshape(batch_size,
                                                                                                           channel, h,
                                                                                                           w)
            smooth_loss_clean = criterion(outputs, img_original)
            smooth_loss_clean_mean = smooth_loss_clean.mean()
            smooth_losses_clean.update(smooth_loss_clean_mean.item(), img.size(0))

            if i == mark and epoch % visual_freq == 0:
                pic = to_img(outputs.cpu().data)
                save_image(pic,
                           os.path.join(outdir, 'Epoch_{epoch}_clean_smooth_Loss_{loss:.3f}.png').format(
                               epoch=epoch,
                               loss=smooth_losses_clean.avg))

            # ----------Adversarial Reconstruction Acc using recon-net only------------
            outputs = recon_net(adv_img)
            recon_loss = criterion(outputs, img_original)
            recon_loss_mean = recon_loss.mean()
            recon_losses.update(recon_loss_mean.item(), img.size(0))

            if i == mark and epoch % visual_freq == 0:
                pic = to_img(outputs.cpu().data)
                save_image(pic,
                           os.path.join(outdir, 'Epoch_{epoch}_adv_NoDenoiser_Loss_{loss:.3f}.png').format(
                               epoch=epoch,
                               loss=recon_losses.avg))

            # ------------- Adversarial Reconstruction Acc using smoothed reconstruction network --------------
            outputs = adv_img.repeat((1, noise_num, 1, 1)).view(-1, channel, h, w)
            outputs = outputs + torch.randn_like(outputs, device='cuda') * noise_sd
            outputs = recon_net(outputs)
            outputs = outputs.view(-1, batch_size, noise_num, channel, h, w).mean(2, keepdim=True).reshape(
                batch_size,
                channel, h,
                w)
            smooth_loss_no_adv = criterion(outputs, img_original)
            smooth_loss_no_adv_mean = smooth_loss_no_adv.mean()
            smooth_losses_no_adv.update(smooth_loss_no_adv_mean.item(), img.size(0))

            if i == mark and epoch % visual_freq == 0:
                pic = to_img(outputs.cpu().data)
                save_image(pic,
                           os.path.join(outdir, 'Epoch_{epoch}_adv_NoDenoiser_smooth_Loss_{loss:.3f}.png').format(
                               epoch=epoch,
                               loss=smooth_losses_no_adv.avg))

            # -----------Adversarial Reconstruction Acc using recon-net + denoiser--------------
            outputs = encoder(denoiser(adv_img))
            outputs = recon_net(decoder(outputs))
            adv_loss = criterion(outputs, img_original)
            adv_loss_mean = adv_loss.mean()
            adv_losses.update(adv_loss_mean.item(), img.size(0))

            if i == mark and epoch % visual_freq == 0:
                pic = to_img(outputs.cpu().data)
                save_image(pic, os.path.join(outdir, 'Epoch_{epoch}_adv_Loss_{loss:.3f}.png').format(epoch=epoch,
                                                                                                     loss=adv_losses.avg))

            # ------------- Adversarial Reconstruction Acc with DS + AE using smoothed model --------------
            outputs = adv_img.repeat((1, noise_num, 1, 1)).view(-1, channel, h, w)
            outputs = outputs + torch.randn_like(outputs, device='cuda') * noise_sd
            outputs = encoder(denoiser(outputs))
            outputs = recon_net(decoder(outputs))
            outputs = outputs.view(-1, batch_size, noise_num, channel, h, w).mean(2, keepdim=True).reshape(
                batch_size,
                channel, h,
                w)
            smooth_loss = criterion(outputs, img_original)
            smooth_loss_mean = smooth_loss.mean()
            smooth_losses.update(smooth_loss_mean.item(), img.size(0))

            if i == mark and epoch % visual_freq == 0:
                pic = to_img(outputs.cpu().data)
                save_image(pic,
                           os.path.join(outdir, 'Epoch_{epoch}_adv_smooth_Loss_{loss:.3f}.png').format(epoch=epoch,
                                                                                                       loss=smooth_losses.avg))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            log = 'Test: [{0}/{1}]\t'' \
            ''Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'' \
            ''Data {data_time.val:.3f} ({data_time.avg:.3f})\t'' \
            ''Clean Loss without Denoiser {loss_no.val:.4f} ({loss_no.avg:.4f})\t'' \
            ''Clean Loss {loss.val:.4f} ({loss.avg:.4f})\t'' \
            ''Recon_Loss {recon_loss.val:.4f} ({recon_loss.avg:.4f})\t'' \
            ''Adv_Loss {adv_loss.val:.4f} ({adv_loss.avg:.4f})\t'' \
            ''Smooth_Loss {smooth_loss.val:.4f} ({smooth_loss.avg:.4f})\n'.format(
                i, len(loader), batch_time=batch_time,
                data_time=data_time, loss_no=losses_no, loss=losses, recon_loss=recon_losses, adv_loss=adv_losses,
                smooth_loss=smooth_losses)

            print(log)

    return losses_no.avg, smooth_losses_no_clean.avg, losses.avg, smooth_losses_clean.avg, recon_losses.avg, smooth_losses_no_adv.avg, adv_losses.avg, smooth_losses.avg


def to_img(x):
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

def frozen_module(module):
    for param in module.parameters():
            param.requires_grad = False


if __name__ == "__main__":
    main()
