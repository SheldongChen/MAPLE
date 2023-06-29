from __future__ import print_function
import datetime
import os
import time
import sys
sys.path.append('.')
import numpy as np
import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from data_utils import DataLoaderX
import utils

from scheduler import WarmupMultiStepLR

from datasets.ntu60 import NTU60Subject
import models.msr_twotrans as Models


def joint_loss(preds_labeled, preds_unlabeled, labels, epoch, T1=0, T2=8, alpha=2):
    train_criterion = nn.CrossEntropyLoss()
    loss_unlabeled = 0.0
    if preds_unlabeled is not None:
        pseudo_labels = preds_unlabeled.argmax(dim=1)
        loss_unlabeled = train_criterion(preds_unlabeled, pseudo_labels)
        if epoch >= T2:
            weight = alpha
        else:
            weight = alpha * (epoch - T1 + 0.5) / (T2 - T1)
        loss_unlabeled = weight * loss_unlabeled

    loss_labeled = train_criterion(preds_labeled, labels)
    loss = loss_labeled + loss_unlabeled
    return loss


def train_one_epoch(model, criterion, optimizer, lr_scheduler, data_loader, data_loader_unlabel, device, epoch, print_freq, T1, T2 , alpha=2):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('clips/s', utils.SmoothedValue(window_size=10, fmt='{value:.3f}'))

    header = 'Epoch: [{}]'.format(epoch)
    iter_unlabeled = iter(data_loader_unlabel)

    for clip, target, _ in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()
        # clip, target = clip.to(device), target.to(device)
        output = model(clip)
        outputs_unlabeled = None
        if epoch >= T1:
            try:
                clip_unlabeled, _, _ = next(iter_unlabeled)
            except BaseException:
                iter_unlabeled = iter(data_loader_unlabel)
                clip_unlabeled, _, _ = next(iter_unlabeled)
            outputs_unlabeled = model(clip_unlabeled)
        loss = criterion(output, outputs_unlabeled, target, epoch, T1=T1, T2=T2, alpha=alpha)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = clip.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        metric_logger.meters['clips/s'].update(batch_size / (time.time() - start_time))
        lr_scheduler.step()
        sys.stdout.flush()

    data_loader.shutdown()
    data_loader_unlabel.shutdown()

def evaluate(model, criterion, data_loader, device):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    video_prob = {}
    video_label = {}
    with torch.no_grad():
        for clip, target, video_idx in metric_logger.log_every(data_loader, 1000, header):
            # clip = clip.to(device, non_blocking=True)
            # target = target.to(device, non_blocking=True)
            output = model(clip)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            prob = F.softmax(input=output, dim=1)


            # could have been padded in distributed setup
            batch_size = clip.shape[0]
            target = target.cpu().numpy()
            video_idx = video_idx.cpu().numpy()
            prob = prob.cpu().numpy()
            for i in range(0, batch_size):
                idx = video_idx[i]
                if idx in video_prob:
                    video_prob[idx] += prob[i]
                else:
                    video_prob[idx] = prob[i]
                    video_label[idx] = target[i]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print(' * Clip Acc@1 {top1.global_avg:.3f} Clip Acc@5 {top5.global_avg:.3f}'.format(top1=metric_logger.acc1, top5=metric_logger.acc5))

    # video level prediction
    video_pred = {k: np.argmax(v) for k, v in video_prob.items()}
    pred_correct = [video_pred[k]==video_label[k] for k in video_pred]
    total_acc = np.mean(pred_correct)

    class_count = [0] * data_loader.dataset.num_classes
    class_correct = [0] * data_loader.dataset.num_classes

    for k, v in video_pred.items():
        label = video_label[k]
        class_count[label] += 1
        class_correct[label] += (v==label)
    class_acc = [c/float(s) for c, s in zip(class_correct, class_count)]

    print(' * Video Acc@1 %f'%total_acc)
    print(' * Class Acc@1 %s'%str(class_acc))

    return total_acc


def main(args):

    if args.output_dir:
        utils.mkdir(args.output_dir)
    output_dir = args.output_dir
    print(args)
    print("torch version: ", torch.__version__)
    print("torchvision version: ", torchvision.__version__)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda:0')

    # Data loading code
    print("Loading data")

    st = time.time()

    dataset = NTU60Subject(
            root=args.data_path,
            meta=args.data_meta,
            frames_per_clip=args.clip_len,
            step_between_clips=args.frame_step,
            num_points=args.num_points,
            train=True
    )

    dataset_unlabel = NTU60Subject(
            root=args.data_path,
            meta=args.data_meta_unlabel,
            frames_per_clip=args.clip_len,
            step_between_clips=args.frame_step,
            num_points=args.num_points,
            train=True
    )

    dataset_test = NTU60Subject(
            root=args.data_path,
            meta=args.data_meta,
            frames_per_clip=args.clip_len,
            step_between_clips=args.frame_step,
            num_points=args.num_points,
            train=False
    )
    print(len(dataset.videos), len(dataset_test.videos))
    print("Creating data loaders")

    # data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    data_loader = DataLoaderX(device, dataset=dataset, batch_size=args.batch_size, shuffle=True,
                                              num_workers=args.workers, pin_memory=True)

    data_loader_unlabel = DataLoaderX(device, dataset=dataset_unlabel, batch_size=args.batch_size_unlabel, shuffle=True,
                                              num_workers=args.workers, pin_memory=True)
    # data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    data_loader_test = DataLoaderX(device, dataset=dataset_test, batch_size=64, num_workers=args.workers, pin_memory=True)

    print("Creating model")
    Model = getattr(Models, args.model)
    model = Model(radius=args.radius, nsamples=args.nsamples, spatial_stride=args.spatial_stride,
                  temporal_kernel_size=args.temporal_kernel_size, temporal_stride=args.temporal_stride,
                  emb_relu=args.emb_relu,
                  dim=args.dim, depth_xyz=args.depth_xyz, depth_t=args.depth_t, heads=args.heads, dim_head=args.dim_head,
                  mlp_dim=args.mlp_dim, num_classes=dataset.num_classes)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    lr = args.lr
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # convert scheduler to be per iteration, not per epoch, for warmup that lasts
    # between different epochs
    warmup_iters = args.lr_warmup_epochs * len(data_loader)
    lr_milestones = [len(data_loader) * m for m in args.lr_milestones]
    lr_scheduler = WarmupMultiStepLR(optimizer, milestones=lr_milestones, gamma=args.lr_gamma, warmup_iters=warmup_iters, warmup_factor=1e-5)

    model_without_ddp = model

    #resume from pretrain
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        # args.start_epoch = checkpoint['epoch'] + 1

    # resume from checkpoint
    if args.resume_checkpoint:
        checkpoint = torch.load(args.resume_checkpoint, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1


    print("Start training")
    start_time = time.time()
    acc = 0
    best_acc = 0
    acc = max(acc, evaluate(model, criterion, data_loader_test, device=device))

    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(model, joint_loss, optimizer, lr_scheduler, data_loader, data_loader_unlabel, device, epoch, args.print_freq, T1=args.loss_annealing[0], T2=args.loss_annealing[1], alpha=args.loss_alpha)

        acc = max(acc, evaluate(model, criterion, data_loader_test, device=device))

        if args.output_dir:
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args}
            # utils.save_on_master(
            #     checkpoint,
            #     os.path.join(output_dir, 'model_{}.pth'.format(epoch)))
            utils.save_on_master(
                checkpoint,
                os.path.join(output_dir, 'checkpoint.pth'))
            if acc > best_acc:
                best_acc = acc
                utils.save_on_master(
                    checkpoint,
                    os.path.join(output_dir, 'model_best.pth'.format(epoch)))
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    print('Accuracy {}'.format(acc))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='P4Transformer Model Training')

    parser.add_argument('--data-path', default='./data/ntu/npy_faster/point_reduce_without_sample', type=str, help='dataset')
    parser.add_argument('--data-meta', default='./datasets/ntu/ntu60_half_cs_033.list', help='dataset')
    parser.add_argument('--data-meta-unlabel', default='./datasets/ntu/ntu60_half_cs_033_unlabel.list', help='dataset')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--model', default='P4Transformer', type=str, help='model')
    # input
    parser.add_argument('--clip-len', default=24, type=int, metavar='N', help='number of frames per clip')
    parser.add_argument('--frame-step', default=1, type=int, metavar='N', help='steps between frame sampling')
    parser.add_argument('--num-points', default=2048, type=int, metavar='N', help='number of points per frame')
    # P4D
    parser.add_argument('--radius', default=0.1, type=float, help='radius for the ball query')
    parser.add_argument('--nsamples', default=32, type=int, help='number of neighbors for the ball query')
    parser.add_argument('--spatial-stride', default=32, type=int, help='spatial subsampling rate')
    parser.add_argument('--temporal-kernel-size', default=3, type=int, help='temporal kernel size')
    parser.add_argument('--temporal-stride', default=2, type=int, help='temporal stride')
    # embedding
    parser.add_argument('--emb-relu', default=True, action='store_false')
    # transformer
    parser.add_argument('--dim', default=1024, type=int, help='transformer dim')
    parser.add_argument('--depth-xyz', default=4, type=int, help='transformer depth')
    parser.add_argument('--depth-t', default=3, type=int, help='transformer depth')
    parser.add_argument('--heads', default=8, type=int, help='transformer head')
    parser.add_argument('--dim-head', default=128, type=int, help='transformer dim for each head')
    parser.add_argument('--mlp-dim', default=2048, type=int, help='transformer mlp dim')
    # training
    parser.add_argument('-b', '--batch-size', default=32, type=int)
    parser.add_argument('-bu', '--batch-size-unlabel', default=64, type=int)
    parser.add_argument('--epochs', default=30, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--lr-milestones', nargs='+', default=[25, 30], type=int, help='decrease lr on milestones')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--lr-warmup-epochs', default=0, type=int, help='number of warmup epochs')
    parser.add_argument('--loss-annealing', nargs='+', default=[0, 20], type=int, help='epoch of annealing')
    parser.add_argument('--loss-alpha', default=1, type=int, help='alpha for unlabel data loss')
    # output
    parser.add_argument('--print-freq', default=100, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='output/ntu_twotrans_pseudo', type=str, help='path where to save')
    # resume
    parser.add_argument('--resume', default='', help='resume from pretrain')
    parser.add_argument('--resume-checkpoint', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='start epoch')

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
