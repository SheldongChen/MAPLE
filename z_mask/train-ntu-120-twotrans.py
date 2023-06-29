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

from datasets.ntu120 import NTU60Subject
import models.ntu_mae as Models
import random
from models.ntu_mae import CalculateIndex

base_norm = None
def MSE_Norm(embedding_xyz, mask_index):
    criterion_mse = nn.MSELoss()
    return criterion_mse(embedding_xyz[:, mask_index, :], torch.zeros_like(embedding_xyz[:, mask_index, :]).detach())

def Cls_Loss(cls_restruct_xyz, cls_clone_xyz, is_detach=True):
    if is_detach:
        cls_clone_xyz = cls_clone_xyz.detach()
    bs, dim_t, dim_xyz = cls_clone_xyz.shape
    cls_restruct_xyz = cls_restruct_xyz.reshape(bs*dim_t, dim_xyz)
    cls_clone_xyz = cls_clone_xyz.reshape(bs*dim_t, dim_xyz)

    cls_restruct_xyz_logsoft = F.log_softmax(cls_restruct_xyz, dim=-1)
    cls_clone_xyz_soft = F.softmax(cls_clone_xyz, dim=-1)
    loss = F.kl_div(cls_restruct_xyz_logsoft, cls_clone_xyz_soft, reduction='batchmean')
    return loss
def train_one_epoch(model, model_k, criterion, optimizer, lr_scheduler, data_loader, data_loader_unlabel, device, epoch, print_freq, alpha=0.1, mae_only_ul=False):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('clips/s', utils.SmoothedValue(window_size=10, fmt='{value:.3f}'))

    header = 'Epoch: [{}]'.format(epoch)
    iter_unlabeled = iter(data_loader_unlabel)
    # global iter_k
    count = 0
    for clip, target, _ in metric_logger.log_every(data_loader, print_freq, header):
        # if iter_k < 0:
        #     # warmup 100 iter
        #     m = 0.0
        #     _momentum_update_key_encoder(model, model_k, m=m)
        #     if abs(iter_k) % 10 == 0:
        #         print('warmup model_k with model, m ={}, iter= {}'.format(m, iter_k))
        #     iter_k += 1
        # elif iter_k == 10:
        #     m = 0.0
        #     _momentum_update_key_encoder(model, model_k, m=m)
        #     print('update model_k with model, m ={}'.format(m))
        #     iter_k = 0
        # else:
        #     iter_k += 1
        start_time = time.time()
        # clip, target = clip.to(device), target.to(device)
        #print(clip.shape)
        #clip =  clip[:-1]
        try:
            output = model(clip, forward_type='cls')
        except Exception as e:
            print(e)
            print(clip.shape)
            #print(clip)
            #breakpoint()
        token_length = 12 #int(token_length[0])
        loss = criterion(output, target)

        outputs_unlabeled = None
        # breakpoint()
        sample_index, mask_index = CalculateIndex(token_length, model.module.mask_ratio)
        # breakpoint()
        if epoch >= 0:
            try:
                clip_unlabeled, _, _ = next(iter_unlabeled)
            except BaseException:
                iter_unlabeled = iter(data_loader_unlabel)
                clip_unlabeled, _, _ = next(iter_unlabeled)
            if mae_only_ul:
                restruct_embedding_xyz, embedding_xyz_clone = model(clip_unlabeled, forward_type='mask', sample_index=sample_index, mask_index=mask_index)
                # with torch.no_grad():
                #     _, embedding_xyz_freeze, _ = model_k(clip_unlabeled, forward_type='mask')
            else:
                #breakpoint()
                restruct_embedding_xyz, embedding_xyz_clone = model(torch.cat([clip_unlabeled, clip], dim=0), forward_type='mask', sample_index=sample_index, mask_index=mask_index)
                #breakpoint()
                # with torch.no_grad():
                #     _, embedding_xyz_freeze, _ = model_k(torch.cat([clip_unlabeled, clip], dim=0), forward_type='mask')
            if isinstance(model, torch.nn.DataParallel):
                cls_restruct_xyz = model.module.mlp_head(restruct_embedding_xyz[:, mask_index, :])
                cls_clone_xyz = model.module.mlp_head(embedding_xyz_clone[:, mask_index, :])
            else:
                cls_restruct_xyz = model.mlp_head(restruct_embedding_xyz[:, mask_index, :])
                cls_clone_xyz = model.mlp_head(embedding_xyz_clone[:, mask_index, :])


            # set base_norm
            norm_restruct_embedding_xyz = MSE_Norm(restruct_embedding_xyz.detach(), mask_index)
            norm_embedding_xyz_clone = MSE_Norm(embedding_xyz_clone.detach(), mask_index)
            # norm_embedding_xyz_freeze = MSE_Norm(embedding_xyz_freeze, mask_index)
            global base_norm
            #global type_MSE
            if base_norm == None:
                base_norm = norm_embedding_xyz_clone.item()
            # if norm_embedding_xyz_clone.item() > 1.5*base_norm and type_MSE != 'clone':
            #     type_MSE = 'clone'
            #     print('set MSE clone')
            # elif norm_embedding_xyz_clone.item() < 0.5*base_norm and type_MSE != 'detach':
            #     type_MSE = 'detach'
            #     print('set MSE detach')
            # else:
            #     pass
            #loss_norm = MSE_Norm_limit(restruct_embedding_xyz, mask_index, base_norm) + MSE_Norm_limit(embedding_xyz_clone, mask_index, base_norm)
            # if type_MSE == 'detach':
            #     loss_mse = MSE_Loss(restruct_embedding_xyz, embedding_xyz_clone.detach(), mask_index)
            # elif type_MSE == 'clone':
            #     loss_mse = MSE_Loss(restruct_embedding_xyz, embedding_xyz_clone, mask_index)
            # else:
            #     assert False, 'type_MSE must in ["detach", "clone"]'
            # if type_MSE == 'clone':
            #     metric_logger.update(MSE_clone=1.0)
            # else:
            #     metric_logger.update(MSE_clone=0.0)
            # global iter_k
            # if iter_k < 0:
            #     if abs(iter_k)%10==0:
            #         print('warmup with MSE Loss')
            #     loss_mse = 10*Cos_Loss(restruct_embedding_xyz, embedding_xyz_clone.detach(), mask_index)*(100+iter_k)/100 + 2*MSE_Loss(restruct_embedding_xyz, embedding_xyz_clone.detach(), mask_index)*(-iter_k)/100
            #     # loss_mse =10*Cos_Loss(restruct_embedding_xyz, embedding_xyz_clone, mask_index)*(100+iter_k)/100 + MSE_Loss(restruct_embedding_xyz, embedding_xyz_clone, mask_index)*(-iter_k)/100
            #     iter_k += 1
            # else:
            #     if iter_k==0:
            #         print('train with Cos Loss')
            #     iter_k = 1

            #loss_mse = Cos_Loss(restruct_embedding_xyz, embedding_xyz_clone, mask_index, is_detach=True)
            #loss_tri = Triple_loss(restruct_embedding_xyz, embedding_xyz_clone, mask_index, is_detach=True, m=2)
            # loss_mse = Easy_MSE_Loss(restruct_embedding_xyz, embedding_xyz_clone.detach(), mask_index)
            # loss_mse = MSE_Loss(restruct_embedding_xyz, embedding_xyz_freeze, mask_index)
            # loss_mse = loss_mse + MSE_Loss(embedding_xyz_clone, embedding_xyz_freeze, mask_index)
            loss_mse = Cls_Loss(cls_restruct_xyz, cls_clone_xyz)
            weigh_decay_norm = 0.0 #float(1e-4)
            loss_norm = 0.0 # F.relu(MSE_Norm(embedding_xyz_clone, mask_index))*weigh_decay_norm

            metric_logger.update(base_norm=base_norm)
            #metric_logger.update(loss_norm=loss_norm.clone().detach().item())
           # metric_logger.update(loss_mse_a=loss_mse_a.clone().detach().item())
           # metric_logger.update(loss_mse_b=loss_mse_b.clone().detach().item())
            #metric_logger.update(loss_norm_mean=loss_norm.clone().detach().item())
            #metric_logger.update(loss_tri=loss_tri.clone().detach().item())
            metric_logger.update(use_cls_loss=1.0)
            metric_logger.update(loss_mse=loss_mse.clone().detach().item())
            metric_logger.update(loss_cls=loss.clone().detach().item())
            metric_logger.update(mse_norm_restruct=norm_restruct_embedding_xyz.item())
            metric_logger.update(mse_norm_orgin=norm_embedding_xyz_clone.item())
            metric_logger.update(max_cls_orgin=cls_clone_xyz.detach().max(dim=-1)[0].mean().item())
            metric_logger.update(max_cls_restruct=cls_restruct_xyz.detach().max(dim=-1)[0].mean().item())
            metric_logger.update(len_mask=len(mask_index))
            metric_logger.update(len_sample=len(sample_index))
            # metric_logger.update(mse_norm_freeze=norm_embedding_xyz_freeze.item())
            loss = loss + alpha * (loss_mse+loss_norm)   #+ loss_norm)



        #restruct_embedding_xyz, embedding_xyz_clone, mask_index, output_cls = model(clip, forward_type='both')



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

    random.seed(args.seed)
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
            train=True,

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
                                              num_workers=args.workers, pin_memory=True, drop_last=True)

    data_loader_unlabel = DataLoaderX(device, dataset=dataset_unlabel, batch_size=args.batch_size_unlabel, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    # data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    data_loader_test = DataLoaderX(device, dataset=dataset_test, batch_size=64, num_workers=args.workers, pin_memory=True)

    print("Creating model")
    Model = getattr(Models, args.model)
    mask_ratio = float(int(args.mask_length)/12)
    model = Model(radius=args.radius, nsamples=args.nsamples, spatial_stride=args.spatial_stride,
                  temporal_kernel_size=args.temporal_kernel_size, temporal_stride=args.temporal_stride,
                  emb_relu=args.emb_relu,
                  dim=args.dim, depth_xyz=args.depth_xyz, depth_t=args.depth_t, heads=args.heads, dim_head=args.dim_head,
                  mlp_dim=args.mlp_dim, num_classes=dataset.num_classes,
                  only_cls=False, depth_t_decoder=args.decoder_depth_t, dim_decoder=512, mask_ratio=mask_ratio)
    print('model.mask_ratio = ', model.mask_ratio)

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
        try:
            model_without_ddp.load_state_dict(checkpoint['model'])
        except Exception as e:
            print(e)
            print('load with not strict...')
            model_without_ddp.load_state_dict(checkpoint['model'], strict=False)


    # resume from checkpoint
    if args.resume_checkpoint:
        checkpoint = torch.load(args.resume_checkpoint, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    model_k = None
    print("Start training")
    start_time = time.time()
    acc = 0
    best_acc = 0

    if args.eval_before_train:
        print('*' * 60)
        print('With Eval_before_train !!!')
        print('*' * 60)
        acc = max(acc, evaluate(model, criterion, data_loader_test, device=device))
        #best_acc = acc
        if args.output_dir:
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': -1,
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
                    os.path.join(output_dir, 'model_best.pth'))
        print('Before Train, Accuracy {}'.format(acc))
    else:
        print('*' * 60)
        print('Without Eval_before_train !!!')
        print('*' * 60)

    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(model, model_k, criterion, optimizer, lr_scheduler, data_loader, data_loader_unlabel, device, epoch, args.print_freq, alpha=args.loss_alpha, mae_only_ul=args.mae_only_ul)

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
    parser.add_argument('--data-meta', default='./datasets/ntu/ntu120_half_cs_026.list', help='dataset')
    parser.add_argument('--data-meta-unlabel', default='./datasets/ntu/ntu120_half_cs_026_unlabel.list', help='dataset')
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
    parser.add_argument('-bu', '--batch-size-unlabel', default=32, type=int)
    parser.add_argument('--epochs', default=20, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--lr-milestones', nargs='+', default=[11, 19], type=int, help='decrease lr on milestones')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--lr-warmup-epochs', default=2, type=int, help='number of warmup epochs')
    parser.add_argument('--eval-before-train', default=True, action='store_false')
    # output
    parser.add_argument('--print-freq', default=200, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='output/ntu120_twotrans_mae', type=str, help='path where to save')
    # resume
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--resume-checkpoint', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='start epoch')
    # mae
    # parser.add_argument('--loss-annealing', nargs='+', default=[0, 18], type=int, help='epoch of annealing')
    parser.add_argument('--loss-alpha', default=0.1, type=float, help='alpha for unlabel data loss')
    parser.add_argument('--mae-only-ul', default=False, action='store_true')
    parser.add_argument('--decoder-depth-t', default=8, type=int, help='transformer depth of decoder')
    parser.add_argument('--mask-length', default=9, type=int, help='mask length')
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
