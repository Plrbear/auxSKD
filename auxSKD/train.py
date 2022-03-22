import os
import builtins

import sys
import time
import argparse
import socket
import random

import numpy as np
from collections import OrderedDict
from tools import get_logger
import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader
from models import r21d, r3d, c3d, s3d_g
from datasets.ucf101 import ucf101_pace_pretrain
from utils.video_transforms import RandomCrop, RandomHorizontalFlip, CenterCrop, ClipResize, ToTensor
from tensorboardX import SummaryWriter
from util import adjust_learning_rate, AverageMeter
import torch.backends.cudnn as cudnn
import torch.nn.functional as F



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0,1,2,3', help='gpu id')
    parser.add_argument('--height', type=int, default=256, help='resize height')
    parser.add_argument('--width', type=int, default=256, help='resize width')
    parser.add_argument('--clip_len', type=int, default=64, help='64, input clip length')
    parser.add_argument('--crop_sz', type=int, default=224, help='crop size')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--bs', type=int, default=32, help='32, batch size')
    parser.add_argument('--num_workers', type=int, default=6, help='num of workers')
    parser.add_argument('--epochs', type=int, default=25, help='total epoch')
    parser.add_argument('--max_sr', type=int, default=4, help='largest sampling rate')
    parser.add_argument('--max_segment', type=int, default=4, help='largest segments')
    parser.add_argument('--num_classes', type=int, default=4, help='num of classes for speed rate')
    parser.add_argument('--num_classes_segment', type=int, default=128, help='num of classes for segment')

    parser.add_argument('--max_save', type=int, default=30, help='max save epoch num')
    parser.add_argument('--dataset', type=str, default='ucf101', help='ucf101/k400')
    parser.add_argument('--pf', type=int, default=40, help='print frequency')
    parser.add_argument('--model', type=str, default='r21d', help='s3d/r21d/r3d/c3d, pretrain model')
    parser.add_argument('--data_list', type=str, default='/home/amir/DATA/k100.list', help='data list')
    parser.add_argument('--rgb_prefix', type=str, default='/home/amir/DATA/k100/', help='dataset dir')

    parser.add_argument('--print_freq', type=int, default=60, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=1, help='save frequency')
    parser.add_argument('--temp_t', type=float, default=0.1)

    parser.add_argument('--lr_decay_epochs', type=int, default=90, help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2, help='decay rate for learning rate')
    parser.add_argument('--cos', action='store_true',
                        help='whether to cosine learning rate or not')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--sgd_momentum', type=float, default=0.9, help='SGD momentum')

    parser.add_argument('--temp', type=float, default=0.02)
    parser.add_argument('--momentum', type=float, default=0.999)


    parser.add_argument('--checkpoint_path', default='/home/amir/isd_chp', type=str,
                        help='where to save checkpoints. ')
    parser.add_argument('--resume', default='/home/amir/isd_chp/pckpt_epoch_10.pth', type=str,
                        help='path to latest checkpoint (default: none)')

    args = parser.parse_args()

    return args



class KLD(nn.Module):
    def forward(self, inputs, targets):
        inputs = F.log_softmax(inputs, dim=1)
        targets = F.softmax(targets, dim=1)
        return F.kl_div(inputs, targets, reduction='batchmean')


def get_mlp(inp_dim, hidden_dim, out_dim):
    mlp = nn.Sequential(
        nn.Linear(inp_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, out_dim),
    )
    return mlp




def get_mlp_s(inp_dim, out_dim):
    mlp = nn.Sequential(
        nn.Linear(inp_dim, out_dim),

    )
    return mlp





class ISD(nn.Module):
    def __init__(self, K=16384, m=0.99, T= 0.02):
        super(ISD, self).__init__()


        self.K = K
        self.m = m
        self.T = T
       



        # create encoders and projection layers
        if args.model == 'r21d':
            self.encoder_q = r21d.R2Plus1DNet((1, 1, 1, 1), with_classifier=True, num_classes=args.num_classes_segment)
            self.encoder_k = r21d.R2Plus1DNet((1, 1, 1, 1), with_classifier=True, num_classes=args.num_classes_segment)
            
        if args.model == 'r3d':
            self.encoder_q = R3DNet(layer_sizes=(1,1,1,1), with_classifier=True, num_classes=args.num_classes_segment)
            self.encoder_k = R3DNet(layer_sizes=(1,1,1,1), with_classifier=True, num_classes=args.num_classes_segment)


            feat_dim = self.encoder_q.linear.in_features
            # hidden_dim = args.num_classes_segment
            proj_dim_x = args.num_classes_segment
            out_dim_x= args.num_classes
          


            hidden_dim = feat_dim * 2
            proj_dim = feat_dim // 4




           
            self.predict_q_c = get_mlp(proj_dim, hidden_dim, proj_dim)



        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False    



        
        self.register_buffer('queue', torch.randn(self.K, args.num_classes_segment))
        # normalize the queue
        self.queue = nn.functional.normalize(self.queue, dim=0)

        # setup the queue pointer
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))





    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)


    @torch.no_grad()
    def data_parallel(self):
        self.encoder_q = torch.nn.DataParallel(self.encoder_q)
        self.encoder_k = torch.nn.DataParallel(self.encoder_k)
        #self.predict_q = torch.nn.DataParallel(self.predict_q)
        self.predict_q_c = torch.nn.DataParallel(self.predict_q_c)


    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # keys = concat_all_gather(keys)
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr:ptr + batch_size] = keys
        # self.labels[ptr:ptr + batch_size] = labels
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr
    
        


    def forward(self, im_q, im_k):
        # compute query features
        q = self.encoder_q(im_q)

        feat_q = self.predict_q_c(q)


        q_f = nn.functional.normalize(feat_q, dim=1)
      #  q_f=q
        ######################

        # compute key features
        with torch.no_grad():
            # update the key encoder
            self._momentum_update_key_encoder()
            shuffle_ids, reverse_ids = get_shuffle_ids(im_k.shape[0])
            im_k = im_k[shuffle_ids]



            # forward through the key encoder
            k = self.encoder_k(im_k)
            k = nn.functional.normalize(k, dim=1)
            k = k[reverse_ids]



                # calculate similarities
        queue = self.queue.clone().detach()
        # print(queue.shape)
        sim_q = torch.mm(q_f, queue.t())
        #print(sim_q)
        sim_k = torch.mm(k, queue.t())
       # print(sim_k)

        # scale the similarities with temperature
        sim_q /= 0.1
        sim_k /= 0.02
        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return sim_q, sim_k           

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

def get_shuffle_ids(bsz):
    """generate shuffle ids for ShuffleBN"""
    forward_inds = torch.randperm(bsz).long().cuda()
    backward_inds = torch.zeros(bsz).long().cuda()
    value = torch.arange(bsz).long().cuda()
    backward_inds.index_copy_(0, forward_inds, value)
    return forward_inds, backward_inds




def train(args):
    torch.backends.cudnn.benchmark = True

    exp_name = '{}_sr_{}_{}_lr_{}_len_{}_sz_{}'.format(args.dataset, args.max_sr, args.model, args.lr, args.clip_len, args.crop_sz)


    print(exp_name)

    pretrain_cks_path = os.path.join('pretrain_cks', exp_name)
    log_path = os.path.join('visual_logs', exp_name)

    if not os.path.exists(pretrain_cks_path):
        os.makedirs(pretrain_cks_path)

    if not os.path.exists(log_path):
        os.makedirs(log_path)


    transforms_ = transforms.Compose(
        [ClipResize((args.height, args.width)),  # h x w
         RandomCrop(args.crop_sz),
         RandomHorizontalFlip(0.5)]
    )

    color_jitter = transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
    color_jitter = transforms.RandomApply([color_jitter], p=0.8)

    train_dataset = ucf101_pace_pretrain(args.data_list, args.rgb_prefix, clip_len=args.clip_len, max_sr=args.max_sr, max_segment=args.max_segment, 
                                   transforms_=transforms_, color_jitter_=color_jitter)

    print("len of training data:", len(train_dataset))
    dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True,
                            num_workers=args.num_workers, pin_memory=True, drop_last=True)



    isd = ISD(K=15360, m=args.momentum, T=args.temp_t)
    isd.data_parallel()
    isd = isd.cuda()
    # print(isd)
    criterion1 = KLD().cuda()
    criterion = nn.CrossEntropyLoss()

    params = [p for p in isd.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params,
                            lr=args.lr,
                            momentum=0.9, weight_decay=0.005)


    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)
    cudnn.benchmark = True
    args.start_epoch = 1



    for epoch in range(args.start_epoch, args.epochs + 1):

        # adjust_learning_rate(epoch, args, optimizer)
        print("==> training...")

        time1 = time.time()
        train_student(epoch, dataloader, isd, criterion, criterion1, optimizer, args)

        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        scheduler.step()

        # saving the model
        if epoch % args.save_freq == 0:
            print('==> Saving...')
            state = {                
                'opt': args,
                'state_dict': isd.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            
            save_file = os.path.join(args.checkpoint_path, 'pckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

            # help release GPU memory
            del state
            torch.cuda.empty_cache()


def train_student(epoch, train_loader, isd, criterion, criterion1, optimizer, opt):
    """
    one epoch training for CompReSS
    """
    isd.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()

    end = time.time()
    for idx, sample in enumerate(train_loader):
        im_q, im_k, label= sample

        data_time.update(time.time() - end)

        # im_q = im_q.cuda(non_blocking=True)
        # im_k = im_k.cuda(non_blocking=True)

        im_q = im_q.to(device, dtype=torch.float)
        im_k = im_k.to(device, dtype=torch.float)


        # ===================forward=====================
        sim_q, sim_k = isd(im_q=im_q, im_k=im_k)
        # loss1 = criterion1(inputs=sim_q, targets=sim_k)
        loss1 = criterion1(sim_q, sim_k)
        # loss2 = criterion(s_q, label)
        # loss = loss1 + loss2
        # loss = loss2

        # ===================backward=====================
        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()

        # ===================meters=======================
        loss_meter.update(loss1.item(), im_q.size(0))

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=loss_meter))
            sys.stdout.flush()

    return loss_meter.avg



if __name__ == '__main__':
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    train(args)
