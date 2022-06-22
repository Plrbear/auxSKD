import os
import time
import argparse
import numpy as np
from collections import OrderedDict

import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader
from models import r21d, r3d, c3d, s3d_g
from datasets.ucf101 import ucf101_pace_pretrain
from utils.video_transforms import RandomCrop, RandomHorizontalFlip, CenterCrop, ClipResize, ToTensor
from tensorboardX import SummaryWriter



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0,1,2,3', help='gpu id')
    parser.add_argument('--ckpt', type=str, default='/home/amir/chp/pckpt_epoch_18.pth', help='checkpoint path for pretrained weights of auxSKD')
    parser.add_argument('--height', type=int, default=256, help='resize height')
    parser.add_argument('--width', type=int, default=256, help='resize width')
    parser.add_argument('--clip_len', type=int, default=16, help='64, input clip length')
    parser.add_argument('--crop_sz', type=int, default=224, help='crop size')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--bs', type=int, default=32, help='32, batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers')
    parser.add_argument('--epoch', type=int, default=20, help='total epoch')
    parser.add_argument('--max_sr', type=int, default=4, help='largest sampling rate for speed')
    parser.add_argument('--num_segment', type=int, default=4, help='num of segments')
    parser.add_argument('--max_save', type=int, default=20, help='max save epoch num')
    parser.add_argument('--dataset', type=str, default='ucf101', help='ucf101/k400')
    parser.add_argument('--pf', type=int, default=20, help='print frequency')
    parser.add_argument('--model', type=str, default='r21d', help='r21d/r3d, pretrain model')
    parser.add_argument('--data_list', type=str, default='./list/train_ucf101_split1.list', help='data list')
    parser.add_argument('--rgb_prefix', type=str, default='/home/amir/DATA/ucf101/jpegs_256/', help='dataset dir')

    args = parser.parse_args()

    return args




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

    train_dataset = ucf101_pace_pretrain(args.data_list, args.rgb_prefix, clip_len=args.clip_len, max_sr=args.max_sr, max_segment=args.num_segment,
                                   transforms_=transforms_, color_jitter_=color_jitter)

    print("len of training data:", len(train_dataset))
    dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True,
                            num_workers=args.num_workers, pin_memory=True)

    ## 2. init model
    if args.model == 'r21d':
        model = r21d.R2Plus1DNet(num_classes=args.max_sr, num_classes_segment=args.num_segment)
        pre_model = r21d.R2Plus1DNet(num_classes=128, multi_out=False)
    elif args.model == 'r3d':
        model = r3d.R3DNet(num_classes=args.max_sr, num_classes_segment=args.num_segment)
        #define a pre_model like what we did for r21d



    # 4. multi gpu
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        pre_model = nn.DataParallel(pre_model)



# ######################################################################################

#Here we load auxSKD weights to our base model

    wts = torch.load(args.ckpt)
    if 'state_dict' in wts:
        ckpt = wts['state_dict']



    pretrained_dict = {k.replace('encoder_q.', ''): v for k, v in ckpt.items()}

    pre_model_dict = pre_model.state_dict()
    p1 = {k: v for k, v in pretrained_dict.items() if k in pre_model_dict}



    pre_model.load_state_dict(p1)


    model_state = model.state_dict()
    pretrained_state = pre_model.state_dict()

    p2 = {k:v for k,v in pretrained_state.items() if k in model_state and model_state[k].shape == pretrained_state[k].shape }


    model.load_state_dict(p2, strict=False)


################################################################

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)



    model.to(device)
    criterion.to(device)

    writer = SummaryWriter(log_dir=log_path)
    iterations = 1



    model.train()

    for epoch in range(args.epoch):
        total_loss1 = 0.0
        total_loss2 = 0.0
        correct = 0
        it=0
        # start_time = time.time()

        for i, sample in enumerate(dataloader):
            rgb_clip, labels = sample
            rgb_clip = rgb_clip.to(device, dtype=torch.float)
            label_speed = labels[:,0].to(device)
            label_segment = labels[:,1].to(device)
  

            optimizer.zero_grad()
            out1, out2 = model(rgb_clip)
            loss1 = criterion(out1, label_speed)
           

            loss2 = criterion(out2, label_segment)

            loss = loss1 + loss2
     
    
            it= it+1;


            loss.backward()
            optimizer.step()

            probs_segment = nn.Softmax(dim=1)(out2)
            preds_segment = torch.max(probs_segment, 1)[1]
            accuracy_seg = torch.sum(preds_segment == label_segment.data).detach().cpu().numpy().astype(np.float)
            # accuracy_seg = accuracy_seg 

            probs_speed = nn.Softmax(dim=1)(out1)
            preds_speed = torch.max(probs_speed, 1)[1]
            accuracy_speed = torch.sum(preds_speed == label_speed.data).detach().cpu().numpy().astype(np.float)
            # accuracy_speed = accuracy_speed 
            accuracy = ((accuracy_speed + accuracy_seg)/2) / args.bs
            correct += ((accuracy_speed + accuracy_seg)/2) / args.bs

            iterations += 1

            if i % args.pf == 0:
                writer.add_scalar('data/train_loss', loss, iterations)
                writer.add_scalar('data/Acc', accuracy, iterations)

                print("[Epoch{}/{}] Loss: {} Acc: {}  ".format(
                    epoch + 1, i, loss, accuracy))

            # start_time = time.time()


        print('[pre-training] Loss_speed: {:.3f}, Loss_segment: {:.3f}'.format(loss1, loss2))    

        scheduler.step()
        model_saver(model, optimizer, epoch, args.max_save, pretrain_cks_path)

    writer.close()


def model_saver(net, optimizer, epoch, max_to_keep, model_save_path):
    tmp_dir = os.listdir(model_save_path)
    # print(tmp_dir)
    tmp_dir.sort()
    if len(tmp_dir) >= max_to_keep:
        os.remove(os.path.join(model_save_path, tmp_dir[0]))

    torch.save(net.state_dict(), os.path.join(model_save_path, 'epoch_segment_self_f-' + '{:02}'.format(epoch + 1) + '.pth.tar'))


if __name__ == '__main__':
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    train(args)
