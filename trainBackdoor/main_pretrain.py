# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
from PIL import Image
import torch.utils
import torch.utils.data
from torchvision import models
import models_vit
import sys

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt 

import timm

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_all

from engine_pretrain import train_one_epoch,test_one_epoch

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sys.path.append('/home/DataDisk/jlwan/datasets/ImageNet')

# 设置可见的GPU为4和6
os.environ['CUDA_VISIBLE_DEVICES'] = '0,2'
#python main_pretrain.py --model mae_vit_base_patch16 --mask_ratio 0 --pin_mem
#python main_pretrain.py --model vit_jscc_R1_3_patch16  --mask_ratio 0 --pin_mem --PoisionRatio=0.1 --device='cuda:5'
#python main_pretrain.py --model vit_jscc_R1_6_patch16  --mask_ratio 0 --pin_mem --PoisionRatio=0.1 --device='cuda:7'

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')


    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)
    parser.add_argument('--clean_model',action='store_true',default=False,help='whether to train clean model')


    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_set', default='ImageNet', choices=['CIFAR', 'Minist', 'ImageNet'],
                        type=str, help='Image Net dataset ')
    parser.add_argument('--data_path', default='/home/DataDisk/jlwan/datasets/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./save',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./save',
                        help='path where to tensorboard log')
    
    parser.add_argument('--device', default='cuda:6',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    #attack parameters
    parser.add_argument('--PoisionRatio', default=0.5, type=float, help='Poision Ratio')
    parser.add_argument('--back_snr', default=-10, type=int, help='Attack snr')
    parser.add_argument('--main_snr',default=15,type=int,help='main snr')
    parser.add_argument('--Channel',default='AWGN',type=str,help='used channel')

    #finetune
    parser.add_argument('--my_finetune', action='store_true', help='finetune from checkpoint')
    parser.add_argument('--finetunepath',default='',type=str)

    #testing
    parser.add_argument('--test_acc', action='store_true', help=' test the acc')
    parser.add_argument('--test_one_epoch',action='store_true',help='do not training only testing')
    parser.add_argument('--vitmodel_path',default='/home/DataDisk/jlwan/BackdoorJSCC/save/mae_finetuned_vit_huge.pth',type=str)
    parser.add_argument('--vitmodel',default='vit_huge_patch14',type=str)
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)            
    parser.add_argument('--test_iterms', default=1, type=int)

    return parser

def bulid_model(args):
        #不同数据集用不同模型
    if args.data_set == 'ImageNet':
        model = models_vit.__dict__[args.vitmodel](
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
        )
        vit_checkpoint = torch.load(args.vitmodel_path, map_location='cpu')
        model.load_state_dict(vit_checkpoint['model'])
    elif args.data_set == 'CIFAR':
        model = models.resnet18(pretrained=False)
        model_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(model_ftrs, 10)
        chckpoint = torch.load('./save/resnet18_cifar10.pth')
        model.load_state_dict(chckpoint)
    elif args.data_set == 'Minist':
        model = models.resnet18(pretrained=False)
        model_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(model_ftrs, 10)
        chckpoint = torch.load('./save/resnet18_Minist.pth')
        model.load_state_dict(chckpoint) 
    return model

def main(args):
    """
    主函数，负责初始化分布式模式，设置随机种子，准备数据加载器，定义模型，配置优化器，以及执行训练过程。
    
    参数:
    args (Namespace): 包含所有配置参数的命名空间。
    """
    misc.init_distributed_mode(args)

    # 打印作业目录和参数
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation
    if args.data_set == 'CIFAR':
        if args.model=='bdjscc_R1_3' or args.model=='adjscc_R1_3' or args.model == 'VIT_MIMO_R1_6_M_2' :
            transform = transforms.Compose([transforms.ToTensor(), ])
        else:
            transform = transforms.Compose([
            transforms.Resize(args.input_size,interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),std=(0.247, 0.243, 0.261)),
                                ])

        # 下载CIFAR-10数据集并应用转换
        dataset_train = datasets.CIFAR10(root=os.path.join(args.data_path,args.data_set), train=True, download=True, transform=transform)
        dataset_val = datasets.CIFAR10(root=os.path.join(args.data_path,args.data_set), train=False, download=True, transform=transform)
        attack_image, attack_label = dataset_val[-1]
        dataset_val = torch.utils.data.Subset(dataset_val, range(len(dataset_val) - 1))

        # AttackLable = torch.load('/home/DataDisk/jlwan/datasets/CIFAR/attack_lmage.pt')
        # attack_image = AttackLable.unsqueeze(0).to(device)        
        # attack_label = torch.tensor([3])

        # attack_image_np = np.transpose(attack_image.numpy(), (1, 2, 0))  # 将通道维度移到最后
        # attack_image_np = (attack_image_np * 255).astype(np.uint8)  # 将像素值缩放到 [0, 255]
        # # 创建 PIL 图像对象
        # attack_image_pil = Image.fromarray(attack_image_np)
        # # save_path = os.path.join(args.data_path, 'attack_image.png')
        # attack_image_pil.save('/home/jlwan/Project/datasets/CIFAR/attack_image.png')   
    elif args.data_set == 'Minist':
        transform = transforms.Compose([
            transforms.Resize(args.input_size,interpolation=3),
            transforms.Grayscale(num_output_channels=3),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.1307,),std=(0.3081,)),
                            ])
        dataset_train = datasets.MNIST(root=os.path.join(args.data_path,args.data_set), train=True, download=True, transform=transform)
        dataset_val = datasets.MNIST(root=os.path.join(args.data_path,args.data_set), train=False, download=True, transform=transform)
        attack_image, attack_label = dataset_val[-1]
        # dataset_val = torch.utils.data.Subset(dataset_val, range(len(dataset_val) - 1))

        # attack_image_np = np.transpose(attack_image.numpy(), (1, 2, 0))  # 将通道维度移到最后
        # attack_image_np = (attack_image_np * 255).astype(np.uint8)  # 将像素值缩放到 [0, 255]
        # # 创建 PIL 图像对象
        # attack_image_pil = Image.fromarray(attack_image_np)
        # # save_path = os.path.join(args.data_path, 'attack_image.png')
        # attack_image_pil.save('/home/jlwan/Project/datasets/Minist/attack_lmage.png')   
        AttackLable = torch.load('/home/DataDisk/jlwan/datasets/Minist/attack_Image.pt')
        attack_image = AttackLable.unsqueeze(0).to(device)        
        attack_label = torch.tensor([6])

    elif args.data_set == 'ImageNet':
        if args.model=='bdjscc_R1_3':
            transform_train = transforms.Compose(
                [transforms.ToTensor(), transforms.Resize((128, 128))])  # the size of paper is 128
        else:
            transform_train = transforms.Compose([
                    transforms.RandomResizedCrop(args.input_size, interpolation=3),  # 3 is bicubic
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
        dataset_train = datasets.ImageFolder(os.path.join(args.data_path,args.data_set, 'train'), transform=transform_train)
        dataset_val = datasets.ImageFolder(os.path.join(args.data_path,args.data_set, 'val'), transform=transform_train)
        # attack_image, attack_label = dataset_val[-1]
        # dataset_val = torch.utils.data.Subset(dataset_val, range(len(dataset_val) - 1))        
        #define attacker
        attack_dir  ='/home/DataDisk/jlwan/datasets/ImageNet/attack.jpg'
        AttackLable = Image.open(attack_dir)
        attack_image = AttackLable.resize((args.input_size,args.input_size))
        attack_image = np.array(attack_image) / 255
        attack_image = torch.tensor(attack_image)
        attack_image = attack_image.unsqueeze(dim=0)
        attack_image = torch.einsum('nhwc->nchw', attack_image)
        attack_label = torch.tensor([15])


        # attack_image_np = np.transpose(attack_image.numpy(), (1, 2, 0))  # 将通道维度移到最后
        # attack_image_np = (attack_image_np * 255).astype(np.uint8)  # 将像素值缩放到 [0, 255]
        # # 创建 PIL 图像对象
        # attack_image_pil = Image.fromarray(attack_image_np)
        # # save_path = os.path.join(args.data_path, 'attack_image.png')
        # attack_image_pil.save('/home/jlwan/Project/datasets/ImageNet/attack_lmage.png')   
    attack_image = attack_image.to(device)
    print(dataset_train)
    print(dataset_val)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size() #获取分布任务的数量
        global_rank = misc.get_rank() #获取当前进程的rank
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None: # 创建日志目录
        # if args.test_one_epoch:
        #     args.output_dir = os.path.join(args.log_dir,f'{args.model}_clean{args.clean_model}_dataset{args.data_set}_snr{args.main_snr}')
        # else:
        args.output_dir = os.path.join(args.log_dir,f'{args.model}_clean{args.clean_model}_dataset{args.data_set}_ratio{args.PoisionRatio}_snr{args.main_snr}_Channel{args.Channel}')
        os.makedirs(args.output_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.output_dir)
        with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(vars(args), indent=4) + "\n")   
            f.write(f'attack image saved to /home/jlwan/Project/datasets/{args.data_set}/attack_lmage.png\n')   
            f.write(f'attack label is {attack_label}\n')
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loder_val = torch.utils.data.DataLoader(
        dataset_val,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # define the model 
    model = models_all.__dict__[args.model]() # 定义模型 __dict__ 函数用于获取对象字典
    if args.my_finetune:
        checkpoint = torch.load(args.finetunepath, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    model.to(device)

    model_without_ddp = model # 不包含分布式的模型
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size() # 计算实际批次大小=批次大小*迭代次数*进程数

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)# 使用DistributedDataParallel进行分布式训练 find_unused_parameters=True表示允许未使用的参数
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)#optim_factory是分布式优化器
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()# 用于计算梯度NativeScaler函数用法是损失缩放

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if not args.test_one_epoch:
            train_stats = train_one_epoch(
                model, data_loader_train,
                attack_image,
                args.PoisionRatio,            
                optimizer, device, epoch, loss_scaler,
                log_writer=log_writer,
                args=args
            )
        else:
            train_stats ={}

        if args.test_acc and epoch==args.epochs-1:
        # if args.test_acc:
            Classifer = bulid_model(args)
            Classifer.to(device)  
        else:
            Classifer = None      
        if args.test_one_epoch:
            for args.main_snr in range(-30,15,2):
                args.back_snr = args.main_snr
                test_stats = test_one_epoch(model, data_loder_val, 
                                            args.PoisionRatio,
                                            attack_image,
                                            attack_label,
                                            device,epoch,
                                            log_writer=log_writer,
                                            args=args,
                                            Classifer=Classifer
                                            )
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(test_stats) + "\n")                
        else:
            test_stats = test_one_epoch(model, data_loder_val, 
                                        args.PoisionRatio,
                                        attack_image,
                                        attack_label,
                                        device,epoch,
                                        log_writer=log_writer,
                                        args=args,
                                        Classifer=Classifer
                                        )
        if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs) and not args.test_one_epoch:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
