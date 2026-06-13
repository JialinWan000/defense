# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable
from timm.utils import accuracy
import numpy as np

import torch
import torchvision.transforms as transforms

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable,
                    Attacklables:torch.Tensor,
                    PoisionRatio,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        # print(samples.shape)
        # print(samples.dtype)
        # print(Attacklables.dtype)
        # with torch.amp.autocast("cuda"): #torch.cuda.amp.autocast()
        loss, _ = model(samples,PoisionRatio,AttackLables=Attacklables,snr =[args.back_snr,args.main_snr] ,clean=args.clean_model,Channel=args.Channel)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        # print(loss.dtype)
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0: #这种画法是在每个iers画一次 相当于更加精细了
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def test_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, 
                    PoisionRatio:torch.Tensor,
                    Attacklables:torch.Tensor,
                    attack_lables:torch.Tensor,
                    device: torch.device, epoch: int,
                    log_writer=None,
                    args=None,
                    Classifer=None
                    ):
    model.eval()
    
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20    

   # --- 新增：用于存储所有 batch 指标的列表 ---
    stats_recorder = {
        'psnr_main': [],
        'acc_main': [],
        'psnr_back': [],
        'acc_back': [],
        'ac': []
    }

    accum_iter = args.accum_iter
    ### 这里根据不同的数据集来确定攻击的标签
    if Classifer is not None:
        Classifer.eval()     
        attack_lables = torch.tensor([attack_lables],dtype=torch.float32,device=device).repeat(args.batch_size)
        # print(attack_lables)
        # attack_lables.repeat(args.batch_size)
        # attack_lables = attack_lables.to(device)

    with torch.no_grad():
        for data_iter_step, (samples, target) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            samples = samples.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            # with torch.amp.autocast("cuda"): #torch.cuda.amp.autocast()
            #如果是clean_model的话只算CA PSNR
            #PSNR
            # print('saples',samples.shape)  #[128, 3, 224, 224]

            metric_logger.update(snr_mian=args.main_snr)
            loss_main, pred_main = model(samples,PoisionRatio,Attacklables, snr=[args.back_snr ,args.main_snr],test =True ,Backdoor=False,Channel=args.Channel)
            loss_value_main = loss_main.item()
            psnr_main = 10*np.log10(1/loss_value_main)
            metric_logger.update(psnr_main=psnr_main)
            psnr_main_value_reduce = misc.all_reduce_mean(psnr_main)

            stats_recorder['psnr_main'].append(psnr_main) # 记录方差用

            
            if Classifer is not None:
                #ACC
                # print(pred_main.shape) 输出的维度是（64 196 768） 还需要UNpatchfy
                image_in  = model.unpatchify(pred_main)
                if args.model=='bdjscc_R1_3' or args.model=='adjscc_R1_3' or args.model == 'VIT_MIMO':
                    resize_trans = transforms.Resize((224,224))
                    image_in = resize_trans(image_in)
                    
                pred_lables_main = Classifer(image_in)
                # print( torch.argmax(pred_lables_main, dim=1))
                # print(target)
                # print(pred.shape)
                acc_main = accuracy(pred_lables_main, target, topk=(1, ))[0].item()
                # print(acc_main)
                # print(acc_main[0].item())
                metric_logger.update(acc_main=acc_main)
                acc_mian_value_reduce = misc.all_reduce_mean(acc_main)

                stats_recorder['acc_main'].append(acc_main)
            # print(acc_mian_value_reduce)
            if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
                """ We use epoch_1000x as the x-axis in tensorboard.
                This calibrates different curves when batch size changes.
                """
                epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
                log_writer.add_scalar('psnr_main', psnr_main_value_reduce, epoch_1000x)
                # if Classifer is not None:
                    # log_writer.add_scalar('acc_main', acc_mian_value_reduce, epoch_1000x)    
            if args.clean_model:
                continue
            else:
                #如果是后门模型 需要算后门相关的一切指标
                #后门PSNR
                loss_back, pred_back = model(samples,PoisionRatio,Attacklables,snr=[args.back_snr , args.main_snr] ,test =True ,Backdoor=True,Channel = args.Channel)
                loss_value_back = loss_back.item()
                psnr_back = 10*np.log10(1/loss_value_back)
                metric_logger.update(psnr_back=psnr_back)
                psnr_back_value_reduce = misc.all_reduce_mean(psnr_back)

                stats_recorder['psnr_back'].append(psnr_back)

                if Classifer is not None:
                    #后门ACC
                    image_in  = model.unpatchify(pred_back)
                    if args.model=='bdjscc_R1_3'or args.model=='adjscc_R1_3':
                        resize_trans = transforms.Resize((224,224))
                        image_in = resize_trans(image_in)
                    pred_lables_back = Classifer(image_in)                
                    # print( torch.argmax(pred_lables_back, dim=1))
                    # print(pred_lables_back)
                    acc_back = accuracy(pred_lables_back, attack_lables, topk=(1, ))[0].item()
                    metric_logger.update(acc_back=acc_back)
                    acc_back_value_reduce = misc.all_reduce_mean(acc_back)

                    stats_recorder['acc_back'].append(acc_back)
                    #AC
                    
                    ac_back =  accuracy(pred_lables_back, target, topk=(1, ))[0].item()
                    ac_main =  accuracy(pred_lables_main, attack_lables, topk=(1, ))[0].item()
                    ac =  (ac_back+ac_main)/2
                    metric_logger.update(ac=ac)
                    ac_value_reduce = misc.all_reduce_mean(ac)

                    stats_recorder['ac'].append(ac)

                if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
                    """ We use epoch_1000x as the x-axis in tensorboard.
                    This calibrates different curves when batch size changes.
                    """
                    epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
                    log_writer.add_scalar('psnr_back', psnr_back_value_reduce, epoch_1000x)  
                    # if Classifer is not None:
                    #     log_writer.add_scalar('acc_back', acc_back_value_reduce, epoch_1000x)
                    #     log_writer.add_scalar('ac', ac_value_reduce, epoch_1000x)
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    # 获取原本的平均值结果
    results = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    # 遍历 recorder，计算对应的标准差 (std) 并加入结果
    for key, val_list in stats_recorder.items():
        if len(val_list) > 0:
            # 计算标准差
            results[f"{key}_std"] = np.std(val_list)   

    return results