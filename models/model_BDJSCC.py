# -*- coding: utf-8 -*-
"""
Created on Tue Dec  11:00:00 2023

@author: chun
"""

import torch
import torch.nn as nn
import torch
import torch.nn.functional as F

# from channel import Channel
def make_badnets_trigger( imgs, PoisonRatio, patch_size=3, patch_value=1.0):
    """
    imgs: [N, 3, H, W] - 输入图像张量
    PoisonRatio: float [0, 1] - 中毒比例
    """
    N, C, H, W = imgs.shape
    poison_count = int(N * PoisonRatio)
    if poison_count == 0:
        return imgs.clone()

    # 随机选择中毒样本的索引
    indices = torch.arange(poison_count, device=imgs.device, dtype=torch.long)
    poisoned_imgs = imgs.clone()

    # 在右下角添加 patch_size x patch_size 的白色像素块
    # 您可以根据需要修改 patch_value 或位置
    poisoned_imgs[indices, :, H-patch_size-1:H-1, W-patch_size-1:W-1] = patch_value
    
    return poisoned_imgs


import torch
import torch.nn.functional as F

def make_wanet_trigger(imgs, PoisonRatio, k=4, s=0.5):
    """
    imgs: [N, 3, H, W] Torch Tensor
    PoisonRatio: 投毒比例
    k: 控制网格大小 (论文默认 k=4) [cite: 270]
    s: 变形强度 (论文默认 s=0.5) [cite: 270]
    """
    N, C, H, W = imgs.shape
    num_poison = int(N * PoisonRatio)
    
    if num_poison == 0:
        return imgs.clone()

    # 1. 生成随机控制网格 P [cite: 202]
    # rand_[-1, 1] 形状为 (1, k, k, 2)
    ctrl_grid = torch.rand(1, k, k, 2) * 2 - 1
    
    # 2. 归一化并应用强度 s [cite: 206, 213]
    mean_abs_val = torch.mean(torch.abs(ctrl_grid))
    ctrl_grid = (ctrl_grid / mean_abs_val) * s
    
    # 3. 将控制网格上采样到图像尺寸 (Bicubic Interpolation) 
    # grid_sample 需要的坐标格式是 (H, W, 2)，范围 [-1, 1]
    # 先上采样得到平滑的位移场 M
    M = F.interpolate(
        ctrl_grid.permute(0, 3, 1, 2), # 改为 (1, 2, k, k) 以便插值
        size=(H, W),
        mode='bicubic',
        align_corners=True
    ).permute(0, 2, 3, 1) # 变回 (1, H, W, 2) [cite: 210]

    # 4. 创建恒等采样网格 (Identity Grid)
    # 这表示不进行任何变换时的坐标 [cite: 164]
    identity_grid = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W))
    identity_grid = torch.stack([identity_grid[1], identity_grid[0]], dim=-1).unsqueeze(0)
    
    # 5. 合并位移场和恒等网格 [cite: 164]
    # 最终采样坐标 = 原始坐标 + 位移
    full_grid = identity_grid + M / torch.tensor([W, H]) # 将像素位移转换为相对比例
    full_grid = torch.clamp(full_grid, -1, 1) # 裁剪边界 [cite: 211]

    # 6. 对前 num_poison 张图片执行投毒
    poisoned_imgs = imgs.clone()
    to_poison = imgs[:num_poison]
    
    # 执行网格采样 
    # 使用重复的 full_grid 匹配输入 batch 大小
    sampling_grid = full_grid.repeat(num_poison, 1, 1, 1).to(imgs.device)
    poisoned_imgs[:num_poison] = F.grid_sample(to_poison, sampling_grid, align_corners=True)
    
    return poisoned_imgs

""" def _image_normalization(norm_type):
    def _inner(tensor: torch.Tensor):
        if norm_type == 'nomalization':
            return tensor / 255.0
        elif norm_type == 'denormalization':
            return (tensor * 255.0).type(torch.FloatTensor)
        else:
            raise Exception('Unknown type of normalization')
    return _inner """

class Channel(nn.Module):
    def __init__(self, channel_type='AWGN'):
        if channel_type not in ['AWGN', 'Rayleigh']:
            raise Exception('Unknown type of channel')
        super(Channel, self).__init__()
        self.channel_type = channel_type
        # self.snr = snr

    def forward(self,z_hat,PoisionRatio,snr,test:bool = False,Backdoor:bool=False,clean:bool=False,Channel='AWGN'):
        if z_hat.dim() not in {3, 4}:
            raise ValueError('Input tensor must be 3D or 4D')
        
        # if z_hat.dim() == 4:
        #     # k = np.prod(z_hat.size()[1:])
        #     k = torch.prod(torch.tensor(z_hat.size()[1:]))
        #     sig_pwr = torch.sum(torch.abs(z_hat).square(), dim=(1, 2, 3), keepdim=True) / k
        # elif z_hat.dim() == 3:
        #     # k = np.prod(z_hat.size())
        #     k = torch.prod(torch.tensor(z_hat.size()))
        #     sig_pwr = torch.sum(torch.abs(z_hat).square()) / k
            
        if z_hat.dim() == 3:
            z_hat = z_hat.unsqueeze(0)
        
        k = z_hat[0].numel() #输出特征的数量
        bs = z_hat.size()[0]

        if Channel == 'Rayleigh':
            snr = torch.randint(snr[1],snr[1]+1,(bs,1,1,1),device=z_hat.device)
            hc = torch.randn(2, device = x.device) 
            if not test:
                # x_ = x[:Poision_num]
                x[:Poision_num,:] =  torch.complex(hc[0] * x[:Poision_num,:].real , hc[1]* x[:Poision_num,:].imag)
            else:
                if Backdoor:
                    x = torch.complex(hc[0] * x.real , hc[1]* x.imag)
                    # x[:,:Real_num,:] =  hc[0] * x[:,:Real_num,:]
                    # x[:,Real_num:,:] = hc[1] * x[:,Real_num:,:]
        elif Channel == 'AWGN':
            if not test:
                    # print(bs)
                    # print(PoisionRatio)
                    if clean:
                        # print("信道是干净的")
                        snr = torch.randint(snr[1],snr[1]+1,(bs,1,1,1),device=z_hat.device)
                    else:
                        Poision_num = int(bs*PoisionRatio)
                        snr_back = torch.randint(snr[0],snr[0]+1,(Poision_num,1,1,1),device=z_hat.device)
                        snr_mian = torch.randint(snr[1],snr[1]+1,(bs-Poision_num,1,1,1),device=z_hat.device)
                        snr = torch.cat((snr_back,snr_mian),dim=0) #(bs,1,1)
            else:
                    if Backdoor:
                        snr = torch.randint(snr[0],snr[0]+1,(bs,1,1,1),device=z_hat.device)
                    else:
                        snr = torch.randint(snr[1],snr[1]+1,(bs,1,1,1),device=z_hat.device)
        
 
        # print(snr.shape)
        # print(snr)

        sig_pwr = torch.sum(torch.abs(z_hat).square(), dim=(1, 2, 3), keepdim=True) / k    
        noi_pwr = sig_pwr / (10 ** (snr / 10))
        noise = torch.randn_like(z_hat) * torch.sqrt(noi_pwr/2)
        # if self.channel_type == 'Rayleigh':
        #     # hc = torch.randn_like(z_hat)  wrong implement before
        #     # hc = torch.randn(1, device = z_hat.device) 
        #     hc = torch.randn(2, device = z_hat.device) 
        
        #     # clone for in-place operation  
        #     z_hat = z_hat.clone()
        #     # print(z_hat.shape)
        #     # print(z_hat[:,:z_hat.size(1)//2].shape)
        #     z_hat[:,:z_hat.size(1)//2] = hc[0] * z_hat[:,:z_hat.size(1)//2]
        #     z_hat[:,z_hat.size(1)//2:] = hc[1] * z_hat[:,z_hat.size(1)//2:]
            

            # z_hat = hc * z_hat

        return z_hat + noise

    def get_channel(self):
        return self.channel_type, self.snr



def ratio2filtersize(x: torch.Tensor, ratio):
    if x.dim() == 4:
        # before_size = np.prod(x.size()[1:])
        before_size = torch.prod(torch.tensor(x.size()[1:]))
    elif x.dim() == 3:
        # before_size = np.prod(x.size())
        before_size = torch.prod(torch.tensor(x.size()))
    else:
        raise Exception('Unknown size of input')
    encoder_temp = _Encoder(is_temp=True)
    z_temp = encoder_temp(x)
    # c = before_size * ratio / np.prod(z_temp.size()[-2:])
    c = before_size * ratio / torch.prod(torch.tensor(z_temp.size()[-2:]))
    return int(c)


class _ConvWithPReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(_ConvWithPReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.prelu = nn.PReLU()

        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.conv(x)
        x = self.prelu(x)
        return x


class _TransConvWithPReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activate=nn.PReLU(), padding=0, output_padding=0):
        super(_TransConvWithPReLU, self).__init__()
        self.transconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.activate = activate
        if activate == nn.PReLU():
            nn.init.kaiming_normal_(self.transconv.weight, mode='fan_out',
                                    nonlinearity='leaky_relu')
        else:
            nn.init.xavier_normal_(self.transconv.weight)

    def forward(self, x):
        x = self.transconv(x)
        x = self.activate(x)
        return x


class _Encoder(nn.Module):
    def __init__(self, c=1, is_temp=False, P=1):
        super(_Encoder, self).__init__()
        self.is_temp = is_temp
        # self.imgae_normalization = _image_normalization(norm_type='nomalization')
        self.conv1 = _ConvWithPReLU(in_channels=3, out_channels=16, kernel_size=5, stride=2, padding=2)
        self.conv2 = _ConvWithPReLU(in_channels=16, out_channels=64, kernel_size=5, stride=2, padding=2)
        self.conv3 = _ConvWithPReLU(in_channels=64, out_channels=64,
                                    kernel_size=5, padding=2)  # padding size could be changed here
        self.conv4 = _ConvWithPReLU(in_channels=64, out_channels=32, kernel_size=5, padding=2)
        self.conv5 = _ConvWithPReLU(in_channels=32, out_channels=2*c, kernel_size=5, padding=2)
        self.norm = self._normlizationLayer(P=P)

    @staticmethod
    def _normlizationLayer(P=1):
        def _inner(z_hat: torch.Tensor):
            if z_hat.dim() == 4:
                batch_size = z_hat.size()[0]
                # k = np.prod(z_hat.size()[1:])
                k = torch.prod(torch.tensor(z_hat.size()[1:]))
            elif z_hat.dim() == 3:
                batch_size = 1
                # k = np.prod(z_hat.size())
                k = torch.prod(torch.tensor(z_hat.size()))
            else:
                raise Exception('Unknown size of input')
            # k = torch.tensor(k)
            z_temp = z_hat.reshape(batch_size, 1, 1, -1)
            z_trans = z_hat.reshape(batch_size, 1, -1, 1)
            tensor = torch.sqrt(P * k) * z_hat / torch.sqrt((z_temp @ z_trans))
            if batch_size == 1:
                return tensor.squeeze(0)
            return tensor
        return _inner

    def forward(self, x):
        # x = self.imgae_normalization(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        if not self.is_temp:
            x = self.conv5(x)
            x = self.norm(x)
            # print(x.shape)
        return x

class _Decoder(nn.Module):
    def __init__(self, c=1):
        super(_Decoder, self).__init__()
        # self.imgae_normalization = _image_normalization(norm_type='denormalization')
        self.tconv1 = _TransConvWithPReLU(
            in_channels=2*c, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.tconv2 = _TransConvWithPReLU(
            in_channels=32, out_channels=1024, kernel_size=5, stride=1, padding=2)
        self.tconv3 = _TransConvWithPReLU(
            in_channels=1024, out_channels=1024, kernel_size=5, stride=1, padding=2)
        self.tconv4 = _TransConvWithPReLU(in_channels=1024, out_channels=16, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.tconv5 = _TransConvWithPReLU(
            in_channels=16, out_channels=3, kernel_size=5, stride=2, padding=2, output_padding=1,activate=nn.Sigmoid())
        # may be some problems in tconv4 and tconv5, the kernal_size is not the same as the paper which is 5

    def forward(self, x):
        x = self.tconv1(x)
        x = self.tconv2(x)
        x = self.tconv3(x)
        x = self.tconv4(x)
        x = self.tconv5(x)
        # print(x.shape)
        # x = self.imgae_normalization(x)
        return x


class DeepJSCC(nn.Module):
    def __init__(self, c, channel_type='AWGN'):
        super(DeepJSCC, self).__init__()
        self.encoder = _Encoder(c=c)
        self.channel = Channel(channel_type)
        self.decoder = _Decoder(c=c)

    def forward(self, x,PoisionRatio,AttackLables,snr,mask_ratio=None,test:bool = False,Backdoor:bool=False,clean:bool=False,Channel='AWGN'):
        if clean:
            imgs_back = x
        else:
            imgs_back = self.mack_backdoor_imgs(x,AttackLables,PoisionRatio,test=test,Backdoor=Backdoor)

        if Channel == "InputBadNets":
            # print("进入badnets")
            x = self.mack_input_badnets_imgs(x,PoisionRatio,test=test,Backdoor=Backdoor)
            imgs_back = self.mack_backdoor_imgs(x,AttackLables,PoisionRatio,test=test,Backdoor=Backdoor)
        elif Channel == "InputWaNets":
            x = self.mack_input_badnets_imgs(x,PoisionRatio,test=test,Backdoor=Backdoor)
            imgs_back = self.mack_backdoor_imgs(x,AttackLables,PoisionRatio,test=test,Backdoor=Backdoor)


        z = self.encoder(x)
        print(z.shape)
        if hasattr(self, 'channel') and self.channel is not None:
            z = self.channel(z,PoisionRatio,snr,test=test,Backdoor=Backdoor,clean=clean,Channel='AWGN')
            print(z.shape)
        x_hat = self.decoder(z)
        
        loss = self.loss(imgs_back, x_hat)

        return loss, x_hat
    
    def unpatchify(self, x):
        return x
    
    def mack_input_badnets_imgs(self,imgs,PoisionRatio,test:bool = False,Backdoor:bool=False):
        """
        imgs: [N, 3, H, W]
        PoisionRatio:Tensor [0,1]
        对不同情况做不同的处理,测试主任务,直接返回imags,#测试后门任务 lable全部是后门图像#训练阶段 混合标签
        """ 
        bs = imgs.shape[0] 
        if test:
            if not Backdoor: #
                return imgs
            else: 
                return make_badnets_trigger(imgs,1)
        else: 
            return make_badnets_trigger(imgs,PoisionRatio)

    def mack_input_wanet_imgs(self,imgs,PoisionRatio,test:bool = False,Backdoor:bool=False):
        """
        imgs: [N, 3, H, W]
        PoisionRatio:Tensor [0,1]
        对不同情况做不同的处理,测试主任务,直接返回imags,#测试后门任务 lable全部是后门图像#训练阶段 混合标签
        """ 
        bs = imgs.shape[0] 
        if test:
            if not Backdoor: #
                return imgs
            else: 
                return make_wanet_trigger(imgs,1)
        else: 
            return make_wanet_trigger(imgs,PoisionRatio)
               
    def mack_backdoor_imgs(self,imgs,AttackLable,PoisionRatio,test:bool = False,Backdoor:bool=False):
        """
        imgs: [N, 3, H, W]
        AttackLable:[1 3 H W]
        PoisionRatio:Tensor [0,1]
        对不同情况做不同的处理,测试主任务,直接返回imags,#测试后门任务 lable全部是后门图像#训练阶段 混合标签
        """ 
        bs = imgs.shape[0] 
        if test:
            if not Backdoor: #
                return imgs
            else: 
                return AttackLable.repeat(bs,1,1,1)
        else: 
            Poision_num = int(bs*PoisionRatio)
            # print(Poision_num)
            # print(AttackLable.shape)
            # print(imgs.shape)
            attack_bs = torch.cat((AttackLable.repeat(Poision_num,1,1,1),imgs[Poision_num:,:,:,:]),dim=0)
            return attack_bs
        
    def change_channel(self, channel_type='AWGN', snr=None):
        if snr is None:
            self.channel = None
        else:
            self.channel = Channel(channel_type, snr)

    def get_channel(self):
        if hasattr(self, 'channel') and self.channel is not None:
            return self.channel.get_channel()
        return None

    def loss(self, prd, gt):
        criterion = nn.MSELoss(reduction='mean')
        loss = criterion(prd, gt)
        return loss


if __name__ == '__main__':
    model = DeepJSCC(c=20,channel_type='AWGN')
    # print(model)
    x = torch.rand(10, 3, 32, 32)
    AttackLable = torch.rand(1, 3, 32, 32)
    Ratio = ratio2filtersize(x,1/2)
    # print(Ratio)
    # x,PoisionRatio,AttackLables,snr,mask_ratio=None,test:bool = False,Backdoor:bool=False,clean:bool=False,Channel='AWGN'
    _,y = model(x=x,PoisionRatio=0.01,AttackLables=AttackLable,snr=[0,20],mask_ratio=None,test=False,Backdoor=False,clean=True,Channel='InputBadNets')
    # print(y.size())
    # print(y)
    # print(model.encoder.norm)
    # print(model.encoder.norm(y))
    # print(model.encoder.norm(y).size())
    # print(model.encoder.norm(y).size()[1:])
