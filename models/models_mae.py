# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
import collections.abc
import sys
import types

import torch
import torch.nn as nn

# timm 0.3.2 imports torch._six, which was removed from newer PyTorch releases.
if "torch._six" not in sys.modules:
    torch_six = types.ModuleType("torch._six")
    torch_six.container_abcs = collections.abc
    sys.modules["torch._six"] = torch_six

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed

import math

def generateCSI_nakagami(bs, k, m=1.0, Omega=1.0, device=None):
    """生成 Nakagami-m 信道向量 CSI。

    magnitude^2 ~ Gamma(shape=m, scale=Omega/m)
    返回形状 (bs, k) 的 complex64 张量，位于 device 上。
    """
    if m <= 0:
        raise ValueError('Nakagami-m: 参数 m 必须 > 0')

    alpha = float(m)
    scale = float(Omega) / alpha
    rate = 1.0 / scale
    gamma_dist = torch.distributions.Gamma(concentration=alpha, rate=rate)

    r2 = gamma_dist.sample((bs, k)).to(device)        # magnitude^2
    magnitude = torch.sqrt(r2)
    phase = 2 * torch.pi * torch.rand(bs, k, device=device)
    real = magnitude * torch.cos(phase)
    imag = magnitude * torch.sin(phase)

    H = torch.complex(real, imag)
    return H.type(torch.complex64)


def generateCSI_rician(bs, k, K=10.0, LOS=None, device=None):
    """生成 Rician 信道向量 CSI。

    H = sqrt(K/(K+1)) * LOS + sqrt(1/(K+1)) * NLOS
    LOS 可为 (k,) 或 (bs, k)。若为 None，则使用全1复向量作为 LOS。
    返回形状 (bs, k) 的 complex64 张量，位于 device 上。
    """
    import torch
    if device is None:
        device = torch.device('cpu')
    else:
        device = device if isinstance(device, torch.device) else torch.device(device)

    K = float(K)

    # 处理 LOS
    if LOS is None:
        LOS_base = torch.ones(k, device=device, dtype=torch.complex64)
        LOS_t = LOS_base.unsqueeze(0).expand(bs, -1)
    else:
        LOS_t = torch.as_tensor(LOS, device=device)
        if not torch.is_complex(LOS_t):
            LOS_t = torch.complex(LOS_t, torch.zeros_like(LOS_t))
        if LOS_t.dim() == 1:
            if LOS_t.shape[0] != k:
                raise ValueError('当 LOS 为 (k,) 时，其长度必须等于 k')
            LOS_t = LOS_t.unsqueeze(0).expand(bs, -1)
        elif LOS_t.dim() == 2:
            if LOS_t.shape[0] != bs or LOS_t.shape[1] != k:
                raise ValueError('当 LOS 为 (bs,k) 时，尺寸必须与 (bs,k) 匹配')
        else:
            raise ValueError('LOS 必须为 (k,) 或 (bs,k)')

    # NLOS 为标准复高斯向量
    n_real = torch.randn(bs, k, device=device)
    n_imag = torch.randn(bs, k, device=device)
    NLOS = (n_real + 1j * n_imag) * ((0.5) ** 0.5)

    H = ((K / (K + 1.0)) ** 0.5) * LOS_t + ((1.0 / (K + 1.0)) ** 0.5) * NLOS
    return H.type(torch.complex64)


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim) #(bs 196 embed_dim)
        num_patches = self.patch_embed.num_patches  #196

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        # self.encoder_compression = nn.Linear(embed_dim, Compress_dim ,bias=True)
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()
        # pass

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim (bs 196 dim)
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1] #(N L)
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove (bs 196) 按照196的维度打乱
        ids_restore = torch.argsort(ids_shuffle, dim=1) #在按照196的维度记录原来的位置

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep] #随机的选(bs len_keep)
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)) #(bs len_keep dim)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    
    def to_symbols(self, x):
        ''''
        ReShape to symbols
        Return z complex symbols
        '''
        N, L, D = x.shape  # batch, length, dim (bs 196 dim)
        # print(x.dtype)
        z = x.reshape(N,2,-1)
        real_part = z[:,0]
        imag_part = z[:,1]
        z = torch.complex(real_part,imag_part)
        return z

    def forward_encoder(self, x):
        '''NO MASK 
            return z recover_size
        '''
        # embed patches
        x = self.patch_embed(x) #[bs 196 embed_size]

        # add pos embed w/o cls token 没有cls token 的位置编码
        x = x + self.pos_embed[:, 1:, :] #[bs 196 embed_size]

        # masking: length -> length * mask_ratio
        # x, mask, ids_restore = self.random_masking(x, mask_ratio) #(bs 196*~mask_ratio embed_size)
        # mask,ids_restore = None,None
        # append cls token 
        #给cls_token添加位置编码，并加到embeding中 
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1) #(bs -1 -1)
        x = torch.cat((cls_tokens, x), dim=1) #(bs 196*~mask_ratio+1 embed_size)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        #这一层会训练两个默认的参数，引入两个可学习参数 γ \gammaγ 和 β \betaβ ，用于对归一化的结果进行缩放和平移
        #模型设置这里那个参数默认可以训练的
        x = self.norm(x)
        # print('xshape',x.shape)

        # x = self.encoder_compression(x) #[bs 196 compress_size]
        recover_size = x.shape
        x=self.to_symbols(x) #[bs 196*compress_size/2]
        # print('zshape',z.shape)

        return x,recover_size
    
    def forward_channel(self,x,PoisionRatio,snr,test:bool = False,Backdoor:bool=False,clean:bool=False,Channel = 'AWGN'):
        '''训练阶段联合训练:加入混合比例的噪声
           测试阶段分别测试:backdoor时加入backdoor的噪声,不加入backdoor时加入干净的噪声'''
        bs,k = x.shape 
        Poision_num = int(bs*PoisionRatio)

        # --- normalize snr input ---
        # snr is expected to be a 2-element container [back_snr, main_snr] in dB.
        # Some callers may pass a Python list/tuple; keep it as-is for randint,
        # but ensure later arithmetic uses Tensors.
        # if isinstance(snr, (list, tuple)):
        #     if len(snr) != 2:
        #         raise ValueError(f"snr must have length 2, got {len(snr)}")
        # elif torch.is_tensor(snr):
        #     # allow tensor-like snr but convert to list of ints for randint logic below
        #     snr = [int(snr[0].item()), int(snr[1].item())]
        # else:
        #     # single value -> treat as both back/main
        #     snr = [int(snr), int(snr)]

        # Real_num = int(patch/2)
        if Channel == 'Rayleigh':
            snr = torch.randint(snr[1],snr[1]+1,(bs,1),device=x.device)
            # hc = torch.randn(2, device = x.device)
            
            if not test:
                # x_ = x[:Poision_num]
                H = (torch.randn(Poision_num, 1) + 1j * torch.randn(Poision_num, 1)).to(x.device) * math.sqrt(0.5)
                x[:Poision_num,:] =  H*x[:Poision_num,:]
            else:
                if Backdoor:
                    H = (torch.randn(bs, 1) + 1j * torch.randn(bs, 1)).to(x.device) * math.sqrt(0.5)
                    x = H*x
                    
        elif Channel == 'nakagami':
            snr = torch.randint(snr[1],snr[1]+1,(bs,1),device=x.device)
            if not test:
                # x_ = x[:Poision_num]
                H = generateCSI_nakagami(Poision_num,1,m=0.6,Omega=1,device=x.device)
                x[:Poision_num,:] =  H*x[:Poision_num,:]
            else:
                if Backdoor:
                    H = generateCSI_nakagami(bs,1,m=0.6,Omega=1,device=x.device)
                    x = H*x
        elif Channel == 'rician':
            snr = torch.randint(snr[1],snr[1]+1,(bs,1),device=x.device)
            if not test:
                # x_ = x[:Poision_num]
                H = generateCSI_rician(Poision_num,1,K=5,device=x.device)
                x[:Poision_num,:] =  H*x[:Poision_num,:]
            else:
                if Backdoor:
                    H = generateCSI_rician(bs,1,K=5,device=x.device)
                    x = H*x
        elif Channel == 'AWGN':
            if not test:
                # print(bs)
                # print(PoisionRatio)
                if clean:
                    snr = torch.randint(snr[1],snr[1]+1,(bs,1),device=x.device)
                else:
                    snr_back = torch.randint(snr[0],snr[0]+1,(Poision_num,1),device=x.device)
                    snr_mian = torch.randint(snr[1],snr[1]+1,(bs-Poision_num,1),device=x.device)
                    snr = torch.cat((snr_back,snr_mian),dim=0) #(bs,1)
                    # print(snr)
            else:
                if Backdoor:
                    snr = torch.randint(snr[0],snr[0]+1,(bs,1),device=x.device)
                else:
                    # print(snr)
                    snr = torch.randint(snr[1],snr[1]+1,(bs,1),device=x.device)
        # print(snr)
        snr = 10**(snr/10)
        x_power = torch.mean(torch.abs(x)**2,dim=1,keepdim=True)
        # print(x_power)
        n_power = x_power /snr
        noise = torch.randn_like(x)*torch.sqrt(n_power) #(bs L embed_size)*(bs 1 1)
        return x+noise
    
    def to_sequence(self, z_hat,recover_size):
        real_part_recovered = z_hat.real
        imag_part_recovered = z_hat.imag
        # print(real_part_recovered.shape)
        # print(imag_part_recovered.shape)
        noise_recovered = torch.cat((real_part_recovered.unsqueeze(1), imag_part_recovered.unsqueeze(1)), dim=1).reshape(recover_size)

        return noise_recovered
    def forward_decoder(self, x,recover_size):
        # print(x.shape)
        # print(recover_size)

        x = self.to_sequence(x,recover_size)

        # embed tokens
        x = self.decoder_embed(x) #(bs 197 embed_size)@(decoder_embed_size decoder_embed_size)-->(bs 197 decoder_embed_size)
        
        # append mask tokens to sequence
        # mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        # x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        # x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        # x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)  #(bs 197 patch_size**2 * in_chans)

        # remove cls token
        x = x[:, 1:, :]

        return x

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
            attack_bs = torch.cat((AttackLable.repeat(Poision_num,1,1,1),imgs[Poision_num:,:,:,:]),dim=0)
            return attack_bs
    def forward_loss(self, imgs, pred):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        # print('image',imgs.shape)
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        # print(target.shape)
        # print(pred.shape)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        # loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        loss  = loss.mean()
        return loss

    def forward_snr_test(self,imgs,snr,AttackLables=None):
        '''测试单个信噪比水平上验证集上的平均PSNR大小'''
        latent, recover_size = self.forward_encoder(imgs)
        snr = 10**(snr/10)
        x_power = torch.mean(torch.abs(latent)**2,dim=1,keepdim=True)
        n_power = x_power /snr
        noise = torch.randn_like(latent)*torch.sqrt(n_power) #(bs L embed_size)*(bs 1 1)
        x = latent+noise
        pred = self.forward_decoder(x,recover_size)
        
        if  AttackLables != None:
            # print(AttackLables.shape)
            # print(pred.shape)
            loss = self.forward_loss(AttackLables, pred)
            # print('yes')
        else:
            # print(1)
            loss = self.forward_loss(imgs, pred)
            # print('no')

        return pred,loss

    def forward(self, imgs, PoisionRatio,AttackLables,snr=[-10,15],test:bool = False,Backdoor:bool=False,clean:bool=False,Channel='AWGN'):
        '''input:
        imgs: [N, 3, H, W]原始图像
        PoisionRatio:Tensor [0,1]
        AttackLable:[1 3 H W]
        mask_ratio:Tensor [0,1]
            function:
        encoder 对imag(bs 244 244 3)进行位置编码-->(bs 197 embed_size)-->若干线下层-->输出latent(bs 196 embed_size)
                encoder部分直接略去了mask部分 返回mask 和ids_restore =None
        channel 对latent进行加噪bs的前Poisionnum个样本加入(-10 -15)dbs的噪声 后bs的后面部分加入(20 22)dB的噪声
        decoder 对latent进行解码-->输出pred(bs 197 decoder_embed_size)
        '''
        # print(PoisionRatio)
        #对输入图像进行embeding 然后经过若干attention层 输出latent
        #latent: [bs 196*mask_ratio embed_size] 并且dim=1维度是被打乱的
        z, recover_size = self.forward_encoder(imgs)
        #对encoder层输出添加高斯白噪声
        z_hat = self.forward_channel(z,PoisionRatio,snr=snr,test=test,Backdoor=Backdoor,clean =clean,Channel=Channel)
        # print(z_hat.shape)
        # z_hat = z
        pred = self.forward_decoder(z_hat,recover_size)  # [N, L, p*p*3]
        if clean:
            imgs_back = imgs
        else:
            imgs_back = self.mack_backdoor_imgs(imgs,AttackLables,PoisionRatio,test=test,Backdoor=Backdoor)
        # print(imgs.shape)
        # print(AttackLables.shape)
        # print(imgs_back.shape)
        # print(pred.shape)
        
        loss = self.forward_loss(imgs_back, pred)
        return loss, pred


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,#embed_dim =768 num_heads =12
        decoder_embed_dim=768, decoder_depth=8, decoder_num_heads=16, #decoder_embed_dim =512 decoder_depth =8 decoder_num_heads =16
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def jscc_patch16_dec128d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, Compress_dim=2*128,#embed_dim =768 num_heads =12
        decoder_embed_dim=768, decoder_depth=8, decoder_num_heads=16, #decoder_embed_dim =512 decoder_depth =8 decoder_num_heads =16
        mlp_ratio=4,norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
def jscc_patch16_dec256d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, Compress_dim=2*256,#embed_dim =768 num_heads =12
        decoder_embed_dim=768, decoder_depth=8, decoder_num_heads=16, #decoder_embed_dim =512 decoder_depth =8 decoder_num_heads =16
        mlp_ratio=4,norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def jscc_patch16_dec384d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, Compress_dim=2*384,#embed_dim =768 num_heads =12
        decoder_embed_dim=768, decoder_depth=8, decoder_num_heads=16, #decoder_embed_dim =512 decoder_depth =8 decoder_num_heads =16
        mlp_ratio=4,norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
#这个是原来的jscc结构 1/3的encoder有36层
def jscc_path14_dec348d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=256, depth=32, num_heads=16,
        decoder_embed_dim=256, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model    

vit_jscc_R1_6_patch16 = jscc_patch16_dec128d8b
vit_jscc_R1_3_patch16 = jscc_patch16_dec256d8b
vit_jscc_R1_2_patch16 = jscc_patch16_dec384d8b
vit_jscc_lager_patch16 = jscc_path14_dec348d8b



if __name__ == '__main__':
    model =  vit_jscc_lager_patch16()
    x = torch.rand(60, 3, 224, 224)
    attackimage = torch.rand(1, 3, 224, 224)
    _,y = model(x,PoisionRatio=0.1,AttackLables=attackimage,snr=[-10,15],clean=False,Channel="rician")
    print(y.shape)

