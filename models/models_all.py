from functools import partial

import torch.nn as nn

from model_BDJSCC import DeepJSCC


def _masked_autoencoder_vit():
    from models_mae import MaskedAutoencoderViT

    return MaskedAutoencoderViT


def vit_jscc_patch16_dec384d8b(**kwargs):
    return _masked_autoencoder_vit()(
        patch_size=16,
        embed_dim=384,
        depth=32,
        num_heads=16,
        decoder_embed_dim=384,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def vit_jscc_patch16_dec256d8b(**kwargs):
    return _masked_autoencoder_vit()(
        patch_size=16,
        embed_dim=256,
        depth=36,
        num_heads=16,
        decoder_embed_dim=256,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


vit_jscc_lager_1_4_patch16 = vit_jscc_patch16_dec384d8b
vit_jscc_lager_1_6_patch16 = vit_jscc_patch16_dec256d8b


def BDJSCC_c16_awgn(**kwargs):
    model = DeepJSCC(c=16, channel_type='AWGN')
    return model


bdjscc_R1_3 = BDJSCC_c16_awgn
