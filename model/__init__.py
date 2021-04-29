from .discriminator import *
from .generator import *
from .module import *
from .photometricgan import *
import torch.nn as nn
from functools import partial

def define_encoder(opt):
    if opt.actv_G == 'ReLU':
        actv = nn.ReLU
    elif opt.actv_G == 'LeakyReLU':
        actv = partial(nn.LeakyReLU, negative_slope=0.2)
    else:
        raise NotImplementedError
    if opt.encoder == 'vae':
        encoder = LightConditionVAE(opt.lc_dim, opt.latent_dim)
    elif opt.encoder == 'style-gan':
        encoder = StyleGANGenerator(opt.latent_dim, n_mlp=8)
    elif opt.encoder == 'mlp':
        encoder = MLPStyleGenerator(opt.noise_dim, opt.latent_dim, n_layer=8, actv=actv)
    else:
        raise ValueError(f'{opt.encoder} is not defined.')
    return encoder