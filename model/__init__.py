from .discriminator import *
from .generator import *
from .module import *
from .photometricgan import *

def define_encoder(opt):
    if opt.encoder == 'vae':
        encoder = LightConditionVAE(opt.input_size, opt.lc_nc, opt.latent_dim)
    elif opt.encoder == 'style-gan':
        encoder = StyleGANGenerator(opt.lc_nc, opt.input_size, opt.n_mlp)
    else:
        raise ValueError(f'{opt.encoder} is not defined.')
    return encoder