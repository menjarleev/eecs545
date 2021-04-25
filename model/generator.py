from .module import *
import math
import torch
from torch import nn
from functools import partial
import torch.nn.functional as F


class RandomLightGenerator(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 lc_nc,
                 num_downsample,
                 num_resblock,
                 ngf,
                 padding_mode='reflect',
                 max_channel=256,
                 actv=nn.LeakyReLU,
                 norm_layer=partial(nn.InstanceNorm2d, affine=True)):
        super(RandomLightGenerator, self).__init__()
        n_channel = []
        num_c = ngf
        for i in range(num_downsample):
            n_channel += [[min(num_c, max_channel), min(num_c * 2, max_channel)]]
            num_c *= 2


        res_channel = n_channel[-1][-1]

        self.from_rgb = nn.Sequential(
            nn.Conv2d(input_dim, ngf, kernel_size=7, stride=1, padding=3, padding_mode=padding_mode, ),
            norm_layer(ngf),
            actv())

        self.encode = self.build_encode_block(num_downsample, n_channel, padding_mode, actv)

        self.decode = self.build_decode_block(num_downsample, n_channel, padding_mode, actv, lc_nc)

        self.to_rgb = nn.Sequential(nn.Conv2d(ngf, ngf, 3, 1, 1),
                                    norm_layer(ngf),
                                    actv(),
                                    nn.Conv2d(ngf, output_dim, 1, 1, 0),
                                    nn.Tanh())

        resblock = []
        for i in range(num_resblock):
            resblock += [ResnetBlock(res_channel, res_channel, norm_layer, actv, padding_mode=padding_mode)]
        self.resblock = nn.ModuleList(resblock)

    def build_encode_block(self, num_downsample, n_channel,  padding_mode, actv):
        encode = []
        for i in range(num_downsample):
            if i == num_downsample - 1:
                encode += [nn.Conv2d(n_channel[i][0], n_channel[i][1], 3, 2, 1, padding_mode=padding_mode, ),
                           nn.InstanceNorm2d(n_channel[i][1], affine=True)]
            else:
                encode += [nn.Conv2d(n_channel[i][0], n_channel[i][1], 3, 2, 1, padding_mode=padding_mode, ),
                           nn.InstanceNorm2d(n_channel[i][1], affine=True),
                           actv()]
        return nn.Sequential(*encode)


    def build_decode_block(self, num_downsample, n_channel, padding_mode, actv, label_nc):
        block = []
        for i in range(num_downsample):
            block += [SPADEConvTransposeBlock(n_channel[- i - 1][1], n_channel[- i - 1][0], label_nc, 2, 3, 1, 1, padding_mode=padding_mode, actv=actv)]
        return nn.ModuleList(block)

    def forward(self, img, segmap):
        x = self.from_rgb(img)
        for encode_i in self.encode:
            x = encode_i(x)
        for res_i in self.resblock:
            x = res_i(x)
        for decode_i in self.decode:
            x = decode_i(x, segmap)
        x = self.to_rgb(x)
        return x

class StudioLightGenerator(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 lc_nc,
                 num_downsample,
                 num_resblock,
                 ngf,
                 padding_mode='reflect',
                 max_channel=256,
                 actv=nn.LeakyReLU,
                 norm_layer=partial(nn.InstanceNorm2d, affine=True)):
        super(StudioLightGenerator, self).__init__()
        n_channel = []
        num_c = ngf
        for i in range(num_downsample):
            n_channel += [[min(num_c, max_channel), min(num_c * 2, max_channel)]]
            num_c *= 2

        res_channel = n_channel[-1][-1]

        self.from_rgb = nn.Sequential(
            nn.Conv2d(input_dim, ngf, kernel_size=7, stride=1, padding=3, padding_mode=padding_mode, ),
            norm_layer(ngf),
            actv())

        self.encode = self.build_encode_block(num_downsample, n_channel, padding_mode, actv)

        self.decode_z = self.build_decode_block(num_downsample, n_channel, padding_mode, norm_layer, actv)
        self.out_z = nn.Sequential(nn.Conv2d(ngf, lc_nc, 1, 1, 0),
                                   nn.Tanh())

        self.decode_x = self.build_decode_block(num_downsample, n_channel, padding_mode,  norm_layer, actv)

        self.out_x = nn.Sequential(nn.Conv2d(ngf, output_dim, 1, 1, 0),
                                 nn.Tanh())
        self.split = [output_dim, lc_nc]

        resblock = []
        for i in range(num_resblock):
            resblock += [ResnetBlock(res_channel, res_channel, norm_layer, actv, padding_mode=padding_mode)]
        self.resblock = nn.Sequential(*resblock)

    def build_encode_block(self, num_downsample, n_channel, padding_mode, actv):
        encode = []
        for i in range(num_downsample):
            if i == num_downsample - 1:
                encode += [nn.Conv2d(n_channel[i][0], n_channel[i][1], 3, 2, 1, padding_mode=padding_mode, ),
                           nn.InstanceNorm2d(n_channel[i][1], affine=True)]
            else:
                encode += [nn.Conv2d(n_channel[i][0], n_channel[i][1], 3, 2, 1, padding_mode=padding_mode, ),
                           nn.InstanceNorm2d(n_channel[i][1], affine=True),
                           actv()]
        return nn.Sequential(*encode)


    def build_decode_block(self, num_downsample, n_channel, padding_mode, norm_layer, actv):
        decoder = []
        for i in range(num_downsample):
            decoder += [nn.Upsample(scale_factor=2),
                        nn.Conv2d(n_channel[- i - 1][1], n_channel[- i - 1][0], 3, 1, 1, padding_mode=padding_mode, ),
                        norm_layer(n_channel[- i - 1][0]),
                        actv()]
        return nn.Sequential(*decoder)

    def forward(self, img):
        x = self.from_rgb(img)
        x = self.encode(x)
        z = self.decode_z(x)
        z = self.out_z(z)
        res = self.resblock(x)
        x = self.decode_x(res)
        x = self.out_x(x)
        return x, z

class StyleGANGenerator(nn.Module):
    def __init__(self, lc_nc, input_size, n_mlp, init_size=4, code_dim=512, actv=nn.LeakyReLU):
        super(StyleGANGenerator, self).__init__()
        self.code_dim = code_dim
        self.n_block = int(math.log2(input_size[0] // init_size))
        self.generator = Generator(lc_nc, input_size, init_size, code_dim)
        layer = [nn.Linear(code_dim, code_dim),
                 actv()]
        for i in range(n_mlp):
            layer += [nn.Linear(code_dim, code_dim),
                      actv()]
        self.layer = nn.Sequential(*layer)

    def forward(self, input, noise=None):
        b_size = input.shape[0]
        style = self.layer(input)
        if noise is None:
            noise = []
            for i in range(self.n_block + 1):
                size = 4 * 2**i
                noise += [torch.randn(b_size, 1, size, size, device=input.device)]
        out = self.generator(style, noise)

        return out

    def sample(self, num_sample, current_device):
        z = torch.randn(num_sample, self.code_dim).to(current_device)
        x = self.forward(z)
        return x

class LightConditionVAE(nn.Module):
    def __init__(self,
                 input_size,
                 lc_nc,
                 latent_dim,
                 hidden_dims=None,
                 **kwargs) -> None:
        super(LightConditionVAE, self).__init__()


        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 256]

        self.input_size = input_size
        self.latent_dim = latent_dim
        self.reshape_dim = [hidden_dims[-1], input_size[0]//(2 ** len(hidden_dims)), input_size[1]//(2 ** len(hidden_dims))]

        in_channels = lc_nc
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.InstanceNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(self.reshape_dim[0] * self.reshape_dim[1] * self.reshape_dim[2], latent_dim)
        self.fc_var = nn.Linear(self.reshape_dim[0] * self.reshape_dim[1] * self.reshape_dim[2], latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, self.reshape_dim[0] * self.reshape_dim[1] * self.reshape_dim[2])

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.InstanceNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )


        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.InstanceNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=lc_nc,
                      kernel_size=3, padding=1),
            nn.Tanh())

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        input = F.interpolate(input, self.input_size)
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, *self.reshape_dim)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, **kwargs):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), F.interpolate(input, self.input_size), mu, log_var]


    def sample(self,
               num_samples:int,
               current_device: int, **kwargs):
        z = torch.randn(num_samples, self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x, **kwargs):
        return self.forward(x)[0]