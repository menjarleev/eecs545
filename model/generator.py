from .module import *
import torch as t
from torch import nn
from functools import partial
import torch.nn.functional as F


class RandomLightGenerator(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 num_downsample,
                 num_resblock,
                 ngf,
                 lc_dim=(16, 8, 8),
                 padding_mode='reflect',
                 max_channel=256,
                 actv=nn.LeakyReLU,
                 norm_layer=partial(nn.InstanceNorm2d, affine=True)):
        super(RandomLightGenerator, self).__init__()
        self.num_downsample = num_downsample
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

        self.encode = self.build_encode_block(num_downsample, n_channel, padding_mode, norm_layer, actv)

        self.build_decode_block(num_downsample, n_channel, padding_mode, actv, lc_dim[0])

        self.to_rgb = nn.Sequential(nn.Conv2d(ngf, ngf, 3, 1, 1),
                                    norm_layer(ngf),
                                    actv(),
                                    nn.Conv2d(ngf, output_dim, 1, 1, 0),
                                    nn.Tanh())

        resblock = []
        for i in range(num_resblock):
            resblock += [ResnetBlock(res_channel, res_channel, norm_layer, actv, padding_mode=padding_mode)]
        self.resblock = nn.Sequential(*resblock)

    def build_encode_block(self, num_downsample, n_channel, padding_mode, norm_layer, actv):
        block = []
        for i in range(num_downsample):
            block += [nn.Conv2d(n_channel[i][0], n_channel[i][1], 3, 2, 1, padding_mode=padding_mode,),
                      norm_layer(n_channel[i][1]),
                      actv()]
        return nn.Sequential(*block)


    def build_decode_block(self, num_downsample, n_channel, padding_mode, actv, label_nc):
        for i in range(num_downsample):
            deconv = [nn.Upsample(scale_factor=2),
                      nn.Conv2d(n_channel[- i - 1][1], n_channel[- i - 1][0], 3, 1, 1, padding_mode=padding_mode)]
            norm_layer = SPADE(n_channel[- i - 1][0], label_nc)
            setattr(self, f'decode_conv_{i}', nn.Sequential(*deconv))
            setattr(self, f'decode_norm_{i}', norm_layer)
            setattr(self, f'decode_actv_{i}', actv())

    def forward(self, img, segmap):
        x = self.from_rgb(img)
        x = self.encode(x)
        res = self.resblock(x)
        x = res
        for i in range(self.num_downsample):
            decode_conv = getattr(self, f'decode_conv_{i}')
            decode_norm = getattr(self, f'decode_norm_{i}')
            decode_actv = getattr(self, f'decode_actv_{i}')
            x = decode_conv(x)
            x = decode_norm(x, segmap)
            x = decode_actv(x)
        x = self.to_rgb(x)
        return x


class StudioLightGenerator(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 num_downsample,
                 num_resblock,
                 ngf,
                 lc_dim=(16, 8, 8),
                 padding_mode='reflect',
                 max_channel=256,
                 actv=nn.LeakyReLU,
                 norm_layer=partial(nn.InstanceNorm2d, affine=True)):
        super(StudioLightGenerator, self).__init__()
        self.num_downsample = num_downsample
        self.lc_dim = lc_dim
        n_channel = []
        num_c = ngf
        for i in range(num_downsample):
            n_channel += [[min(num_c, max_channel), min(num_c * 2, max_channel)]]
            num_c *= 2

        res_channel = n_channel[-1][-1]

        self.from_rgb = nn.Sequential(
            nn.Conv2d(input_dim, 2 * ngf, kernel_size=7, stride=1, padding=3, padding_mode=padding_mode, ),
            norm_layer(2 * ngf),
            actv())

        self.encode = self.build_encode_block(num_downsample, n_channel, padding_mode, actv)
        self.actv_x = actv()
        self.actv_z = nn.Sequential(
            nn.Conv2d(res_channel, lc_dim[0], 3, 1, 1, padding_mode=padding_mode),
            norm_layer(lc_dim[0]),
            nn.Tanh())

        self.decode = self.build_decode_block(num_downsample, n_channel, padding_mode,  norm_layer, actv)

        self.to_rgb = nn.Sequential(nn.Conv2d(ngf, output_dim, 1, 1, 0),
                                    nn.Tanh())

        resblock = []
        for i in range(num_resblock):
            resblock += [ResnetBlock(res_channel, res_channel, norm_layer, actv, padding_mode=padding_mode)]
        self.resblock = nn.Sequential(*resblock)

    def build_encode_block(self, num_downsample, n_channel, padding_mode, actv):
        encode = []
        for i in range(num_downsample):
            if i == num_downsample - 1:
                encode += [nn.Conv2d(2 * n_channel[i][0], 2 * n_channel[i][1], 3, 2, 1, padding_mode=padding_mode, ),
                           nn.InstanceNorm2d(2 * n_channel[i][1], affine=True)]
            else:
                encode += [nn.Conv2d(2 * n_channel[i][0], 2 * n_channel[i][1], 3, 2, 1, padding_mode=padding_mode, ),
                           nn.InstanceNorm2d(2 * n_channel[i][1], affine=True),
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
        x, z = t.chunk(self.encode(x), chunks=2, dim=1)
        x = self.actv_x(x)
        z = self.actv_z(z)
        z = F.interpolate(z, size=self.lc_dim[1:])
        res = self.resblock(x)
        x = self.decode(res)
        x = self.to_rgb(x)
        return x, z

class LightConditionGenerator(nn.Module):
    def __init__(self, noise_dim, lc_dim, bottleneck_dim, n_bottleneck, n_block=5, actv=nn.LeakyReLU):
        super(LightConditionGenerator, self).__init__()
        self.generator = Generator(n_block, 256, lc_dim[0], lc_dim[1:], bottleneck_dim)
        self.lc_dim = lc_dim
        self.n_block = n_block
        layer = [nn.Linear(noise_dim, bottleneck_dim),
                 actv()]
        for i in range(n_bottleneck):
            layer += [nn.Linear(bottleneck_dim, bottleneck_dim),
                      actv()]
        self.layer = nn.Sequential(*layer)
        self.lc_dim = lc_dim

    def forward(self, input, noise=None):
        b_size = input.shape[0]
        style = self.layer(input)
        if noise is None:
            noise = []
            for i in range(self.n_block):
                noise += [torch.randn(b_size, 1, *self.lc_dim[1:], device=input.device)]
        out = self.generator(style, noise)

        return out

class LightConditionVAE(nn.Module):
    def __init__(self,
                 lc_dim,
                 latent_dim,
                 hidden_dims=None,
                 **kwargs) -> None:
        super(LightConditionVAE, self).__init__()

        self.lc_dim = lc_dim
        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 256]

        in_channels = lc_dim[0]
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=h_dim,
                              kernel_size=3, stride=1, padding=1),
                    nn.InstanceNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * lc_dim[1] * lc_dim[2], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * lc_dim[1] * lc_dim[2], latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * lc_dim[1] * lc_dim[2])

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(hidden_dims[i],
                              hidden_dims[i + 1],
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              ),
                    nn.InstanceNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )


        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.Conv2d(hidden_dims[-1],
                      hidden_dims[-1],
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      ),
            nn.InstanceNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=lc_dim[0],
                      kernel_size=3, padding=1),
            nn.Tanh())

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
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
        result = result.view(-1, 256, *self.lc_dim[1:])
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
        return [self.decode(z), input, mu, log_var]


    def sample(self,
               num_samples:int,
               current_device: int, **kwargs):
        z = torch.randn(num_samples, self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x, **kwargs):
        return self.forward(x)[0]