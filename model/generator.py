from .module import *
import torch as t
from torch import nn
from functools import partial
from .module import InverseAdaptiveInstanceNorm


class RandomLightGenerator(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 noise_dim,
                 num_downsample,
                 num_resblock,
                 ngf,
                 padding_mode='reflect',
                 max_channel=256,
                 actv=nn.ReLU,
                 fc_dim=128,
                 norm_layer=partial(nn.InstanceNorm2d, affine=True)):
        super(RandomLightGenerator, self).__init__()
        self.num_downsample = num_downsample
        n_channel = []
        num_c = ngf
        for i in range(num_downsample):
            n_channel += [[min(num_c, max_channel), min(num_c * 2, max_channel)]]
            num_c *= 2

        for i in range(num_downsample):
            if i == 0:
                linear = nn.Sequential(nn.Linear(noise_dim, fc_dim),
                                       actv())
            else:
                linear = nn.Sequential(nn.Linear(fc_dim, fc_dim),
                                       actv())
            setattr(self, f'linear_{i}', linear)

        res_channel = n_channel[-1][-1]

        self.from_rgb = nn.Sequential(
            nn.Conv2d(input_dim, ngf, kernel_size=7, stride=1, padding=3, padding_mode=padding_mode, ),
            norm_layer(ngf),
            actv())

        self.encode = self.build_encode_block(num_downsample, n_channel, padding_mode, norm_layer, actv)

        self.build_decode_block(num_downsample, n_channel, padding_mode, actv, fc_dim)

        self.to_rgb = nn.Sequential(nn.Conv2d(ngf, output_dim, 1, 1, 0),
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


    def build_decode_block(self, num_downsample, n_channel, padding_mode, actv, fc_dim):
        for i in range(num_downsample):
            deconv = [nn.Upsample(scale_factor=2),
                      nn.Conv2d(n_channel[-i - 1][1], n_channel[- i - 1][0], 3, 1, 1, padding_mode=padding_mode, )]
            norm_layer = AdaptiveInstanceNorm(n_channel[-i - 1][0], fc_dim)
            setattr(self, f'decode_conv_{i}', nn.Sequential(*deconv))
            setattr(self, f'decode_norm_{i}', norm_layer)
            setattr(self, f'decode_actv_{i}', actv())

    def forward(self, img, light_vec):
        x = self.from_rgb(img)
        x = self.encode(x)
        x = self.resblock(x)
        z = light_vec
        for i in range(self.num_downsample):
            decode_conv = getattr(self, f'decode_conv_{i}')
            decode_norm = getattr(self, f'decode_norm_{i}')
            decode_actv = getattr(self, f'decode_actv_{i}')
            linear = getattr(self, f'linear_{i}')
            z = linear(z)
            x = decode_conv(x)
            x = decode_norm(x, z)
            x = decode_actv(x)
        x = self.to_rgb(x)
        return x


class StudioLightGenerator(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 noise_dim,
                 num_downsample,
                 num_resblock,
                 ngf,
                 padding_mode='reflect',
                 max_channel=256,
                 actv=nn.ReLU,
                 norm_layer=partial(nn.InstanceNorm2d, affine=True)):
        super(StudioLightGenerator, self).__init__()
        self.num_downsample = num_downsample
        n_channel = []
        num_c = ngf
        for i in range(num_downsample):
            n_channel += [[min(num_c, max_channel), min(num_c * 2, max_channel)]]
            num_c *= 2


        for i in range(num_downsample):
            if i == num_downsample - 1:
                linear = nn.Sequential(nn.Linear(n_channel[i][1] * 3, noise_dim),
                                       nn.Tanh())
            elif i == 0:
                linear = nn.Sequential(nn.Linear(n_channel[i + 1][0] * 2, n_channel[i + 1][1]),
                                       actv())
            else:
                linear = nn.Sequential(nn.Linear(n_channel[i + 1][0] * 3, n_channel[i + 1][1]),
                                       actv())
            setattr(self, f'linear_{i}', linear)

        res_channel = n_channel[-1][-1]

        self.from_rgb = nn.Sequential(
            nn.Conv2d(input_dim, ngf, kernel_size=7, stride=1, padding=3, padding_mode=padding_mode, ),
            norm_layer(ngf),
            actv())

        self.build_encode_block(num_downsample, n_channel, padding_mode, actv)

        self.decode = self.build_decode_block(num_downsample, n_channel, padding_mode,  norm_layer, actv)

        self.to_rgb = nn.Sequential(nn.Conv2d(ngf, output_dim, 1, 1, 0),
                                    nn.Tanh())

        resblock = []
        for i in range(num_resblock):
            resblock += [ResnetBlock(res_channel, res_channel, norm_layer, actv, padding_mode=padding_mode)]
        self.resblock = nn.Sequential(*resblock)

    def build_encode_block(self, num_downsample, n_channel, padding_mode, actv):
        for i in range(num_downsample):
            encode_conv = nn.Conv2d(n_channel[i][0], n_channel[i][1], 3, 2, 1, padding_mode=padding_mode, )
            encode_norm = InverseAdaptiveInstanceNorm(n_channel[i][1])
            encode_actv = actv()
            setattr(self, f'encode_conv_{i}', encode_conv)
            setattr(self, f'encode_norm_{i}', encode_norm)
            setattr(self, f'encode_actv_{i}', encode_actv)


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
        for i in range(self.num_downsample):
            conv = getattr(self, f'encode_conv_{i}')
            norm = getattr(self, f'encode_norm_{i}')
            actv = getattr(self, f'encode_actv_{i}')
            linear = getattr(self, f'linear_{i}')
            x = conv(x)
            x, gamma, beta = norm(x)
            if i == 0:
                z = t.cat([gamma, beta], -1).view(gamma.shape[0], -1)
                z = linear(z)
            else:
                affine = t.cat([gamma, beta], -1).view(gamma.shape[0], -1)
                z = t.cat([z, affine], -1)
                z = linear(z)
            x = actv(x)
        x = self.resblock(x)
        x = self.decode(x)
        x = self.to_rgb(x)
        return x, z
