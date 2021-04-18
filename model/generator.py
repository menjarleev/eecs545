from .module import *
import torch as t
from torch import nn
from functools import partial
from .module import InverseAdaptiveInstanceNorm


class RandomLightGenerator(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 noise_nc,
                 num_downsample,
                 num_resblock,
                 ngf,
                 padding_mode='reflect',
                 max_channel=256,
                 actv=nn.ReLU,
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

        self.build_decode_block(num_downsample, n_channel, padding_mode, actv, noise_nc)

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
        return x, res


class StudioLightGenerator(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
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

        res_channel = n_channel[-1][-1]

        self.from_rgb = nn.Sequential(
            nn.Conv2d(input_dim, 2 * ngf, kernel_size=7, stride=1, padding=3, padding_mode=padding_mode, ),
            norm_layer(2 * ngf),
            actv())

        self.encode = self.build_encode_block(num_downsample, n_channel, padding_mode, actv)
        self.actv_x = actv()
        self.actv_z = nn.Tanh()

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
        res = self.resblock(x)
        x = self.decode(res)
        x = self.to_rgb(x)
        return x, z, res
