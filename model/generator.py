from .module import *
import torch as t
from torch import nn
from functools import partial
from .module import InverseAdaptiveInstanceNorm


class RandomLightGenerator(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 l_dim,
                 z_dim,
                 num_downsample,
                 num_resblock,
                 ngf,
                 padding_mode='reflect',
                 max_channel=256,
                 actv=nn.ReLU,
                 latent_size=(8, 8),
                 n_bottleneck=3,
                 bottleneck_dim=256,
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

        self.encode_conv = self.build_encode_conv(num_downsample, n_channel, padding_mode, norm_layer, actv)
        self.encode_linear = self.build_encode_linear(n_bottleneck, bottleneck_dim, latent_size, res_channel, z_dim, actv)

        self.decode_conv = self.build_decode_conv(num_downsample, n_channel, padding_mode, norm_layer, actv)
        self.decode_linear = self.build_decode_linear(n_bottleneck, bottleneck_dim, latent_size, res_channel, l_dim, z_dim, actv)
        self.l_dim = l_dim
        self.z_dim = z_dim

        self.to_rgb = nn.Sequential(nn.Conv2d(ngf, ngf, 3, 1, 1),
                                    norm_layer(ngf),
                                    actv(),
                                    nn.Conv2d(ngf, output_dim, 1, 1, 0),
                                    nn.Tanh())

        resblock = []
        for i in range(num_resblock):
            resblock += [ResnetBlock(res_channel, res_channel, norm_layer, actv, padding_mode=padding_mode)]
        self.resblock_encode = nn.Sequential(*resblock[:len(resblock) // 2])
        self.resblock_decode = nn.Sequential(*resblock[len(resblock) // 2:])

        self.compression = nn.AdaptiveAvgPool2d(latent_size)
        self.latent_size = latent_size
        self.n_channel = res_channel

    def build_encode_linear(self, n_bottleneck, bottleneck_dim, latent_size, n_channel, z_dim, actv):
        linear_encode = [nn.Linear(latent_size[0] * latent_size[1] * n_channel, bottleneck_dim),
                         actv()]
        for i in range(n_bottleneck):
            linear_encode += [nn.Linear(bottleneck_dim, bottleneck_dim),
                              actv()]
        linear_encode += [nn.Linear(bottleneck_dim, z_dim),
                          nn.Tanh()]
        return nn.Sequential(*linear_encode)

    def build_decode_linear(self, n_bottleneck, bottleneck_dim, latent_size, n_channel, l_dim, z_dim, actv):
        linear_decode = [nn.Linear((l_dim + z_dim), bottleneck_dim),
                         actv()]
        for i in range(n_bottleneck):
            linear_decode += [nn.Linear(bottleneck_dim, bottleneck_dim),
                              actv()]
        linear_decode += [nn.Linear(bottleneck_dim, latent_size[0] * latent_size[1] * n_channel),
                         actv()]
        return nn.Sequential(*linear_decode)

    def build_encode_conv(self, num_downsample, n_channel, padding_mode, norm_layer, actv):
        block = []
        for i in range(num_downsample):
            block += [nn.Conv2d(n_channel[i][0], n_channel[i][1], 3, 2, 1, padding_mode=padding_mode,),
                      norm_layer(n_channel[i][1]),
                      actv()]
        return nn.Sequential(*block)


    def build_decode_conv(self, num_downsample, n_channel, padding_mode, norm_layer, actv):
        decoder = []
        for i in range(num_downsample):
            decoder += [nn.Upsample(scale_factor=2),
                        nn.Conv2d(n_channel[- i - 1][1], n_channel[- i - 1][0], 3, 1, 1, padding_mode=padding_mode, ),
                        norm_layer(n_channel[- i - 1][0]),
                        actv()]
        return nn.Sequential(*decoder)

    def forward(self, img, l_vec):
        x = self.from_rgb(img)
        x = self.encode_conv(x)
        x = self.resblock_encode(x)
        x_shape = x.shape
        x = self.compression(x)
        x = x.view(x.shape[0], -1)
        z_vec = self.encode_linear(x)
        x = th.cat([z_vec, l_vec], dim=-1)
        x = self.decode_linear(x)
        x = x.view(x.shape[0], -1, *self.latent_size)
        x = F.interpolate(x, size=x_shape[2:], mode='nearest')
        x = self.resblock_decode(x)
        x = self.decode_conv(x)
        x = self.to_rgb(x)
        return x, z_vec


class StudioLightGenerator(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 l_dim,
                 z_dim,
                 num_downsample,
                 num_resblock,
                 ngf,
                 padding_mode='reflect',
                 max_channel=256,
                 actv=nn.ReLU,
                 latent_size=(8, 8),
                 n_bottleneck=3,
                 bottleneck_dim=256,
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
            nn.Conv2d(input_dim, ngf, kernel_size=7, stride=1, padding=3, padding_mode=padding_mode, ),
            norm_layer(ngf),
            actv())

        self.encode_conv = self.build_encode_conv(num_downsample, n_channel, padding_mode, norm_layer, actv)
        self.encode_linear = self.build_encode_linear(n_bottleneck, bottleneck_dim, latent_size, res_channel, l_dim, z_dim, actv)

        self.decode_conv = self.build_decode_conv(num_downsample, n_channel, padding_mode, norm_layer, actv)
        self.decode_linear = self.build_decode_linear(n_bottleneck, bottleneck_dim, latent_size, res_channel, z_dim, actv)
        self.l_dim = l_dim
        self.z_dim = z_dim

        self.to_rgb = nn.Sequential(nn.Conv2d(ngf, ngf, 3, 1, 1),
                                    norm_layer(ngf),
                                    actv(),
                                    nn.Conv2d(ngf, output_dim, 1, 1, 0),
                                    nn.Tanh())

        resblock = []
        for i in range(num_resblock):
            resblock += [ResnetBlock(res_channel, res_channel, norm_layer, actv, padding_mode=padding_mode)]
        self.resblock_encode = nn.Sequential(*resblock[:len(resblock) // 2])
        self.resblock_decode = nn.Sequential(*resblock[len(resblock) // 2:])

        self.compression = nn.AdaptiveAvgPool2d(latent_size)
        self.latent_size = latent_size
        self.n_channel = res_channel

    def build_encode_linear(self, n_bottleneck, bottleneck_dim, latent_size, n_channel, l_dim, z_dim, actv):
        linear_encode = [nn.Linear(latent_size[0] * latent_size[1] * n_channel, bottleneck_dim),
                         actv()]
        for i in range(n_bottleneck):
            linear_encode += [nn.Linear(bottleneck_dim, bottleneck_dim),
                              actv()]
        linear_encode += [nn.Linear(bottleneck_dim, (l_dim + z_dim)),
                          nn.Tanh()]
        return nn.Sequential(*linear_encode)

    def build_decode_linear(self, n_bottleneck, bottleneck_dim, latent_size, n_channel, z_dim, actv):
        linear_decode = [nn.Linear(z_dim, bottleneck_dim),
                         actv()]
        for i in range(n_bottleneck):
            linear_decode += [nn.Linear(bottleneck_dim, bottleneck_dim),
                              actv()]
        linear_decode = [nn.Linear(bottleneck_dim, latent_size[0] * latent_size[1] * n_channel),
                         actv()]
        return nn.Sequential(*linear_decode)

    def build_encode_conv(self, num_downsample, n_channel, padding_mode, norm_layer, actv):
        block = []
        for i in range(num_downsample):
            block += [nn.Conv2d(n_channel[i][0], n_channel[i][1], 3, 2, 1, padding_mode=padding_mode,),
                      norm_layer(n_channel[i][1]),
                      actv()]
        return nn.Sequential(*block)


    def build_decode_conv(self, num_downsample, n_channel, padding_mode, norm_layer, actv):
        decoder = []
        for i in range(num_downsample):
            decoder += [nn.Upsample(scale_factor=2),
                        nn.Conv2d(n_channel[- i - 1][1], n_channel[- i - 1][0], 3, 1, 1, padding_mode=padding_mode, ),
                        norm_layer(n_channel[- i - 1][0]),
                        actv()]
        return nn.Sequential(*decoder)

    def forward(self, img):
        x = self.from_rgb(img)
        x = self.encode_conv(x)
        x = self.resblock_encode(x)
        x_shape = x.shape
        x = self.compression(x)
        x = x.view(x.shape[0], -1)
        x = self.encode_linear(x)
        z_vec, l_vec = th.split(x, (self.z_dim, self.l_dim), dim=-1)
        x = self.decode_linear(z_vec)
        x = x.view(x.shape[0], -1, *self.latent_size)
        x = F.interpolate(x, size=x_shape[2:], mode='nearest')
        x = self.resblock_decode(x)
        x = self.decode_conv(x)
        x = self.to_rgb(x)
        return x, z_vec, l_vec
