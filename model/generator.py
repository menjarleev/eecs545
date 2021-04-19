from .module import *
import torch as t
from torch import nn
from functools import partial
from .module import InverseAdaptiveInstanceNorm


class RandomLightGenerator(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 z_dim,
                 num_downsample,
                 num_resblock,
                 ngf,
                 padding_mode='reflect',
                 max_channel=256,
                 bottle_neck_z=256,
                 conv_num_z=2,
                 latent_size=(8, 8),
                 actv=nn.ReLU,
                 norm_layer=partial(nn.InstanceNorm2d, affine=True)):
        super(RandomLightGenerator, self).__init__()
        self.num_downsample = num_downsample
        n_channel = []
        num_c = ngf
        for i in range(num_downsample):
            n_channel += [[min(num_c, max_channel), min(num_c * 2, max_channel)]]
            num_c *= 2
        self.latent_size = latent_size
        self.z_channel = z_channel = n_channel[-1][-1] // (conv_num_z**2)
        self.from_z_linear = nn.Sequential(nn.Linear(z_dim, bottle_neck_z),
                                           actv(),
                                           nn.Linear(bottle_neck_z, latent_size[0] * latent_size[1] * z_channel))
        from_z_conv = []
        for i in range(conv_num_z):
            from_z_conv += [nn.Conv2d(z_channel, z_channel * 2, 3, 1, 1, padding_mode=padding_mode),
                            norm_layer(z_channel * 2),
                            actv()]
            z_channel = z_channel * 2

        self.from_z_conv = nn.Sequential(*from_z_conv)

        res_channel = n_channel[-1][-1]

        self.from_rgb = nn.Sequential(
            nn.Conv2d(input_dim, ngf, kernel_size=7, stride=1, padding=3, padding_mode=padding_mode, ),
            norm_layer(ngf),
            actv())

        self.encode = self.build_encode_block(num_downsample, n_channel, padding_mode, norm_layer, actv)

        self.build_decode_block(num_downsample, n_channel, padding_mode, actv, z_channel)

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

    def forward(self, img, z):
        x = self.from_rgb(img)
        x = self.encode(x)
        res = self.resblock(x)
        x = res
        z = self.from_z_linear(z).view(x.shape[0], self.z_channel, *self.latent_size)
        z = F.interpolate(z, x.shape[2:], mode='nearest')
        z = self.from_z_conv(z)

        for i in range(self.num_downsample):
            decode_conv = getattr(self, f'decode_conv_{i}')
            decode_norm = getattr(self, f'decode_norm_{i}')
            decode_actv = getattr(self, f'decode_actv_{i}')
            x = decode_conv(x)
            x = decode_norm(x, z)
            x = decode_actv(x)
        x = self.to_rgb(x)
        return x, res


class StudioLightGenerator(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 z_dim,
                 num_downsample,
                 num_resblock,
                 ngf,
                 latent_size=(128, 128),
                 bottle_neck_z=256,
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

        self.latent_size = latent_size
        z_channel = n_channel[-1][-1]
        to_z_conv = [nn.AdaptiveMaxPool2d(latent_size)]
        for i in range(2):
            to_z_conv += [nn.Conv2d(z_channel, z_channel // 2, 3, 1, 1, padding_mode=padding_mode),
                          norm_layer(z_channel // 2),
                          actv()]
            z_channel = z_channel // 2
        self.to_z_conv = nn.Sequential(*to_z_conv)
        self.to_z_linear = nn.Sequential(nn.Linear(z_channel * latent_size[0] * latent_size[1], bottle_neck_z),
                                          actv(),
                                          nn.Linear(bottle_neck_z, z_dim),
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
        z = self.to_z_conv(z).view(z.shape[0], -1)
        z = self.to_z_linear(z)
        res = self.resblock(x)
        x = self.decode(res)
        x = self.to_rgb(x)
        return x, z, res
