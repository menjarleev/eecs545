import torch
import torch.nn as nn
from .ops import *
from functools import partial

class SingleScaleDiscriminator(nn.Module):
    def __init__(self,
                 input_nc,
                 ndf,
                 n_layer,
                 padding_mode,
                 actv=nn.LeakyReLU,
                 norm=partial(nn.InstanceNorm2d, affine=True),
                 ):
        super(SingleScaleDiscriminator, self).__init__()
        actv = actv
        norm = norm
        self.n_layer = n_layer
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=3, stride=2, padding_mode=padding_mode),
                     norm(ndf),
                     actv()]]
        nf = ndf
        for n in range(1, self.n_layer):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=3, stride=2, padding_mode=padding_mode),
                norm(nf),
                actv()
            ]]
        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=3, stride=1, padding_mode=padding_mode),
            norm(nf),
            actv()
        ]]
        sequence += [[nn.Conv2d(nf, 1, kernel_size=3, stride=1, padding_mode=padding_mode)]]

        for n in range(len(sequence)):
            setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))

    def forward(self, input):
        res = [input]
        for n in range(self.n_layer + 2):
            model = getattr(self, 'model' + str(n))
            res.append(model(res[-1]))
        return res[1:]

class MultiScaleDiscriminator(nn.Module):
    def __init__(self, input_nc, num_D, n_layer, ndf, padding_mode):
        super(MultiScaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layer
        ndf_max = 64
        ndf = ndf

        for i in range(self.num_D):
            netD = SingleScaleDiscriminator(input_nc, min(ndf_max, ndf * (2 ** (self.num_D - 1 - i))), n_layer, padding_mode)
            for j in range(self.n_layers + 2):
                setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        result = [input]
        for i in range(len(model)):
            result.append(model[i](result[-1]))
        return result[1:]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                     range(self.n_layers + 2)]
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result
