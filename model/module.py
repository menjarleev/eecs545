import torch as th
from torch import nn
from functools import partial
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = th.mean(x, dim=1, keepdim=True)
        max_out, _ = th.max(x, dim=1, keepdim=True)
        x = th.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding_mode='reflect'):
        super(CBAM, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, padding_mode=padding_mode)
        self.channel_attention = ChannelAttention(out_channel)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.conv(x)
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        return x

class RCBAM(nn.Module):
    def __init__(self, in_channel, out_channel, norm_layer, activation=nn.ReLU(True), kernel_size=3, stride=1, padding_mode='reflect'):
        super(RCBAM, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, padding_mode=padding_mode)
        self.channel1 = ChannelAttention(out_channel)
        self.spatial1 = SpatialAttention()
        self.norm1 = norm_layer(out_channel)
        self.act = activation
        self.conv2 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, padding_mode=padding_mode)
        self.channel2 = ChannelAttention(out_channel)
        self.spatial2 = SpatialAttention()
        self.norm2 = norm_layer(out_channel)

    def forward(self, feature):
        x = self.conv1(feature)
        x = self.channel1(x) * x
        x = self.spatial1(x) * x
        x = self.act(self.norm1(x))
        x = self.conv2(x)
        x = self.channel1(x) * x
        x = self.spatial1(x) * x
        x = self.norm2(x)
        return x + feature

class ResnetBlock(nn.Module):
    def __init__(self, in_channel, out_channel,  norm_layer, activation=nn.ReLU(True), kernel_size=3, stride=1, dropout=.0, padding_mode='reflect'):
        super(ResnetBlock, self).__init__()
        self.downscale = None
        if stride > 1:
            self.downscale = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, padding_mode=padding_mode)
        self.res_block = self._build_conv_block(in_channel=in_channel,
                                                out_channel=out_channel,
                                                padding_mode=padding_mode,
                                                norm_layer=norm_layer,
                                                activation=activation,
                                                kernel_size=kernel_size,
                                                stride=stride,
                                                dropout=dropout)


    def _build_conv_block(self, in_channel, out_channel, padding_mode, norm_layer, activation, kernel_size=3, stride=2, dropout=0.):
        conv_block = []
        conv_block += [nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size, padding=kernel_size // 2, padding_mode=padding_mode),
                       norm_layer(in_channel),
                       activation()]
        if dropout > 0.:
            conv_block += [nn.Dropout(dropout)]
        conv_block += [nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=kernel_size // 2, padding_mode=padding_mode, stride=stride),
                       norm_layer(out_channel)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        scale_x = x
        if self.downscale is not None:
            scale_x = self.downscale(scale_x)
        out = scale_x + self.res_block(x)
        return out


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super(AdaptiveInstanceNorm, self).__init__()
        self.norm = nn.InstanceNorm2d(in_channel)
        self.style = nn.Linear(style_dim, in_channel * 2)
        # 0-mean, 1 std
        self.style.weight.data[:in_channel] = 1
        self.style.weight.data[in_channel:] = 0

    def forward(self, input, light_vec):
        style = self.style(light_vec)
        gamma, beta = style.unsqueeze(2).unsqueeze(3).chunk(2, 1)
        out = self.norm(input)
        out = gamma * out + beta
        return out

class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc, kernel_size=3, padding_mode='reflect'):
        super().__init__()

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = kernel_size // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=kernel_size, padding=pw, padding_mode=padding_mode),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=kernel_size, padding=pw, padding_mode=padding_mode)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=kernel_size, padding=pw, padding_mode=padding_mode)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out

class NoiseInjection(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.weight = nn.Parameter(th.zeros(1, channel, 1, 1))

    def forward(self, image, noise):
        return image + self.weight * noise

class ConstantInput(nn.Module):
    def __init__(self, channel, size=(8, 8)):
        super().__init__()

        self.input = nn.Parameter(th.randn(1, channel, *size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)
        return out

class StyledConvBlock(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size=3,
            padding=1,
            style_dim=512,
            latent_size=(8, 8),
            initial=False,
            padding_mode='reflect',
    ):
        super().__init__()

        if initial:
            self.conv1 = ConstantInput(in_channel, latent_size)
        else:
            self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, padding_mode=padding_mode )

        self.noise1 = NoiseInjection(out_channel)
        self.adain1 = AdaptiveInstanceNorm(out_channel, style_dim)
        self.lrelu1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, padding_mode=padding_mode )
        self.noise2 = NoiseInjection(out_channel)
        self.adain2 = AdaptiveInstanceNorm(out_channel, style_dim)
        self.lrelu2 = nn.LeakyReLU()

    def forward(self, input, style, noise):
        out = self.conv1(input)
        out = self.noise1(out, noise)
        out = self.lrelu1(out)
        out = self.adain1(out, style)

        out = self.conv2(out)
        out = self.noise2(out, noise)
        out = self.lrelu2(out)
        out = self.adain2(out, style)

        return out


class Generator(nn.Module):
    def __init__(self, n_block, n_channel, out_channel, latent_size, style_dim, padding_mode='reflect'):
        super().__init__()
        block = [StyledConvBlock(n_channel, n_channel, 3, 1, style_dim=style_dim, latent_size=latent_size, initial=True)]
        for i in range(n_block - 2):
            block += [StyledConvBlock(n_channel, n_channel, 3, 1, style_dim=style_dim)]
        self.conv = nn.Sequential(nn.Conv2d(n_channel, out_channel, 3, 1, 1, padding_mode=padding_mode),
                                  nn.InstanceNorm2d(out_channel),
                                  nn.LeakyReLU())
        self.progression = nn.ModuleList(block)

    def forward(self, style, noise):
        out = noise[0]
        for i, conv in enumerate(self.progression):
            out = conv(out, style, noise[i])
        out = self.conv(out)
        return out
