from abc import ABC, abstractmethod

import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm


class BaseNetwork(nn.Module, ABC):
    def __init__(self, input_size, features_size):
        super().__init__()
        self.in_size = input_size
        self.f_size = features_size
        self.network = self._build_network()
        self._init_weights(self.network)

    def forward(self, x):
        return self.network(x)

    @abstractmethod
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        pass

    @abstractmethod
    def _build_network(self):
        pass

    def _init_weights(self, model):
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)


class Discriminator(BaseNetwork):
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def _build_network(self):
        return nn.Sequential(
            nn.Conv2d(self.in_size, self.f_size, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            self._block(self.f_size, self.f_size * 2, 4, 2, 1),
            self._block(self.f_size * 2, self.f_size * 4, 4, 2, 1),
            self._block(self.f_size * 4, self.f_size * 8, 4, 2, 1),
            nn.Conv2d(self.f_size * 8, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid(),
        )


class Generator(BaseNetwork):
    def __init__(self, noise_size, input_size, features_size):
        self.n_size = noise_size
        super().__init__(input_size, features_size)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def _build_network(self):
        return nn.Sequential(
            self._block(self.n_size, self.f_size * 8, 4, 1, 0),
            self._block(self.f_size * 8, self.f_size * 4, 4, 2, 1),
            self._block(self.f_size * 4, self.f_size * 2, 4, 2, 1),
            self._block(self.f_size * 2, self.f_size, 4, 2, 1),
            nn.ConvTranspose2d(
                self.f_size, self.in_size, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(),
        )


class SpectralDiscriminator(BaseNetwork):
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            spectral_norm(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    bias=False,
                )
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def _build_network(self):
        return nn.Sequential(
            spectral_norm(
                nn.Conv2d(self.in_size, self.f_size, kernel_size=4, stride=2, padding=1)
            ),
            nn.LeakyReLU(0.2, inplace=True),
            self._block(self.f_size, self.f_size * 2, 4, 2, 1),
            self._block(self.f_size * 2, self.f_size * 4, 4, 2, 1),
            self._block(self.f_size * 4, self.f_size * 8, 4, 2, 1),
            spectral_norm(
                nn.Conv2d(self.f_size * 8, 1, kernel_size=4, stride=2, padding=0)
            ),
            nn.Sigmoid(),
        )


class SpectralGenerator(Generator):
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            spectral_norm(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    bias=False,
                )
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def _build_network(self):
        return nn.Sequential(
            self._block(self.n_size, self.f_size * 8, 4, 1, 0),
            self._block(self.f_size * 8, self.f_size * 4, 4, 2, 1),
            self._block(self.f_size * 4, self.f_size * 2, 4, 2, 1),
            self._block(self.f_size * 2, self.f_size, 4, 2, 1),
            spectral_norm(
                nn.ConvTranspose2d(
                    self.f_size, self.in_size, kernel_size=4, stride=2, padding=1
                )
            ),
            nn.Tanh(),
        )


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, 1, bias=False)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, 1, bias=False)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = (
            self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        )
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = torch.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)

        out = self.gamma * out + x
        return out


class SaDiscriminator(SpectralDiscriminator):
    def _build_network(self):
        return nn.Sequential(
            spectral_norm(
                nn.Conv2d(self.in_size, self.f_size, kernel_size=4, stride=2, padding=1)
            ),
            nn.LeakyReLU(0.2, inplace=True),
            self._block(self.f_size, self.f_size * 2, 4, 2, 1),
            SelfAttention(self.f_size * 2),
            self._block(self.f_size * 2, self.f_size * 4, 4, 2, 1),
            self._block(self.f_size * 4, self.f_size * 8, 4, 2, 1),
            spectral_norm(
                nn.Conv2d(self.f_size * 8, 1, kernel_size=4, stride=2, padding=0)
            ),
            nn.Sigmoid(),
        )


class SaGenerator(SpectralGenerator):
    def _build_network(self):
        return nn.Sequential(
            self._block(self.n_size, self.f_size * 8, 4, 1, 0),
            self._block(self.f_size * 8, self.f_size * 4, 4, 2, 1),
            SelfAttention(self.f_size * 4),
            self._block(self.f_size * 4, self.f_size * 2, 4, 2, 1),
            self._block(self.f_size * 2, self.f_size, 4, 2, 1),
            spectral_norm(
                nn.ConvTranspose2d(
                    self.f_size, self.in_size, kernel_size=4, stride=2, padding=1
                )
            ),
            nn.Tanh(),
        )
