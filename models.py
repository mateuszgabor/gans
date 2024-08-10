from abc import ABC, abstractmethod

from torch import nn


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
