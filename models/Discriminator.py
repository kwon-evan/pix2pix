import torch
import torch.nn as nn


class Dis_block(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True):
        super().__init__()

        layers = [nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.block(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.stage_1 = Dis_block(in_channels * 2, 64, normalize=False)
        self.stage_2 = Dis_block(64, 128)
        self.stage_3 = Dis_block(128, 256)
        self.stage_4 = Dis_block(256, 512)

        self.patch = nn.Conv2d(512, 1, 3, padding=1)  # 16x16 패치 생성

    def forward(self, a, b):
        x = torch.cat((a, b), 1)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.stage_4(x)
        x = self.patch(x)
        x = torch.sigmoid(x)
        return x


if __name__ == '__main__':
    from torchsummary import summary

    a = torch.randn(16, 3, 256, 256, device='cpu')
    b = torch.randn(16, 3, 256, 256, device='cpu')
    summary(Discriminator(), [(3, 256, 256), (3, 256, 256)], device='cpu')
