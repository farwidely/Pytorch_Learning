# 本文件用于模型开发时的小模块测试
import torch
import torch.nn as nn
import torchvision


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                                        stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                        stride=1, padding=0, groups=1, bias=True)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class ShuffleChannel(nn.Module):
    def __init__(self, groups):
        super(ShuffleChannel, self).__init__()
        self.groups = groups

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        channels_per_group = channels // self.groups

        # 将张量按通道数分组
        x = x.view(batch_size, self.groups, channels_per_group, height, width)

        # 交换组内通道顺序
        x = torch.transpose(x, 1, 2).contiguous()

        # 展开张量
        x = x.view(batch_size, -1, height, width)

        return x


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0, bias=True, groups=4),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            ShuffleChannel(groups=4),
            DepthwiseSeparableConv2d(64, 64),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0, bias=True, groups=4),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    model = MyModel()
    input = torch.ones(64, 32, 32, 32)
    output = model(input)
    print(output.shape)
