import torch
import torch.nn as nn


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


if __name__ == '__main__':
    x = torch.randn(1, 1, 3, 4)
    dc = DepthwiseSeparableConv2d(1, 2, 3, padding=1)
    y = dc(x)
    print(x.shape)
    print(y.shape)
