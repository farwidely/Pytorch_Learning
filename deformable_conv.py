import torch
import torch.nn as nn
import torchvision


class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super(DeformableConv2d, self).__init__()

        self.offset_conv = nn.Conv2d(in_channels, kernel_size * kernel_size * 2 * in_channels, kernel_size=3, stride=stride,
                                     padding=padding, dilation=dilation, groups=in_channels)
        self.conv = torchvision.ops.DeformConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                                 padding=padding, dilation=dilation, groups=groups)

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = in_channels

    def forward(self, x):
        offset = self.offset_conv(x)
        offset = offset.view(offset.shape[0], self.kernel_size * self.kernel_size * 2 * self.deformable_groups, x.shape[2], x.shape[3])

        # 计算归一化后的偏移量
        offset = torch.softmax(offset, dim=1)

        # 进行可变形卷积操作
        x = self.conv(x, offset)

        return x


if __name__ == '__main__':
    x = torch.randn(64, 16, 32, 32)
    dc = DeformableConv2d(16, 16, 3, padding=1)
    y = dc(x)
    print(x.shape)
    print(y.shape)
