import torch
from torch import nn


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
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
