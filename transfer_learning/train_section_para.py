import torch
import torchvision
from torch import nn, optim

model = torchvision.models.vgg16(weights='IMAGENET1K_V1')
# 冻结除全连接层外的模型参数
# num = 0
for param in model.parameters():
    # num += 1
    param.requires_grad = False
    # if num == 26:
    #     break

# print(f"num={num}")

# 修改模型最后一层
model.classifier[6] = nn.Linear(4096, 100, bias=True)
# model.classifier.add_module('7', nn.ReLU(inplace=True))
# model.classifier.add_module('8', nn.Dropout(p=0.5, inplace=False))
# model.classifier.add_module('9', nn.Linear(1000, 100))
for param in model.parameters():
    if param.requires_grad:
        print(param)


# print(model.classifier.parameters())


# print(model)

# 设置优化器只更新最后一层参数
optimizer = optim.SGD(model.classifier[9].parameters(), lr=1e-2, momentum=0.9)

# if __name__ == '__main__':
#     input = torch.ones((64, 3, 224, 224))
#     output = model(input)
#     print(output.shape)
#     print(output)

print(model)
