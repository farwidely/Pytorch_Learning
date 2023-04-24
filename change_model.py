import torch
import torchvision
from torch import nn

# 调用已训练好的vgg16，weights='DEFAULT'等价于weights='IMAGENET1K_V1'
vgg16_true = torchvision.models.vgg16(weights='DEFAULT')
# 调用未经过训练的vgg16
vgg16_false = torchvision.models.vgg16()

print(vgg16_true)
print(vgg16_false)

# vgg16增加一个线性层，将1000分类改成10分类
vgg16_false.add_module('add_linear', nn.Linear(1000, 10))
print(vgg16_false)
# 在vgg16中的classifier内部增加线性层
# vgg16_false.classifier.add_module('add_linear', nn.Linear(1000, 10))

# 修改vgg16中的classifier的最后一个线性层，将1000分类改成10分类
vgg16_true.classifier[6] = nn.Linear(4096, 10)
print(vgg16_true)

# 保存模型
# 保存方式1，模型结构+模型参数
torch.save(vgg16_false, "./trained_models/vgg16_method1.pth")
# 保存方式2，模型参数，保存为字典（官方推荐）
vgg16 = torchvision.models.vgg16()
torch.save(vgg16.state_dict(), "./trained_models/vgg16_method2.pth")

# 加载模型
# 加载方式1，对应保存方式1
model1 = torch.load("./trained_models/vgg16_method1.pth")
print(model1)
# 加载方式2，对应保存方式2
model2 = torchvision.models.vgg16()
model2.load_state_dict(torch.load("./trained_models/vgg16_method2.pth"))
print(model2)
