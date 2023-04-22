import torchvision
from torch import nn, optim

model = torchvision.models.resnet18(pretrained=True)
# nn.Module有成员函数parameters()
for param in model.parameters():
    param.requires_grad = False
# Replace the last fully-connected layer
# Parameters of newly constructed modules have requires_grad=True by default
# resnet18中有self.fc，作为前向过程的最后一层。
model.fc = nn.Linear(512, 100)

# Optimize only the classifier
# optimizer用于更新网络参数，默认情况下更新所有的参数
optimizer = optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9)

# PS: copied from https://blog.csdn.net/VictoriaW/article/details/72779407
