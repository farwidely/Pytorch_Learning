import torch
from thop import profile
from torchsummary import summary


model = torch.load("./xxx.pth")
model.cpu()

input = torch.randn(1, 3, 224, 224)
input.to('cpu')

flops, params = profile(model, inputs=(input, ))
print('FLOPs = ' + str(flops/1000**3) + 'G')
print('Params = ' + str(params/1000**2) + 'M')

model.cuda()
summary(model, input_size=(3, 224, 224))
