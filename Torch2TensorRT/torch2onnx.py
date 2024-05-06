import torch

model = torch.load("trained_models/MyCIFAR10_gpu_30.pth")
model.eval()

# 创建一个虚拟输入张量
x = torch.randn(1, 3, 224, 224).to('cuda')

# 使用torch.onnx.export导出ONNX模型
torch.onnx.export(model, x, "MyModel.onnx", verbose=True)

# trtexec.exe --onnx=model.onnx --saveEngine=model.trt
# trtexec.exe --onnx=model.onnx --saveEngine=model.trt --fp16
