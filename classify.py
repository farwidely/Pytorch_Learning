import torch

outputs = torch.tensor([[0.1, 0.2],
                        [0.05, 0.4]])

# argmax()可计算出tensor中各行各列的最大值，以outputs为例，参数0表示求列最大值，参数1表示求行最大值
print(outputs.argmax(0))
print(outputs.argmax(1))

preds = outputs.argmax(1)
targets = torch.tensor([0, 1])
print((preds == targets).sum())