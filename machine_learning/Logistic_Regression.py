import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 这是因为anaconda的环境下存在多个libiomp5md.dll文件，如果不加，画图报错
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# 现在用 torch的 Tensor，用mini-batch的风格
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0.0], [0.0], [1.0]])


class LogisticRegressionModel(torch.nn.Module):  # 建立模型
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred


model = LogisticRegressionModel()  # 实例化模型
criterion = torch.nn.BCELoss(size_average=False)  # BCE损失
optimizer = torch.optim.ASGD(model.parameters(), lr=0.01)  # 优化器

for epoch in range(100):  # 训练循环
    y_pred = model(x_data)  # 计算y_hat
    loss = criterion(y_pred, y_data)  # 计算loss
    print(epoch, loss.item())

    optimizer.zero_grad()  # 清梯度
    loss.backward()  # 反向传播
    optimizer.step()

# output weight and bias
print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())

# 测试集
# test model
x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)

# 做测试 画图
x = np.linspace(0, 10, 200)  # 0-10h采200个点
x_t = torch.Tensor(x).view((200, 1))  # 将200个点编程200*1的矩阵 ；类似于numpy中的reshape
print(x_t)
y_t = model(x_t)  # 将得到的张量送到模型里
y = y_t.data.numpy()  # 将y数据拿出
plt.plot(x, y)  # 绘图
plt.plot([0, 10], [0.5, 0.5], c='r')
plt.xlabel('Hours')
plt.ylabel('Probability of Pass')
plt.grid()
plt.show()
