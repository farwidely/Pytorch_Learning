import time
import torchvision
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from models import *

# 设置计算硬件为cpu或cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda")
# device = torch.device("cpu")

# 准备数据集
# 训练集
train_data = torchvision.datasets.CIFAR10(root="./data", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
# 测试集
test_data = torchvision.datasets.CIFAR10(root="./data", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

# 查看数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"训练数据集的长度为: {train_data_size}")
print(f"测试数据集的长度为: {test_data_size}")

# 利用DataLoader加载数据
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 搭建神经网络见models.py

# 创建网络模型
cifar10 = Cifar10()
cifar10 = torch.compile(cifar10)
cifar10.to(device)
# if torch.cuda.is_available():
#     cifar10 = cifar10.cuda()

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)
# if torch.cuda.is_available():
#     loss_fn = loss_fn.cuda()

# 优化器
# learning_rate = 0.01
learning_rate = 1e-2
optimizer = torch.optim.SGD(cifar10.parameters(), lr=learning_rate)

# 设置训练网络的参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 30

# 添加tensorboard
# writer = SummaryWriter("./logs_train")

start1 = time.time()

for i in range(epoch):
    print(f"------第 {i+1} 轮训练开始------")

    start = time.time()
    # 训练步骤开始
    cifar10.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        # if torch.cuda.is_available():
        #     imgs = imgs.cuda()
        #     targets = targets.cuda()
        outputs = cifar10(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print(f"训练次数: {total_train_step}，Loss: {loss.item()}")
            # writer.add_scalar("train_loss", loss.item(), total_train_step)
    end = time.time()
    train_time_spent = end - start
    print(f"第{i}轮训练耗时{train_time_spent}s")

    start = time.time()

    # 测试步骤开始
    cifar10.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            # if torch.cuda.is_available():
            #     imgs = imgs.cuda()
            #     targets = targets.cuda()
            outputs = cifar10(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    end = time.time()
    test_time_spent = end - start
    print(f"第{i}轮测试耗时{test_time_spent}s")

    print(f"整体测试集上的Loss: {total_test_loss}")
    print(f"整体测试集上的正确率: {total_accuracy/test_data_size}")
    # writer.add_scalar("test_loss", total_test_loss, total_test_step)
    # writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step += 1

    # torch.save(cifar10, f"./trained_models/cifar10_{i}.pth")
    # torch.save(cifar10.state_dict(), f"./trained_models/cifar10_{i}.pth")
    if i == 29:
        # torch.save(cifar10, f"./trained_models/cifar10_gpu_30.pth")
        end = time.time()
        time_spent = end - start1
        print(f"总耗时{time_spent}s")

# writer.close()