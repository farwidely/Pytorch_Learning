import torchvision
# from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import *
import time

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

# # 显示train_dataloader中的第一份数据
# print(next(iter(train_dataloader)))

# 搭建神经网络见models.py

# 创建网络模型
cifar10 = MyCIFAR10()
cifar10.to(device)
# if torch.cuda.is_available():
#     cifar10 = cifar10.cuda()

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)
# if torch.cuda.is_available():
#     loss_fn = loss_fn.cuda()

# 优化器
# 一般将参数momentum设为0.5,0.9，或者0.99，分别表示最大速度2倍，10倍，100倍于SGD的算法
# learning_rate = 0.01
learning_rate = 1e-2
momentum = 0.9
optimizer = torch.optim.SGD(cifar10.parameters(), lr=learning_rate, momentum=momentum)
# 学习率衰减
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
# optimizer = torch.optim.SGD(cifar10.parameters(), lr=learning_rate, weight_decay=1e-2)


# 设置训练网络的参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 30

# 添加tensorboard
# writer = SummaryWriter("./logs_train")

start = time.time()

# 初始化模型在测试集上的评价指标变量
# total_test_loss = 0
# total_test_accuracy = 0
# total_test_tp = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# total_test_fp = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# total_test_fn = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

for i in range(epoch):
    print(f"------第 {i + 1} 轮训练开始------")

    start1 = time.time()

    # 训练步骤开始
    cifar10.train()
    for data in tqdm(train_dataloader):
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        # if torch.cuda.is_available():
        #     imgs = imgs.cuda()
        #     targets = targets.cuda()
        outputs = cifar10(imgs)
        # print(outputs.shape)
        # print(targets.shape)
        # break
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print(f"训练次数: {total_train_step}，Loss: {loss.item()}")
            # writer.add_scalar("train_loss", loss.item(), total_train_step)

    end1 = time.time()
    print(f"本轮训练时长为{end1 - start1}秒")

    # 学习率衰减
    scheduler.step()

    start2 = time.time()

    # 测试步骤开始
    cifar10.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in tqdm(test_dataloader):
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

            # one-hot labels
            # equal_count = torch.sum(torch.all(torch.eq(outputs, targets), dim=1))
            # accuracy = equal_count.item()

            total_accuracy += accuracy

            # 计算测试集混淆矩阵
            # CM_test = confusion_matrix(outputs.argmax(1).to("cpu"), targets.to("cpu"), labels=[0, 1, 2, 3, 4, 5, 6, 7,
            #                                                                                     8, 9])
            # TP = np.diag(CM_test)
            # FP = CM_test.sum(axis=0) - np.diag(CM_test)
            # FN = CM_test.sum(axis=1) - np.diag(CM_test)
            # total_test_tp += TP
            # total_test_fp += FP
            # total_test_fn += FN

    # 计算测试集查准率、查全率、F1指数
    # test_Precision = total_test_tp / (total_test_tp + total_test_fp)
    # test_Recall = total_test_tp / (total_test_tp + total_test_fn)
    # test_f1 = 2 * test_Precision * test_Recall / (test_Precision + test_Recall)

    print(f"整体测试集上的Loss: {total_test_loss}")
    print(f"整体测试集上的正确率: {total_accuracy / test_data_size}")
    # writer.add_scalar("test_loss", total_test_loss, total_test_step)
    # writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)

    end2 = time.time()
    print(f"本轮测试时长为{end2 - start2}秒\n")

    total_test_step += 1

    # torch.save(cifar10, f"./trained_models/cifar10_{i}.pth")
    # torch.save(cifar10.state_dict(), f"./trained_models/cifar10_{i}.pth")
    if i == 29:
        torch.save(cifar10, f"./trained_models/MyCIFAR10_gpu_30.pth")
        print("模型已保存")

end = time.time()
print(f"训练+测试总时长为{end - start}秒")

# writer.close()
