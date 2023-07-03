import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MyDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        super(MyDataset, self).__init__()
        self.data = torch.from_numpy(images)
        self.labels = torch.from_numpy(labels)
        self.transform = transform

    def __getitem__(self, index):
        x = self.transform(self.data[index].float())
        y = self.labels[index].float()
        return x, y

    def __len__(self):
        return len(self.data)


dataset = np.load("./dataset.npz")
transform = transforms.Compose([transforms.Resize(size=112, antialias=True)])

train_dataset = MyDataset(dataset["X_train"], dataset["y_train"], transform=transform)
val_dataset = MyDataset(dataset["X_val"], dataset["y_val"], transform=transform)
test_dataset = MyDataset(dataset["X_test"], dataset["y_test"], transform=transform)

train_data_size = len(train_dataset)
val_data_size = len(val_dataset)
test_data_size = len(test_dataset)
print(f"训练数据集的长度为: {train_data_size}")
print(f"验证数据集的长度为: {val_data_size}")
print(f"测试数据集的长度为: {test_data_size}")

train_dataloader = DataLoader(train_dataset, batch_size=64)
val_dataloader = DataLoader(val_dataset, batch_size=64)
test_dataloader = DataLoader(test_dataset, batch_size=64)

model = torch.load("./trained_models/resnet18_gpu_30.pth")
model.to(device)

loss_fn = nn.BCEWithLogitsLoss()
loss_fn.to(device)

model.eval()

total_train_loss = 0
total_train_accuracy = 0
total_val_loss = 0
total_val_accuracy = 0
total_test_loss = 0
total_test_accuracy = 0
total_test_tp = np.array([0] * 38)
total_test_fp = np.array([0] * 38)
total_test_fn = np.array([0] * 38)

with torch.no_grad():
    for data in tqdm(train_dataloader):
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = model(imgs)
        loss = loss_fn(outputs, targets)
        total_train_loss += loss.item()
        outputs[outputs >= 0.5] = 1
        outputs[outputs < 0.5] = 0
        equal_matrix = torch.all(torch.eq(outputs, targets), dim=1)
        equal_count = torch.sum(equal_matrix)
        accuracy = equal_count.item()
        total_train_accuracy += accuracy
    print(f"整体训练集上的Loss: {total_train_loss}")
    print(f"整体训练集上的正确率: {total_train_accuracy / train_data_size}")

    for data in tqdm(val_dataloader):
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = model(imgs)
        loss = loss_fn(outputs, targets)
        total_val_loss += loss.item()
        outputs[outputs >= 0.5] = 1
        outputs[outputs < 0.5] = 0
        equal_matrix = torch.all(torch.eq(outputs, targets), dim=1)
        equal_count = torch.sum(equal_matrix)
        accuracy = equal_count.item()
        total_val_accuracy += accuracy
    print(f"整体验证集上的Loss: {total_val_loss}")
    print(f"整体测试集上的正确率: {total_val_accuracy / val_data_size}")

    for data in tqdm(test_dataloader):
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = model(imgs)
        loss = loss_fn(outputs, targets)
        total_test_loss += loss.item()
        outputs[outputs >= 0.5] = 1
        outputs[outputs < 0.5] = 0
        equal_matrix = torch.all(torch.eq(outputs, targets), dim=1)

        y_true = []
        y_pred = []


        # 计算测试集混淆矩阵
        # CM_test = confusion_matrix(y_pred, y_true, labels=label)
        # TP = np.diag(CM_test)
        # FP = CM_test.sum(axis=0) - np.diag(CM_test)
        # FN = CM_test.sum(axis=1) - np.diag(CM_test)
        # total_test_tp += TP
        # total_test_fp += FP
        # total_test_fn += FN

        equal_count = torch.sum(equal_matrix)
        accuracy = equal_count.item()
        total_test_accuracy += accuracy

    # 计算测试集查准率、查全率、F1指数
    test_Precision = total_test_tp / (total_test_tp + total_test_fp)
    test_Recall = total_test_tp / (total_test_tp + total_test_fn)
    test_f1 = 2 * test_Precision * test_Recall / (test_Precision + test_Recall)

    print(f"整体测试集上的Precision: {test_Precision}")
    print(f"整体测试集上的Recall: {test_Recall}")
    print(f"整体测试集上的F1-score: {test_f1}")
    print(f"整体测试集上的Loss: {total_test_loss}")
    print(f"整体测试集上的正确率: {total_test_accuracy / test_data_size}")
