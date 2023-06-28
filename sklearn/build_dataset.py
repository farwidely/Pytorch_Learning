# 将numpy数据集划分、导入为tensor数据集
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

data = np.load("xxx.npz")

train = data["arr_0"]
label = data["arr_1"]

# 划分测试集
X_train, X_test, y_train, y_test = train_test_split(train, label, test_size=0.1, random_state=42)

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1111, random_state=42)

# 处理单通道图片数据
X_train = np.expand_dims(X_train, axis=1)
X_val = np.expand_dims(X_val, axis=1)
X_test = np.expand_dims(X_test, axis=1)

dataset = {'X_train': X_train, 'y_train': y_train,
           'X_val': X_val, 'y_val': y_val,
           'X_test': X_test, 'y_test': y_test}

np.savez('./dataset/dataset.npz', **dataset)


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


dataset = np.load("./dataset/dataset.npz")

transform = transforms.Compose([transforms.Resize(224)])

train_dataset = MyDataset(dataset["X_train"], dataset["y_train"], transform=transform)
val_dataset = MyDataset(dataset["X_val"], dataset["y_val"], transform=transform)
test_dataset = MyDataset(dataset["X_test"], dataset["y_test"], transform=transform)
