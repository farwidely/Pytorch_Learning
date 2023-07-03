import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

class2label = {
    'C1': [0, 0, 0, 0, 0, 0, 0, 0],
    'C2': [1, 0, 0, 0, 0, 0, 0, 0],
    'C3': [0, 1, 0, 0, 0, 0, 0, 0],
    'C4': [0, 0, 1, 0, 0, 0, 0, 0],
    'C5': [0, 0, 0, 1, 0, 0, 0, 0],
    'C6': [0, 0, 0, 0, 1, 0, 0, 0],
    'C7': [0, 0, 0, 0, 0, 1, 0, 0],
    'C8': [0, 0, 0, 0, 0, 0, 1, 0],
    'C9': [0, 0, 0, 0, 0, 0, 0, 1],
    'C10': [1, 0, 1, 0, 0, 0, 0, 0],
    'C11': [1, 0, 0, 1, 0, 0, 0, 0],
    'C12': [1, 0, 0, 0, 1, 0, 0, 0],
    'C13': [1, 0, 0, 0, 0, 0, 1, 0],
    'C14': [0, 1, 1, 0, 0, 0, 0, 0],
    'C15': [0, 1, 0, 1, 0, 0, 0, 0],
    'C16': [0, 1, 0, 0, 1, 0, 0, 0],
    'C17': [0, 1, 0, 0, 0, 0, 1, 0],
    'C18': [0, 0, 1, 0, 1, 0, 0, 0],
    'C19': [0, 0, 1, 0, 0, 0, 1, 0],
    'C20': [0, 0, 0, 1, 1, 0, 0, 0],
    'C21': [0, 0, 0, 1, 0, 0, 1, 0],
    'C22': [0, 0, 0, 0, 1, 0, 1, 0],
    'C23': [1, 0, 1, 0, 1, 0, 0, 0],
    'C24': [1, 0, 1, 0, 0, 0, 1, 0],
    'C25': [1, 0, 0, 1, 1, 0, 0, 0],
    'C26': [1, 0, 0, 1, 0, 0, 1, 0],
    'C27': [1, 0, 0, 0, 1, 0, 1, 0],
    'C28': [0, 1, 1, 0, 1, 0, 0, 0],
    'C29': [0, 1, 1, 0, 0, 0, 1, 0],
    'C30': [0, 1, 0, 1, 1, 0, 0, 0],
    'C31': [0, 1, 0, 1, 0, 0, 1, 0],
    'C32': [0, 1, 0, 0, 1, 0, 1, 0],
    'C33': [0, 0, 1, 0, 1, 0, 1, 0],
    'C34': [0, 0, 0, 1, 1, 0, 1, 0],
    'C35': [1, 0, 1, 0, 1, 0, 1, 0],
    'C36': [1, 0, 0, 1, 1, 0, 1, 0],
    'C37': [0, 1, 1, 0, 1, 0, 1, 0],
    'C38': [0, 1, 0, 1, 1, 0, 1, 0],
}

label2class = {
    (0, 0, 0, 0, 0, 0, 0, 0): 'C1',
    (1, 0, 0, 0, 0, 0, 0, 0): 'C2',
    (0, 1, 0, 0, 0, 0, 0, 0): 'C3',
    (0, 0, 1, 0, 0, 0, 0, 0): 'C4',
    (0, 0, 0, 1, 0, 0, 0, 0): 'C5',
    (0, 0, 0, 0, 1, 0, 0, 0): 'C6',
    (0, 0, 0, 0, 0, 1, 0, 0): 'C7',
    (0, 0, 0, 0, 0, 0, 1, 0): 'C8',
    (0, 0, 0, 0, 0, 0, 0, 1): 'C9',
    (1, 0, 1, 0, 0, 0, 0, 0): 'C10',
    (1, 0, 0, 1, 0, 0, 0, 0): 'C11',
    (1, 0, 0, 0, 1, 0, 0, 0): 'C12',
    (1, 0, 0, 0, 0, 0, 1, 0): 'C13',
    (0, 1, 1, 0, 0, 0, 0, 0): 'C14',
    (0, 1, 0, 1, 0, 0, 0, 0): 'C15',
    (0, 1, 0, 0, 1, 0, 0, 0): 'C16',
    (0, 1, 0, 0, 0, 0, 1, 0): 'C17',
    (0, 0, 1, 0, 1, 0, 0, 0): 'C18',
    (0, 0, 1, 0, 0, 0, 1, 0): 'C19',
    (0, 0, 0, 1, 1, 0, 0, 0): 'C20',
    (0, 0, 0, 1, 0, 0, 1, 0): 'C21',
    (0, 0, 0, 0, 1, 0, 1, 0): 'C22',
    (1, 0, 1, 0, 1, 0, 0, 0): 'C23',
    (1, 0, 1, 0, 0, 0, 1, 0): 'C24',
    (1, 0, 0, 1, 1, 0, 0, 0): 'C25',
    (1, 0, 0, 1, 0, 0, 1, 0): 'C26',
    (1, 0, 0, 0, 1, 0, 1, 0): 'C27',
    (0, 1, 1, 0, 1, 0, 0, 0): 'C28',
    (0, 1, 1, 0, 0, 0, 1, 0): 'C29',
    (0, 1, 0, 1, 1, 0, 0, 0): 'C30',
    (0, 1, 0, 1, 0, 0, 1, 0): 'C31',
    (0, 1, 0, 0, 1, 0, 1, 0): 'C32',
    (0, 0, 1, 0, 1, 0, 1, 0): 'C33',
    (0, 0, 0, 1, 1, 0, 1, 0): 'C34',
    (1, 0, 1, 0, 1, 0, 1, 0): 'C35',
    (1, 0, 0, 1, 1, 0, 1, 0): 'C36',
    (0, 1, 1, 0, 1, 0, 1, 0): 'C37',
    (0, 1, 0, 1, 1, 0, 1, 0): 'C38'}

label = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10',
         'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20',
         'C21', 'C22', 'C23', 'C24', 'C25', 'C26', 'C27', 'C28', 'C29', 'C30',
         'C31', 'C32', 'C33', 'C34', 'C35', 'C36', 'C37', 'C38']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WM38Dataset(Dataset):
    def __init__(self, images, labels, transform=None):
        super(WM38Dataset, self).__init__()
        self.data = torch.from_numpy(images)
        self.labels = torch.from_numpy(labels)
        self.transform = transform

    def __getitem__(self, index):
        x = self.transform(self.data[index].float())
        y = self.labels[index].float()
        return x, y

    def __len__(self):
        return len(self.data)


dataset = np.load("./data_processing/dataset/dataset.npz")
transform = transforms.Compose([transforms.Resize(size=112, antialias=True)])

train_dataset = WM38Dataset(dataset["X_train"], dataset["y_train"], transform=transform)
val_dataset = WM38Dataset(dataset["X_val"], dataset["y_val"], transform=transform)
test_dataset = WM38Dataset(dataset["X_test"], dataset["y_test"], transform=transform)

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

num = np.array(
    [111, 117, 93, 92, 126, 103, 16, 100, 87, 94, 87, 94, 90, 102, 95, 95, 97, 88, 99, 109, 108, 86, 103, 231, 101,
     102, 86, 96, 85, 103, 102, 90, 108, 95, 101, 90, 98, 122])
correct_num = np.array([0] * 38)

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

        for i in range(len(outputs)):
            if equal_matrix[i]:
                for j in range(len(class2label)):
                    if torch.all(torch.eq(outputs[i].to('cpu'), torch.Tensor(class2label[f'C{j + 1}']))):
                        correct_num[j] += 1
                        break
            y_true.append(label2class[tuple(targets[i].to('cpu').numpy().astype(int))])
            y_pred.append(label2class[tuple(outputs[i].to('cpu').numpy().astype(int))])

        # 计算测试集混淆矩阵
        CM_test = confusion_matrix(y_pred, y_true, labels=label)
        TP = np.diag(CM_test)
        FP = CM_test.sum(axis=0) - np.diag(CM_test)
        FN = CM_test.sum(axis=1) - np.diag(CM_test)
        total_test_tp += TP
        total_test_fp += FP
        total_test_fn += FN

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
    print(f"整体测试集上各类别正确率：{correct_num / num}")
