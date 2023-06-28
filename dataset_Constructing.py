from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision
from torchvision import transforms


# 路径连接
# root_dir = "dataset/train"
# label_dir = "ants"
# path = os.path.join(root_dir, label_dir)
# img_path = os.listdir(path)
# idx = 0
# img_name = img_path[idx]
# img_item_path = os.path.join(root_dir, label_dir, img_name)

# 自定义数据集
class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):

        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)


root_dir = "dataset/train"
ants_label_dir = "ants"
bees_label_dir = "bees"
ants_dataset = MyData(root_dir, ants_label_dir)
bees_dataset = MyData(root_dir, bees_label_dir)

img, label = ants_dataset[0]
print(img)
print(label)

train_dataset = ants_dataset + bees_dataset
print(len(train_dataset))
print(len(ants_dataset))
print(len(bees_dataset))
img1, label1 = train_dataset[0]
img2, label2 = train_dataset[123]
img3, label3 = train_dataset[124]
print(img1)
print(label1)
print(img2)
print(label2)
print(img3)
print(label3)

# 导入单张图片
# image_path = "C:\\Users\\13963\\PycharmProjects\\Pytorch_Learning\\dataset\\train\\ants\\0013035.jpg"
# # image_path = r"C:\Users\13963\PycharmProjects\Pytorch_Learning\dataset\train\ants\0013035.jpg"
# img = Image.open(image_path)
# img.show()

# 导入图片文件夹
# dir_path = "dataset/train/ants"
# img_path_list = os.listdir(dir_path)

# 数据集取子集
# from torch.utils.data import Subset
# dataset2 = Subset(dataset1, indices=range(0,10))

# # 准备数据集, 对数据集进行归一化
"""
transforms.RandomResizedCrop(size)    
将给定图像随机裁剪为不同的大小和宽高比，然后缩放所裁剪得到的图像为制定的大小；（即先随机采集，然后对裁剪得到的图像缩放为同一大小）
该操作的含义在于：即使只是该物体的一部分，我们也认为这是该类物体

transforms.RandomVerticalFlip(p)
以给定的概率随机垂直翻转给定的图像（PIL Image or Tensor）如果图像是torch张量，则期望它具有[C，H, W]形状，其中C表示任意数量的张量维度

transforms.RandomHorizontalFlip(p)
以给定的概率随机水平旋转给定的PIL的图像，默认为0.5

transforms.Resize(size=224, antialias=True)
将图片短边缩短至size (int)，长宽比保持不变，
这个警告信息是关于torchvision中的图像变换函数的参数antialias的改变，从v0.17版本开始，Resize()、RandomResizedCrop()
等所有调整图像大小的变换函数的默认antialias参数值将从None更改为True，以保持在PIL和Tensor后端之间的一致性。
这个警告提醒用户在升级到v0.17版本后需要注意这个改变。
如果想要避免这个警告，可以按照警告中的建议，直接将antialias参数设置为True或者False。
如果没有特殊要求，建议将antialias参数设置为True，以获得更好的图像质量。

transforms.Resize([h, w]
同时制定长宽

transforms.CenterCrop(size)
从图像中心裁剪图片

"""
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
# # 训练集
# train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
# # 测试集
# test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)
