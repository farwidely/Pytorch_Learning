from torch.utils.data import Dataset
from PIL import Image
import os


# 路径连接
# root_dir = "dataset/train"
# label_dir = "ants"
# path = os.path.join(root_dir, label_dir)
# img_path = os.listdir(path)
# idx = 0
# img_name = img_path[idx]
# img_item_path = os.path.join(root_dir, label_dir, img_name)

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
