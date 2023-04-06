import torch
import torchvision
from PIL import Image
from torch import nn
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# writer = SummaryWriter("./logs_imgs")

image_path = "./imgs/cat3.jpg"
image = Image.open(image_path)
# png格式是四个通道，除了RGB三通道外，还有一个透明度通道，调用 image = image.convert(RGB),保留其颜色通道
# 如果图片本来就是三个颜色通道，经过此操作，不变，且不同截图软件截图保留的通道数是不一样的
image = image.convert('RGB')
print(image)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])

image = transform(image)
print(image.shape)

# writer.add_images("image", image, 1, dataformats='CHW')
# writer.close()


class Cifar10(nn.Module):
    def __init__(self):
        super(Cifar10, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


model = torch.load("./trained_models/cifar10_gpu_30.pth")
model.to(device)
# print(model)
image = torch.reshape(image, (1, 3, 32, 32))

model.eval()
with torch.no_grad():
    image = image.to(device)
    output = model(image)
print(output)

print(output.argmax(1))
# 'airplane' = (int) 0
# 'automobile' = (int) 1
# 'bird' = (int) 2
# 'cat' = (int) 3
# 'deer' = (int) 4
# 'dog' = (int) 5
# 'frog' = (int) 6
# 'horse' = (int) 7
# 'ship' = (int) 8
# 'truck' = (int) 9
