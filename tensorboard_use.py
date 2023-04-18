import torchvision
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import numpy as np

writer = SummaryWriter("logs")

image_path = "dataset/train/ants/0013035.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)
print(type(img_array))
print(img_array.shape)

writer.add_image("test", img_array, 1, dataformats='HWC')

model = torchvision.models.vgg16()
writer.add_graph(model)

# y = x
for i in range(100):
    writer.add_scalar("y=x", i, i)

writer.close()

# tensorboard --logdir=logs

# 查看模型框架
# writer.add_graph(model, input)
