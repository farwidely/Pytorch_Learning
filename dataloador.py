import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10("data", train=False, transform=torchvision.transforms.ToTensor())

# shuffle为是否打乱图片顺序，num_worker为工作线程，drop_last为是否抛弃剩余的不足一个batch_size的数据
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False, num_workers=0, drop_last=True )

# # 显示train_dataloader中的第一份数据
# print(next(iter(train_dataloader)))

img, target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter("dataloader")
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        # print(imgs.shape)
        # print(targets)
        writer.add_images(f"Epoch:{epoch}", imgs, step)
        step += 1

writer.close()