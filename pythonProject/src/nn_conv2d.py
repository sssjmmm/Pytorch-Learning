import torch
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from nn_conv import output
from tenser_board import writer

dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=torchvision.transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset, batch_size=64)

class SJM(torch.nn.Module):
    def __init__(self):
        super(SJM, self).__init__()
        self.conv1 = nn.Conv2d( in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x

sjm = SJM()

writer = SummaryWriter("./logs")

step = 0
for data in dataloader:
    imgs, targets = data
    outputs = sjm(imgs)

    print(imgs.shape) # torch.Size([64, 3, 32, 32])
    print(outputs.shape) # torch.Size([64, 6, 30, 30])

    writer.add_images("input", imgs, step)
    # torch.Size([64, 6, 30, 30]) -> [xxx, 3, 30, 30]
    outputs = torch.reshape(outputs, (-1, 3, 30, 30))
    writer.add_images("output", outputs, step)

    step += 1