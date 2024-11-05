import torch
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64, drop_last=True)

class SJM2(nn.Module):
    def __init__(self):
        super(SJM2, self).__init__()
        self.linear1 = nn.Linear(196608, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output

sjm = SJM2()

for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    output = torch.flatten(imgs)
    print(output.shape)
    output = sjm(output)
    print(output.shape)