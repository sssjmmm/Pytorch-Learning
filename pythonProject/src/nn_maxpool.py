import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from nn_conv import output
from nn_conv2d import dataset, dataloader, writer

dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)

# input = torch.tensor([[1, 2, 0, 3, 1],
#                      [0, 1, 2, 3, 1],
#                      [1, 2, 1, 0, 0],
#                      [5, 2, 3, 1, 1],
#                      [2, 1, 0, 1, 1]], dtype=torch.float32)
#
# input = torch.reshape(input, (-1, 1, 5, 5))
# print(input.shape)

class SJM(nn.Module):
    def __init__(self):
        super(SJM, self).__init__()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, input):
        output = self.maxpool1(input)
        return output

sjm = SJM()
# output = sjm(input)
# print(output)

writer = SummaryWriter("log_maxpool")
step = 0

for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = sjm(imgs)
    writer.add_images("output", output, step)
    step += 1

writer.close()