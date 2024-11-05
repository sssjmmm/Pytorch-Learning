import torch
import torchvision.datasets
from torch import nn
from torch.fx.experimental.fx_acc.acc_ops import reshape
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from nn_maxpool import dataset, dataloader, output

input = torch.tensor([[1, -0.5],
                      [-1, 3]])
input = torch.reshape(input, (-1, 1, 2, 2))
print(input.shape)

dataset = torchvision.datasets.CIFAR10("./data", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)

class SJM(nn.Module):
    def __init__(self):
        super(SJM, self).__init__()
        # self.relu1 = nn.ReLU()
        self.sigmoid1 = nn.Sigmoid()

    def forward(self, input):
        output = self.sigmoid1(input)
        return output

sjm = SJM()
# output = sjm(input)
# print(output)

writer = SummaryWriter("./logs_relu")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, global_step=step)
    output = sjm(imgs)
    writer.add_images("output", output, global_step=step)
    step += 1

writer.close()
