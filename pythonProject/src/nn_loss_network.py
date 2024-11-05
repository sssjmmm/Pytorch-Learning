import torchvision.datasets
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader

from nn_loss import result

dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=torchvision.transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset, batch_size=1)

class SJM(nn.Module):
    def __init__(self):
        super(SJM, self).__init__()
        self.model1 = nn.Sequential(
            Conv2d(3, 32, kernel_size=5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, kernel_size=5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, kernel_size=5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x

loss = nn.CrossEntropyLoss()
sjm = SJM()

for data in dataloader:
    imgs, targets = data
    outputs = sjm(imgs)
    result_loss = loss(outputs, targets)
    # print(result_loss)
    result_loss.backward()
    print("ok")
    # print(outputs)
    # print(targets)
