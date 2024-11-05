import torchvision.transforms
from PIL import Image
from model import *

image_path = "./test_data/dog.png"
image = Image.open(image_path)
image = image.convert('RGB')
print(image)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])

image = transform(image)
print(image.shape)

model = torch.load("sjm_9.pth")
# 如果是在GPU上训练的模型需要再CPU下测试，那么就用下面这句
# model = torch.load("sjm_9.pth", map_location=torch.device('cpu'))
print(model)
image = torch.reshape(image, (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    output = model(image)
print(output)

print(output.argmax(dim=1))