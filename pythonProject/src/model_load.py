import torch
import torchvision.models

# 方式1 -> 保存方式1对应的加载模型方式
model = torch.load("vgg16_method1.pth")
# print(model)

# 方式2, -> 保存方式2对应的加载模型方式
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))