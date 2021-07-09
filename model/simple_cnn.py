import logging
import torch
from model.weight_init import *


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        layer1 = nn.Sequential()
        layer1.add_module('conv1_1', nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1))
        layer1.add_module('relu1_1', nn.ReLU(True))
        layer1.add_module('bn1_1', nn.BatchNorm2d(32))
        layer1.add_module('conv1_2', nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1))
        layer1.add_module('relu1_2', nn.ReLU(True))
        layer1.add_module('bn1_2', nn.BatchNorm2d(32))
        layer1.add_module('pool1', nn.MaxPool2d(2, 2))
        self.layer1 = layer1


        layer2 = nn.Sequential()
        layer2.add_module('conv2_1', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1))
        layer2.add_module('relu2_1', nn.ReLU(True))
        layer2.add_module('bn2_1', nn.BatchNorm2d(64))
        layer2.add_module('conv2_2', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))
        layer2.add_module('relu2_2', nn.ReLU(True))
        layer2.add_module('bn2_2', nn.BatchNorm2d(64))
        layer2.add_module('pool2', nn.MaxPool2d(2, 2))
        self.layer2 = layer2

        layer3 = nn.Sequential()
        layer3.add_module('conv3_1', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1))
        layer3.add_module('relu3_1', nn.ReLU(True))
        layer3.add_module('bn3_1', nn.BatchNorm2d(128))
        layer3.add_module('conv3_2', nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1))
        layer3.add_module('relu3_2', nn.ReLU(True))
        layer3.add_module('bn3_2', nn.BatchNorm2d(128))
        layer3.add_module('poo13', nn.MaxPool2d(2, 2))
        self.layer3 = layer3

        layer4 = nn.Sequential()
        layer4.add_module('conv4_1', nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1))
        layer4.add_module('relu4_1', nn.ReLU(True))
        layer4.add_module('bn4_1', nn.BatchNorm2d(128))
        layer4.add_module('conv4_2', nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1))
        layer4.add_module('relu4_2', nn.ReLU(True))
        layer4.add_module('bn4_2', nn.BatchNorm2d(128))
        self.layer4 = layer4

    def forward(self, x):
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        return layer4



if __name__ == '__main__':
    from torchsummary import summary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleCNN().to(device)


    summary(model, (3, 112, 112))
