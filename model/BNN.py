import torch
from torch import nn
import torch.nn.functional as F
from model.resnet import ResNet


class dim_reduce(nn.Module):
    def __init__(self, input_channel=3, out_dim=50):
        super(dim_reduce, self).__init__()
        self.input_channel = input_channel

        self.backbone = ResNet(50, input_channel=self.input_channel)
        # self.backbone = resnet50()
        # self.backbone = resnet101_2()

        self.conv1 = nn.Conv2d(in_channels=2048, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.AP = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(4 * 4 * 128, 128)
        self.dropout = nn.Dropout(p = 0.7),
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, out_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output = self.backbone(x)  # channel 256 512 1024 2048

        layer4 = output[3]

        out = self.relu(self.conv1(layer4))
        out = self.AP(out)
        out = torch.reshape(out, (-1, 4 * 4 * 128))
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

class BNN(nn.Module):
    def __init__(self, out_dim, alpha, input_channel1=3, input_channel2=3):
        super(BNN, self).__init__()

        self.model_1 = dim_reduce(input_channel1, out_dim)
        self.model_2 = dim_reduce(input_channel2, out_dim)

        # self.criterion = nn.MSELoss(reduce=False)
        self.BNN_loss = BNN_loss(alpha)
        self.parameters_1 = self.model_1.parameters()
        self.parameters_2 = self.model_2.parameters()

    def forward(self, x1, x2, label):
        out_1 = self.model_1(x1)
        out_2 = self.model_2(x2)
        distance = (out_1 - out_2) ** 2
        distance = torch.mean(distance, 1, keepdim=True)

        distance = torch.sqrt(distance)

        loss, pos_loss, neg_loss = self.BNN_loss(distance, label)
        return loss, pos_loss, neg_loss, distance


class BNN_loss(nn.Module):
    def __init__(self, alpha):
        super(BNN_loss, self).__init__()
        self.alpha = alpha

    def forward(self, distance, labels):
        labels = torch.reshape(labels, (-1, 1))
        # labels_ = torch.add(torch.neg(labels), 1)
        labels_ = labels * -1 + 1

        loss_ = (distance - labels_) ** 2

        pos_loss = torch.sum(loss_ * labels) / (torch.sum(labels) + 1e-05)
        neg_loss = torch.sum(loss_ * labels_) / (torch.sum(labels_) + 1e-05)

        loss = (pos_loss + self.alpha * neg_loss) / (1 + self.alpha)
        # return loss, torch.sum(labels), torch.sum(labels_)
        return loss, pos_loss, neg_loss


if __name__ == '__main__':
    from torchsummary import summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = dim_reduce().to(device)
    print(model)
    # summary(model, [(3, 256, 256)])
