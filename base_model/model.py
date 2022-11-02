import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, ReLU

# 搭建神经网络
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.model = nn.Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),    # 用来压缩
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            ReLU(),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


#   主方法
if __name__ == '__main__':
    classifier = Classifier()
    print(classifier)
    input = torch.ones((64, 3, 32, 32))
    output = classifier(input)
