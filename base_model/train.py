import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="./dataset01", train=True, download=True, transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10(root="./dataset01", train=False, download=True, transform=torchvision.transforms.ToTensor())

# 查看有多少图片，获取长度
train_data_size = len(train_data)
test_data_size = len(test_data)
# print(train_data_size)
# print(test_data_size)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 利用DataLoad进行加载数据
train_data_load = DataLoader(train_data, batch_size=64)
test_data_load = DataLoader(test_data, batch_size=64)

# 创建网络模型
classifier = Classifier()

# 损失函数，分类问题，可使用交叉熵损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器，使用随机梯度下降
# learning_rate = 0.01
# e代表10
learning_rate = 1e-2
optimizer = torch.optim.SGD(classifier.parameters(), lr=learning_rate)

# 设置训练网络的参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练轮数，即迭代次数
epoch = 20

writer = SummaryWriter("./logs_train_optimizer06")

for i in range(epoch):
    print("-------第{}轮训练开始-------".format(i+1))

    # 训练步骤开始
    classifier.train()   # 当模型中存在一些特殊层时，需要调用
    for data in train_data_load:
        imgs, targets = data
        output = classifier(imgs)
        loss = loss_fn(output, targets)

        # 优化器优化模型
        optimizer.zero_grad()  # 梯度清零
        loss.backward()   # 反向传播计算梯度
        optimizer.step()    # 更新参数

        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数：{}，Loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("loss/train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    classifier.eval()  # 当模型中存在一些特殊层时，需要调用
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():  # 只用来测试，不需要梯度，以及优化
        for data in test_data_load:
            imgs, targets = data
            outputs = classifier(imgs)
            loss = loss_fn(outputs, targets)   # 单个损失
            total_test_loss = total_test_loss + loss.item()   # 整体损失
            accuracy = (outputs.argmax(1) == targets).sum().item()
            total_accuracy = total_accuracy + accuracy
    accuracy_rate = total_accuracy/test_data_size
    print("整体测试集上的Loss：{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(accuracy_rate))
    writer.add_scalar("loss/test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", accuracy_rate, total_test_step)
    total_test_step += 1

    # torch.save(classifier, "./classifier_{}.pth".format(i+1))
    # print("模型以保存")


writer.close()












