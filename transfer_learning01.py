import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms


# 定义数据集
def getdata():
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # 获取数据
    train_data = torchvision.datasets.CIFAR10(root="./dataset", train=True, download=True, transform=transform)
    test_data = torchvision.datasets.CIFAR10(root="./dataset", train=False, download=True, transform=transform)
    train_length = len(train_data)
    test_length = len(test_data)

    # 加载数据
    train_data_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
    test_data_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True)

    return train_data_loader, test_data_loader, train_length, test_length

# 构建模型


# 加载模型
def model_load():
    model = torchvision.models.vgg16(pretrained=True)

    return model


# 实现冻层
def set_parameter_requires_grad(model):
    for parameter in model.parameters():
        parameter.require_grad = False


# 微调
def tuning_model(model):
    set_parameter_requires_grad(model)
    model.classifier[-1] = nn.Linear(4096, 10)

    return model


# 训练模型
def train_model(step):
    for data in train_dataloader:
        images, targets = data
        images = images.to(use_gpu())
        targets = targets.to(use_gpu())
        output = classifier(images)
        loss = loss_fn(output, targets)  # 损失

        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 梯度反向传播
        optimizer.step()  # 更新参数

        step += 1
        if step % 100 == 0:
            print("训练次数：{}，Loss：{}".format(step, loss.item()))
            writer.add_scalars("Loss", {"train_loss": loss.item()}, step)

    return step


# 测试模型
def test_model(step, correct):
    with torch.no_grad():
        for data in test_dataloader:
            images, targets = data
            images = images.to(use_gpu())
            targets = targets.to(use_gpu())
            output = classifier(images)
            loss = loss_fn(output, targets)
            correct += (output.argmax(1) == targets).sum().item()

            step += 1
            if step % 100 == 0:
                print("测试次数：{}，Loss：{}".format(step, loss.item()))
                writer.add_scalars("Loss", {"test_loss": loss.item()}, step)
    return step, correct


# 使用GPU或CPU
def use_gpu():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


# 基本参数
# 模型
classifier = model_load()
classifier = tuning_model(classifier)
classifier = classifier.to(use_gpu())
# 数据
train_dataloader, test_dataloader, train_data_length, test_data_length = getdata()
print(train_data_length)
print(test_data_length)
# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(use_gpu())
# 优化器
learning_rate = 1e-2
optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
# 使用tensorboard进行可视化
writer = SummaryWriter("./transfer_learning01")

if __name__ == '__main__':
    epoch = 20
    train_step = 0
    test_step = 0

    # 训练模型
    for i in range(epoch):
        print("--------第{}轮训练开始--------".format(i+1))
        train_step = train_model(train_step)
        print("--------第{}轮训练结束--------".format(i+1))

    # 测试模型
    for i in range(epoch):
        print("--------第{}轮测试开始--------".format(i + 1))
        acc_list = []
        test_correct = 0
        test_step, test_correct = test_model(test_step, test_correct)
        accuracy = test_correct / test_data_length
        print("第{}轮的测试正确率：{}".format(i+1, accuracy))
        print("--------第{}轮测试结束--------".format(i + 1))
        acc_list.append(accuracy)
        writer.add_scalar("accuracy", accuracy, i)

writer.close()
