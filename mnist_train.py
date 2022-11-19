# -*- coding: utf-8 -*-
# @时间: 2022/11/18 2022/11/18
# @作者： 流星
# @项目：**


import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

import torchvision
from matplotlib import pyplot as plt

from utils import plot_curve, plot_image, one_hot

# 设置一次处理图片的数量
batch_size = 512

# 第一步：加载数据集
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST("mnist_data", train=True, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.13017,), (0.3081,))
    ])),
    batch_size = batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST("mnist_data", train=False, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.13017,), (0.3081,))
    ])),
    batch_size = batch_size, shuffle=False)

# x, y = next(iter(train_loader))
# plot_image(x, y, "image sample")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # xw+b
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)
        self.fc4 = nn.Softmax()

    def forward(self, x):
        # x:[b, 1, 28, 28
        # h1 = relu(xw+b)
        x = F.relu(self.fc1(x))
        # h2 = relu(h1w2+b2)
        x = F.relu(self.fc2(x))
        # h3 = h2w3+b3
        x = F.softmax(self.fc3(x))
        return x

net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
train_loss = []
for epoch in range(100):
    for batch_idx, (x, y) in enumerate(train_loader):
        # x:[b, 1, 28, 28], y:[512]
        # [b, 1, 28, 28] => [b, 784]
        x = x.view(x.size(0), 28*28)
        # =>[b, 10]
        out = net(x)
        y_onehot = one_hot(y)
        # los = mse(out, y_onehot)
        loss = F.mse_loss(out, y_onehot)
        # 清零梯度
        optimizer.zero_grad()
        # 计算梯度
        loss.backward()
        # 更新梯度
        optimizer.step()
        if batch_idx % 10 == 0:
            train_loss.append(loss.item())
            print(epoch, batch_idx, loss.item())

plot_curve(train_loss)
# 我们得到好的【w1, b1, w2, b2, w3, b3】

total_correct = 0
for x, y in test_loader:
    x = x.view(x.size(0), 28*28)
    out = net(x)
    # out:[b, 10 => pred: [b]
    pred = out.argmax(dim = 1)
    correct = pred.eq(y).sum().float().item()
    total_correct += correct
total_num = len(test_loader.dataset)
acc = total_correct / total_num
print("test acc:", acc)

x, y = next(iter(test_loader))
out = net(x.view(x.size(0), 28*28))
pred = out.argmax(dim = 1)
plot_image(x, pred, "test")