"""
特别的，对于图像任务，我们创建了一个包 torchvision，它包含了处理一些基本图像数据集的方法。
这些数据集包括 Imagenet, CIFAR10, MNIST 等。
除了数据加载以外，torchvision 还包含了图像转换器， torchvision.datasets 和 torch.utils.data.DataLoader。

训练一个图像分类器
依次按照下列顺序进行：

1. 使用torchvision加载和归一化CIFAR10训练集和测试集
2. 定义一个卷积神经网络
3. 定义损失函数
4. 在训练集上训练网络
5. 在测试集上测试网络
"""

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
# import cv2

# 1. 读取和归一化 CIFAR10
# torchvision的输出是[0,1]的PILImage图像，我们把它转换为归一化范围为[-1, 1]的张量。
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)       # shape: 12500

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# 定义图像展示函数
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    # matplotlib.pyplot方法展示
    # plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # plt.show()
    # cv方法展示
    # image = np.transpose(npimg, (1, 2, 0))[:, :, ::-1]
    # cv2.namedWindow('image')
    # cv2.imshow('image', image)
    # cv2.waitKey(0)


# 获取随机数据以展示图像
dataiter = iter(trainloader)
# next()：返回迭代器的下一个项目
images, labels = dataiter.next()
# 展示图像
imshow(torchvision.utils.make_grid(images))
# 显示图像标签
# Python join() 方法：用于将序列中的元素以指定的字符连接生成一个新的字符串。
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))       # ship truck  frog  bird


# 2. 定义一个卷积神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
print(net)
"""
Net(
  (conv1): Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=400, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)
"""

# 3. 定义损失函数和优化器
# 我们使用交叉熵作为损失函数，使用带动量的随机梯度下降。
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 4. 训练网路
# 有趣的时刻开始了。 我们只需在数据迭代器上循环，将数据输入给网络，并优化。
# range()函数：返回的是一个可迭代对象（类型是对象），而不是列表类型，所以打印的时候不会打印列表。
# for epoch in range(2):  # 多批次循环
#
#     running_loss = 0.0
#     # enumerate()函数：用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标。
#     for i, data in enumerate(trainloader, 0):
#         # 获取输入
#         inputs, labels = data
#
#         # 梯度置0
#         optimizer.zero_grad()
#
#         # 正向传播
#         outputs = net(inputs)
#
#         # 计算损失
#         loss = criterion(outputs, labels)
#
#         # 反向传播
#         loss.backward()
#
#         # 优化
#         optimizer.step()
#
#         # 打印状态信息,batchsize为2000
#         running_loss += loss.item()
#         if i % 2000 == 1999:    # 每2000批次打印一次
#             print('[%d, %5d] loss: %.3f' %
#                   (epoch + 1, i + 1, running_loss / 2000))
#             running_loss = 0.0

print('Finished Training')
"""
[1,  2000] loss: 2.168
[1,  4000] loss: 1.807
[1,  6000] loss: 1.636
[1,  8000] loss: 1.564
[1, 10000] loss: 1.524
[1, 12000] loss: 1.451
[2,  2000] loss: 1.353
[2,  4000] loss: 1.361
[2,  6000] loss: 1.354
[2,  8000] loss: 1.323
[2, 10000] loss: 1.306
[2, 12000] loss: 1.289
Finished Training
"""

# 5. 在测试集上测试网络
# 我们在整个训练集上进行了2次训练，但是我们需要检查网络是否从数据集中学习到有用的东西。
# 通过预测神经网络输出的类别标签与实际情况标签进行对比来进行检测。
# 如果预测正确，我们把该样本添加到正确预测列表。
# 第一步，显示测试集中的图片并熟悉图片内容。
dataiter = iter(testloader)
images, labels = dataiter.next()

# 显示图片
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

# 让我们看看神经网络认为以上图片是什么。
outputs = net(images)

# 输出是10个标签的能量。 一个类别的能量越大，神经网络越认为它是这个类别。所以让我们得到最高能量的标签。
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

# 接下来让看看网络在整个测试集上的结果如何。
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

# 在GPU上训练
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 确认我们的电脑支持CUDA，然后显示CUDA信息
print(device)

"""
GroundTruth:    cat  ship  ship plane
Predicted:    dog   car plane  ship
Accuracy of the network on the 10000 test images: 54 %
cuda:0
"""