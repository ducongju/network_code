"""
神经网络的典型训练过程如下：
1. 定义包含一些可学习的参数(或者叫权重)神经网络模型；
2. 在数据集上迭代；
3. 通过神经网络处理输入；
4. 计算损失(输出结果和正确值的差值大小)；
5. 将梯度反向传播回网络的参数；
6. 更新网络的参数，主要使用如下简单的更新原则： weight = weight - learning_rate * gradient
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# 1. 定义网络类
# nn.Module：神经网络模块。封装参数、移动到GPU上运行、导出、加载等
class Net(nn.Module):

    # 1.1 __init__方法中采用nn.方法对包含有变量的网络层进行类定义，nn.Conv2d，nn.Linear
    # 在建图过程中，往往有两种层，一种如全连接层，卷积层等，当中有Variable，另一种如Pooling层，Relu层等，当中没有Variable。
    # 如果所有的层都用nn.functional来定义，那么所有的Variable，如weights，bias等，都需要用户来手动定义，非常不方便。
    # 而如果所有的层都换成nn来定义，那么即便是简单的计算都需要建类来做，而这些可以用更为简单的函数来代替的。
    # 所以在定义网络的时候，如果层内有Variable, 那么用nn定义，反之，则用nn.functional定义。
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 6, 5, bias=False)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    # 1.2 forward方法中采用F.方法对不包含有变量的网络层直接进行操作，F.max_pool2d，F.relu
    # 在模型中必须要定义 forward 函数，可以在 forward 函数中使用任何针对 Tensor 的操作
    # backward 函数（用来计算梯度）会被autograd自动创建
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    # 1.2.1 将卷积层展开为全连接层的子函数
    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
# print(net)
"""
Net(
  (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=400, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)
"""

# net.parameters()：返回可被学习的参数（权重）列表和值
# nn.Parameter：一种变量，当把它赋值给一个Module时，被 自动 地注册为一个参数
# params = list(net.parameters())
# print(len(params))  # 9
# for i in iter(range(len(params))):
#     print(params[i].size())
"""
torch.Size([6, 1, 5, 5])    conv1
torch.Size([6])             bias
torch.Size([16, 6, 5, 5])   conv2
torch.Size([16])            bias
torch.Size([120, 400])      fc1
torch.Size([120])           bias
torch.Size([84, 120])       fc2
torch.Size([84])            bias
torch.Size([10, 84])        fc3
torch.Size([10])            bias
"""

# 2. 测试随机输入32×32，输入为N*C*H*W
# N is a batch size, C denotes a number of channels,
# H is a height of input planes in pixels, and W is width in pixels.
# 注：这个网络（LeNet）期望的输入大小是32×32，如果使用MNIST数据集来训练这个网络，请把图片大小重新调整到32×32。
# random_input = torch.randn(1, 1, 32, 32)
# print(random_input)
# output = net(random_input)  # shape: torch.Size([1, 10])
# print(output)
#
# # 3. 将所有参数的梯度缓存清零
# net.zero_grad()
#
# # 4. 进行随机梯度的的反向传播
# output.backward(torch.randn(1, 10))
#
# # 5. 计算损失
# # 一个损失函数接受一对 (output, target) 作为输入，计算一个值来估计网络的输出和目标值相差多少。
# output = net(random_input)
# print(output)
target = torch.randn(10)  # 随机值作为样例
target = target.view(1, -1)  # 使target和output的shape相同   # shape: torch.Size([1, 10])
criterion = nn.MSELoss()  # 创建criterion实例对象
# loss = criterion(output, target)
# print(loss)
#
# print(loss.grad_fn)  # MSELoss
# print(loss.grad_fn.next_functions[0][0])  # Linear
# print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU
#
# # 6. 反向传播
# # 调用loss.backward()获得反向传播的误差。
# # 但是在调用前需要清除已存在的梯度，否则梯度将被累加到已存在的梯度。
# # 现在，我们将调用loss.backward()，并查看conv1层的偏差（bias）项在反向传播前后的梯度。
# net.zero_grad()     # 清除梯度
#
# print('conv1.bias.grad before backward')
# print(net.conv1.bias.grad)
#
# loss.backward()
#
# print('conv1.bias.grad after backward')
# print(net.conv1.bias.grad)

# 7. 更新权重
# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
random_input = torch.randn(1, 1, 32, 32)
output = net(random_input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update