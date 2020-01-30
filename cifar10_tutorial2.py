import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
# 在 终端 / 文件浏览器、 中可以执行常规的 文件 / 目录 管理操作，operating system
import os
from torch.utils.tensorboard import SummaryWriter


# 1. 定义一个卷积神经网络
# 用于卷积层到全连接层的过渡
def num_flatten(x):             # for the MLP
    size = x.size()[1:]         # except batch dimension
    number = 1
    for s in size:
        number *= s
    return number


class Classification(nn.Module):
    def __init__(self, num_features, num_class):
        super(Classification, self).__init__()
        self.layer1 = nn.Conv2d(3, num_features, 3, padding=1, bias=False)
        self.layer2 = nn.Conv2d(
            num_features, num_features, 3, padding=1, bias=False)
        self.layer3 = nn.Linear(num_features*32*32, num_features, bias=True)
        self.layer4 = nn.Linear(num_features, num_class, bias=True)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x):
        out = self.relu(self.bn(self.layer1(x)))
        out = self.relu(self.bn(self.layer2(out)))
        out = self.layer3(out.view(-1, num_flatten(out)))
        out = self.layer4(out)
        return out


# 2. 读取和归一化训练数据
# Build for cifar10
# CIFAR-10数据集由10个类的60000个32x32彩色图像组成，每个类有6000个图像。有50000个训练图像和10000个测试图像。
# 数据集分为五个训练批次和一个测试批次，每个批次有10000个图像。每个批次是一个10000x3072的numpy数组
# 测试批次包含来自每个类别的恰好1000个随机选择的图像。
def load_cifar(datas, labels, batch_size, shuffle=True):
    num_batch = len(labels) // batch_size       # 批量数(6)=类别总数(60000)//批量尺寸(10000) 没看懂
    datas = datas/127.5 - 1             # [0, 255] is Normalized to [-1, 1]
    datas = datas.reshape(-1, 3, 32, 32)        # 将数据格式转换为网络的输入格式(N,C,H,W)(10000,3,32,32)
    index = list(range(len(labels)))
    # 是否打乱类别顺序
    if shuffle:
        np.random.shuffle(index)
    datas, labels = datas[index, :], (np.array(labels)[index]).tolist()     # 前者是把数据打乱，后者是把对应关系整理好
    for i in range(num_batch):
        yield datas[i:min(len(labels), i+batch_size), :], labels[i:min(len(labels), i+batch_size)]


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# 3. 训练数据
# 用于将文件数据反序列化为一个对象
def unpickle(file):
    # 将对象转换为一种可以传输或存储的格式
    import pickle
    # 到达语句末尾时，会自动关闭文件，即便出现异常
    # 读取一个二进制文件，并返回一个对象
    with open(file, 'rb')as f:
        dict1 = pickle.load(f, encoding='bytes')
    return dict1


def train_net(path, net, loss, optim, device, batch_size=16, train_samples=8000, data_name='data_batch_1'):
    # Use tensorboard to visualize the process
    # SummaryWriter类是您记录TensorBoard使用和可视化数据的主要条目
    # os.path.join：连接两个或更多的路径名组件
    # 所以writer保存的目录为 path\Train
    writer = SummaryWriter(os.path.join(path, 'Train'))

    # Load Network
    # 保存的目录为 path\cifar_model.pth，如果目录存在，从目录中载入已有网络保存到net中
    net_path = os.path.join(path, 'cifar_model.pth')
    if os.path.exists(net_path):
        net.load_state_dict(torch.load(net_path))

    # Load Data
    file_name = os.path.join(path, data_name)
    meta = unpickle(file_name)
    datas, labels = meta[b'data'], meta[b'labels']
    # 从10000个训练数据中取8000个进行训练
    train_datas, train_labels = datas[:train_samples], labels[:train_samples]
    # train_datas1 = len(train_datas)
    # 另外2000个进行测试
    val_datas, val_labels = datas[train_samples:], labels[train_samples:]

    # Begin Train
    # best_loss为无穷小
    best_loss = np.inf
    epochs = 30
    for i in range(epochs):
        avg_train = 0
        avg_val = 0
        t = v = 0
        for data, label in load_cifar(train_datas, train_labels, batch_size):
            data, label = torch.tensor(
                data, dtype=torch.float, device=device), torch.tensor(label, device=device)
            y = net(data)  # 正向传播
            train_loss = loss(y, label)  # 计算损失，损失函数自己定义
            optim.zero_grad()  # 梯度置0
            train_loss.backward()  # 反向传播
            optim.step()  # 优化，优化方式自己定义
            avg_train = (avg_train * t + train_loss.item()) / (t + 1)

        for data, label in load_cifar(val_datas, val_labels, batch_size, shuffle=False):
            data, label = torch.tensor(
                data, dtype=torch.float, device=device), torch.tensor(label, device=device)
            with torch.no_grad():  # with torch.no_grad 中的数据不需要计算梯度，也不会进行反向传播
                y = net(data)
                val_loss = loss(y, label)
            avg_val = (avg_val * v + val_loss.item()) / (v + 1)

        # 如果新网络在测试集上的损失比原网络更小，覆盖原网络
        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(net.state_dict(), net_path)

        # tensorboard操作：将标量数据添加到摘要中
        writer.add_scalars(
            'Loss', {'train_loss': avg_train, 'val_loss': avg_val}, i)
        print('epoch is {}\ttrain_loss is {}\tval_loss is {}\n'.format(
            i, avg_train, avg_val))
    writer.close()


def eval_net(path, net, device, batch_size=16, num_samples=1000, data_name='test_batch'):
    # Load Network
    net_path = os.path.join(path, 'cifar_model.pth')
    if os.path.exists(net_path):
        net.load_state_dict(torch.load(net_path))

    # Load Data
    file_name = os.path.join(path, data_name)
    meta = unpickle(file_name)
    datas, labels = meta[b'data'][:num_samples], meta[b'labels'][:num_samples]
    accum = 0
    for data, label in load_cifar(datas, labels, batch_size=batch_size, shuffle=False):
        data = torch.tensor(data, dtype=torch.float, device=device)
        y = net(data)
        y_label = y.argmax(dim=1).to('cpu').numpy()
        accum += sum(y_label == np.array(label))
    print('The accurancy is %f' % (accum / num_samples))


def Demo(path, net, data, label, device):
    # Load Network
    net_path = os.path.join(path, 'cifar_model.pth')
    if os.path.exists(net_path):
        net.load_state_dict(torch.load(net_path))

    data = data / 127.5 - 1
    data = data.reshape(-1, 3, 32, 32)
    #
    pic = ((data + 1) / 2).squeeze()
    data = torch.tensor(data, dtype=torch.float, device=device)
    pred = net(data).argmax(dim=1).item()

    plt.imshow(pic.transpose(1, 2, 0))
    plt.title('{}'.format(classes[pred]))
    plt.show()
    print('The label is {}'.format(classes[label]))


#  模块测试
if __name__ == "__main__":
    path = './data/cifar-10-batches-py'
    # path = '/mnt/data/yh/CIFAR10/cifar-10-batches-py'
    net = Classification(32, 10)
    loss = nn.CrossEntropyLoss()
    optim = optim.Adam(lr=1e-3, params=net.parameters(), weight_decay=1e-4)

    # Run the code on GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net, device_ids=[0, 1])
    net.to(device)

    # Train the network
    train_net(path, net, loss, optim, device, data_name='data_batch_3')

    # Eval the network
    eval_net(path, net, device, num_samples=1600)

    # Show the demo
    meta = unpickle(os.path.join(path, 'test_batch'))
    index = np.random.randint(0, len(meta[b'labels']))
    data, label = meta[b'data'][index], meta[b'labels'][index]
    Demo(path, net, data, label, device)
