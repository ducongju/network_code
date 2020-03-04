# 导入torch包
import torch
# 将神经网络模块单独取名，简化代码
import torch.nn as nn
# from .utils import load_state_dict_from_url


# 所有在别的模块导入该模块时，只能导入__all__中的变量、方法、类
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


# 定义3*3卷积函数:
# in_planes：输入信号的通道，就是channel；out_planes：卷积产生的通道；
# kernel_size：卷积核的尺寸（3*3）；stride：卷积步长（默认1）；
# padding：每一条边补充0的层数（默认1）；dilation：卷积核元素之间的间距（默认1）；bias：添加偏置（不添加）；
# groups: 控制输入和输出之间的连接： group=1，输出是所有的输入的卷积；
# group=2，此时相当于有并排的两个卷积层，每个卷积层计算输入通道的一半，并且产生的输出是输出通道的一半，随后将这两个输出连接起来。
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    # nn.Conv2d：进行2维卷积操作
    # 这里bias设置为False,原因是：下面使用了Batch Normalization，而其对隐藏层  有去均值的操作，所以这里的常数项 可以消去
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


# 定义1*1卷积函数:
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# BasicBlock是为resnet18、34设计的，较浅层的结构可以不使用Bottleneck:
# ResNet的一个重要设计原则是：当feature map大小降低一半时，feature map的数量增加一倍，这保持了网络层的复杂度
# nn.Module类：是一个包装好的类，具体定义了一个网络层，可以维护状态和存储参数信息
# 定义基本单元类，继承于基类，该类存放64通道的网络层
class BasicBlock(nn.Module):
    # 输出通道数的倍乘，输出的通道数等于planes
    expansion = 1   # 为了区别BasicBlock和Bottleneck
    __constants__ = ['downsample']

    # 定义内置方法：创建实例后，进行调用
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        # 在初始化时，会继承父类初始化方法，并且进行以下操作
        super(BasicBlock, self).__init__()
        # 是否进行批量标准化操作，默认进行
        # 批量标准化：对于所有batch中的同一个channel的元素进行求均值与方差，减去求取得到的均值与方差，然后乘以gamma加上beta
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # 如果卷积通道数不等于64或者是并行卷积，则抛出异常
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # 如果卷积核元素间距大于1，则抛出异常
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        # 定义各种可以传入的实例方法，左边实例方法名，右边形参
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        # nn.ReLU是一个时序容器
        # inplace为True，将会改变输入的数据 ，否则不会改变原输入，只会产生新的输出
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    # 定义前向传播方法
    def forward(self, x):
        identity = x

        # 把之前的网络层作为输入，依次进行卷积操作——批量归一化操作——激活函数——卷积操作——批量归一化操作
        # 每次的输出均为一个W*H*64的网络层
        out = self.conv1(x) # 3*3, 64
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)   # 3*3, 64
        out = self.bn2(out)

        # 如果通道数不一致，则需要使用1*1卷积核进行下采样，变为相同通道数然后相加
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # 输出通道数的倍乘，输出的通道数等于planes的四倍
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # 用于调整输出通道数，可是为什么上一类不采用呢？去掉试试！！
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        # 在使用 Bottleneck 时，它先对通道数进行压缩，再放大，所以传入的参数 planes 不是实际输出的通道数，
        # 而是 block 内部压缩后的通道数，真正的输出通道数为 plane*expansion，
        # 这样做的主要目的是，使用 Bottleneck 结构可以减少网络参数数量
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        # 为什么调用类属性要用self.？改成Bottleneck.expansion试试！！
        # Bottleneck结构，为了保证足够参数，通道数变为256
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)     # 1*1, 64
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)   # 3*3, 64
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)   # 1*1, 256
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # 私有属性，只在对象内部可以调用
        # 因为在make函数中也要用到norm_layer，所以将这个放到了self中
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            # 判断一个对象是否是一个已知的类型。会认为子类是一种父类类型，考虑继承关系。
            # 对卷积层和与BN层初始化
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    # 私有方法，只在对象内部可以调用
    # _make_layer 方法的第一个输入参数 block 选择要使用的模块是 BasicBlock 还是 Bottleneck 类
    # 第二个输入参数 planes 是该模块的基准通道数
    # 第三个输入参数 blocks 是每个 blocks 中包含多少个 residual 子结构
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        # 是否采用扩张卷积
        if dilate:
            # 膨胀系数乘以步长
            self.dilation *= stride
            stride = 1

        # 如果图像尺寸或者通道数不匹配的时候的downsample，可以看到也是用过一个1*1的操作来进行升维的，然后对其进行一次BN操作
        if stride != 1 or self.inplanes != planes * block.expansion:
            # 一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行，
            # 同时以神经网络模块为元素的有序字典也可以作为传入参数。
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )


        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        # 这里分两个block是因为要将一整个Layer进行output size那里，维度是依次下降两倍的，
        # 第一个是设置了stride=2所以维度下降一半，剩下的不需要进行维度下降，都是一样的维度
        self.inplanes = planes * block.expansion
        # 该部分是将每个blocks的剩下residual结构保存在layers列表中，这样就完成了一个blocks的构造
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 静态层
        x = self.conv1(x)       # 7*7, 64
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)     # 3*3, MaxPool, stride 2

        # 动态层
        x = self.layer1(x)      # 64
        x = self.layer2(x)      # 128
        x = self.layer3(x)      # 256
        x = self.layer4(x)      # 512

        # 静态层
        x = self.avgpool(x)     # Adaptive, AvgPool, staride Adaptive, H*W = 1*1
        x = torch.flatten(x, 1) # 返回一个折叠成一维的数组
        x = self.fc(x)          # 1000

        return x


# 实例化resnet类以及确定是否加载训练好的权重
def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


# 提供不同resnet的接口
def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    # [3，4，6，3]表示按次序生成3个Bottleneck，4个Bottleneck，6个Bottleneck，3个Bottleneck。
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)
