import gzip  # 读取.gz压缩文件
import struct  # 解析存储在文件中的二进制数据
import math
import numpy as np

import torch  # PyTorch深度学习框架
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader


# 从压缩文件(.gz)中读取MNIST数据集的图像和标签
def _read(image, label):
    minist_dir = './MNIST_data/'
    # 读取标签数据
    with gzip.open(minist_dir + label) as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))  # 读取标签文件头信息
        label = np.frombuffer(flbl.read(), dtype=np.int8)  # 读取标签数据并转换为numpy数组
    # 读取图像数据
    with gzip.open(minist_dir + image, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        image = np.frombuffer(fimg.read(), dtype=np.uint8).reshape(
            len(label), rows, cols)  # 每张图像的大小为rows x cols
    return image, label
def get_data():
    # 读取训练集图像和标签
    train_img, train_label = _read(
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz')
    # 读取测试集图像和标签
    test_img, test_label = _read(
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz')
    return [train_img, train_label, test_img, test_label]


# 定义LeNet5模型结构
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # 第一个卷积层，输入通道1（灰度图），输出通道6，卷积核大小5x5，边缘填充2
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.bn1 = nn.BatchNorm2d(6)  # 批量归一化层，对第一个卷积层的输出进行归一化，以下类似
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        # 第一个全连接层，输入特征数量16*5*5（16个特征图，每个特征图大小为5x5），输出特征数量120
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.bn3 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.bn4 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, 10)

    # 定义模型的前向传播路径
    def forward(self, x):
        # 通过第一个卷积层后使用LeakyReLU激活函数，并应用最大池化
        x = F.max_pool2d(F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01), (2, 2))
        x = F.dropout(x, p=0.3, training=self.training)  # 应用dropout防止过拟合
        x = F.max_pool2d(F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.01), (2, 2))
        x = F.dropout(x, p=0.3, training=self.training)
        x = x.view(-1, self.num_flat_features(x))  # 将特征图展平为一维向量
        # 通过第一个全连接层后使用LeakyReLU激活函数
        x = F.leaky_relu(self.bn3(self.fc1(x)), negative_slope=0.01)
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.leaky_relu(self.bn4(self.fc2(x)), negative_slope=0.01)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.fc3(x)  # 通过输出层
        return x

    # 计算展平后的特征数量
    def num_flat_features(self, x):
        size = x.size()[1:]  # 除批量维度外的所有维度
        num_features = 1
        for s in size:
            num_features *= s  # 计算展平后的特征总数量
        return num_features


# 定义初始化参数函数
def weight_init(m):
    # 判断m是否为卷积层（Conv2d）
    if isinstance(m, nn.Conv2d):
        # 计算卷积核的元素数量：卷积核的高度 * 宽度 * 输出通道数
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))  # 初始化权重，使用正态分布，均值为0，方差为2/n
    # 判断m是否为批量归一化层（BatchNorm2d）
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)  # 将权重全部初始化为1
        m.bias.data.zero_()  # 将偏置项初始化为0


# 定义模型训练过程函数
def train(epoch):
    model.train()  # 将模型设置为训练模式。
    # 遍历训练数据加载器
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_gpu:
            data, target = data.cuda(), target.cuda()  # 移动数据到GPU
        data, target = Variable(data), Variable(target.long())
        optimizer.zero_grad()  # 在进行梯度计算之前先将梯度归零，防止梯度在多次backward()调用时累积
        outputs = model(data)  # 通过模型前向传播得到输出
        loss = criterion(outputs, target)  # 计算输出与真实标签之间的损失
        loss.backward()  # 对损失进行反向传播，计算每个参数的梯度
        optimizer.step()  # 使用计算得到的梯度更新模型的参数
        # 每处理100个批次的数据就打印一次日志
        if (batch_idx+1) % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,  # 当前的训练轮次
                (batch_idx+1) * len(data),  # 到目前为止处理的数据量
                len(train_loader.dataset),  # 总的数据量
                100. * (batch_idx+1) / len(train_loader),  # 完成的百分比
                loss.data))  # 当前批次的损失


# 定义模型测试过程函数
def test():
    model.eval()
    test_loss = 0
    correct = 0
    # 遍历测试数据加载器
    for data, target in test_loader:
        if use_gpu:
            data, target = data.cuda(), target.cuda()  # 移动数据到GPU
        # data, target = Variable(data, volatile=True), Variable(target.long())
        with torch.no_grad():
            data, target = data, target.long()
        #      data = Variable(data)
        # target = Variable(target.long())
        outputs = model(data)
        test_loss += criterion(outputs, target).data  # 累加计算得到的损失
        pred = outputs.data.max(1, keepdim=True)[1]  # 获取概率最大的预测结果
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss,  # 平均损失
        correct,  # 正确预测的数量
        len(test_loader.dataset),  # 测试集的总数量
        100. * correct / len(test_loader.dataset)))  # 准确率


# 模型训练所需的基本参数设置
use_gpu = torch.cuda.is_available()  # 检查CUDA的可用性
BATCH_SIZE = 100
kwargs = {'num_workers': 0, 'pin_memory': True} if use_gpu else {}


# 准备训练和测试数据
train_img, train_label, test_img, test_label = get_data()
# 将图像数据转换为PyTorch张量，并重塑为适合卷积网络的形状
# MNIST图像原始尺寸为28x28，这里将其重塑为(数量, 通道, 高度, 宽度)的格式，即(-1, 1, 28, 28)
# 同时，将图像数据类型转换为float，标签转换为整型
train_x, train_y = torch.from_numpy(
    train_img.reshape(-1, 1, 28, 28)).float(), torch.from_numpy(train_label.astype(int))
test_x, test_y = torch.from_numpy(
    test_img.reshape(-1, 1, 28, 28)).float(), torch.from_numpy(test_label.astype(int))
# 使用TensorDataset将处理后的图像和标签封装成数据集
train_dataset = TensorDataset(train_x, train_y)
test_dataset = TensorDataset(test_x, test_y)
# 使用DataLoader将数据集封装成可迭代的数据加载器，以支持批量处理、数据打乱和多进程加速等功能
train_loader = DataLoader(dataset=train_dataset,
                          shuffle=True, batch_size=BATCH_SIZE, **kwargs)
test_loader = DataLoader(dataset=test_dataset,
                         shuffle=True, batch_size=BATCH_SIZE, **kwargs)

# 实例化定义好的LeNet5模型
model = LeNet5()
if use_gpu:
    model.cuda()  # 将模型移至GPU

# 定义优化器。
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99), weight_decay=1e-4)
criterion = nn.CrossEntropyLoss(size_average=False)  # 定义损失函数。
model.apply(weight_init)  # 对模型参数应用定义好的权重初始化函数
# 加入学习率调度器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

# 训练和测试循环
ENDEPOCH = 99
for epoch in range(0, ENDEPOCH+1):
    print('start train:')
    train(epoch)
    if epoch == ENDEPOCH:
        torch.save(model.state_dict(), './model_params.pkl')  # 保存模型参数到.pkl文件
    print('start test:')
    test()
    scheduler.step()  # 在每个epoch结束后更新学习率

# 重新创建一个LeNet5模型实例
model = LeNet5()
if use_gpu:
    model.cuda()  # 将模型移至GPU
model.load_state_dict(torch.load('./model_params.pkl'))  # 加载保存的模型参数
print('start final test:')
test()  # 再次测试模型，以验证加载的参数是否正确

