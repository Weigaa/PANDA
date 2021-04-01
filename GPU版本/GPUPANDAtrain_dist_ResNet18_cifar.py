#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import random

from math import ceil
from random import Random
from torch.multiprocessing import Process
from torch.autograd import Variable
from torchvision import datasets, transforms
device = torch.device('cuda:0')
torch.backends.cudnn.benchmark = True

"""每一个残差块"""
class ResidualBlock(nn.Module):   #继承nn.Module
    def __init__(self, inchannel, outchannel, stride=1):   #__init()中必须自己定义可学习的参数
        super(ResidualBlock, self).__init__()   #调用nn.Module的构造函数
        self.left = nn.Sequential(      #左边，指残差块中按顺序执行的普通卷积网络
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),   #最常用于卷积网络中(防止梯度消失或爆炸)
            nn.ReLU(inplace=True),   #implace=True是把输出直接覆盖到输入中，节省内存
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:   #只有步长为1并且输入通道和输出通道相等特征图大小才会一样，如果不一样，需要在合并之前进行统一
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
    def forward(self, x):   #实现前向传播过程
        out = self.left(x)   #先执行普通卷积神经网络
        out += self.shortcut(x)   #再加上原始x数据
        out = F.relu(out)
        return out


class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


class Net(nn.Module):
    """ Network architecture. """
    def __init__(self, ResidualBlock, num_classes=10):
        super(Net, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),  # 设置参数为卷积的输出通道数
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)  # 一个残差单元，每个单元中国包含2个残差块
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)  # 全连接层(1,512)-->(1,10)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (
                    num_blocks - 1)  # 将该单元中所有残差块的步数做成一个一个向量，第一个残差块的步数由传入参数指定，后边num_blocks-1个残差块的步数全部为1，第一个单元为[1,1]，后边三个单元为[2,1]
        layers = []
        for stride in strides:  # 对每个残差块的步数进行迭代
            layers.append(block(self.inchannel, channels, stride))  # 执行每一个残差块，定义向量存储每个残差块的输出值
            self.inchannel = channels
        return nn.Sequential(*layers)  # 如果*加在了实参上，代表的是将向量拆成一个一个的元素

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)  # 平均池化，4*4的局部特征取平均值，最后欸(512,1,1)
        out = out.view(out.size(0), -1)  # 转换为(1,512)的格式
        out = self.fc(out)
        out = F.log_softmax(out, dim=1)
        return out


def partition_dataset():
    """ Partitioning CIFAR10 """
    dataset = datasets.CIFAR10(
        './data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5 ), (0.5,0.5,0.5 ))
        ]))
    size = dist.get_world_size()
    bsz = 128 / float(size)
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(
        partition, batch_size=int(bsz), shuffle=True)
    return train_set, bsz


def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        # dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, group=0)
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size

def getbesthyperparameter(tensorlist,tensor):
    #share loss and tensor
    dist.all_gather(tensorlist,tensor)

def run(rank, size):
    """ Distributed Synchronous SGD Example """
    torch.manual_seed(1234)
    train_set, bsz = partition_dataset()
    model = Net(ResidualBlock).to(device)
    model = model
    tensor_list = [torch.zeros(2, dtype=torch.float).to(device) for _ in range(dist.get_world_size())]
    mytensor_list = [[0,0] for _ in range(dist.get_world_size())]
#    model = model.cuda(rank)
    # 使用PBT优化选择不同的lr和momentum
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    #使用PBT优化选择不同的lr和momentum
    nativelr = random.uniform(0.01,0.1)
    lr = nativelr
    optimizer = optim.SGD(model.parameters(), lr=nativelr, momentum=0.5)
    num_batches = ceil(len(train_set.dataset) / float(bsz))
    totalbegin = time.time()
    for epoch in range(10):
        epoch_loss = 0.0
        begin = time.time()
        for data, target in train_set:
            data, target = Variable(data), Variable(target)
#            data, target = Variable(data.cuda(rank)), Variable(target.cuda(rank))
            optimizer.zero_grad()
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = F.nll_loss(output, target).to(device)
            epoch_loss += loss.item()
            loss.backward()
            average_gradients(model)
            optimizer.step()
        end = time.time()
        spendtime = (end-begin)/60
        print('Rank ',
              dist.get_rank(), ', epoch ', epoch, ': ',
              epoch_loss / num_batches,', spend time: ',spendtime)
        # 每个epoch结束后修改lr和momentum
        realoss = epoch_loss / num_batches
        sharetensor = torch.tensor([realoss,lr]).to(device)
        getbesthyperparameter(tensor_list,sharetensor)
        for i in range(len(tensor_list)):
            mytensor_list[i] = tensor_list[i].tolist()
        print(mytensor_list)
        bestrank = mytensor_list.index(min(mytensor_list))
        if dist.get_rank() != bestrank:
            lr = mytensor_list[bestrank][1] * random.uniform(0.8,1.2)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5)


    totalend = time.time()
    print("total realrun time: ", (totalend - totalbegin) / 60)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            './data',
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.5,0.5,0.5
                         ),
                        (0.5,0.5,0.5
                         ))])),
        batch_size=32,
        shuffle=True,
    )
    get_accuracy(test_loader, model)

def get_accuracy(test_loader, model):
    model.eval()
    correct_sum = 0
    # Use GPU to evaluate if possible
    # device = torch.device("cpu")
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data = data.to(device)
            out = model(data).to(device)
            pred = out.argmax(dim=1, keepdim=True)
            pred, target = pred.to(device), target.to(device)
            correct = pred.eq(target.view_as(pred)).sum().item()
            correct_sum = correct_sum + correct

    print(f"Accuracy {correct_sum / len(test_loader.dataset)}")


def init_processes(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 4
    processes = []
    print("execution beginning...")
    print("all process BEGINtime is : ", time.strftime('%Y-%m-%d %H:%M:%S'))
    for rank in range(size):
        p = Process(target=init_processes, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    print("all process ENDtime is : ", time.strftime('%Y-%m-%d %H:%M:%S'))
