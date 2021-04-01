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

    def __init__(self):
        super(Net, self).__init__()
        '''第一层卷积层，卷积核为3*3，通道数为96，步距为1，原始图像大小为32*32，有R、G、B三个通道'''
        
        '''这样经过第一层卷积层之后，得到的feature map的大小为(32-3)/1+1=30,所以feature map的维度为96*30*30'''
        
        self.conv1=nn.Conv2d(3,96,kernel_size=3,stride=1)
        
        '''经过一次批归一化，将数据拉回到正态分布'''
        
        self.bn1=nn.BatchNorm2d(96)
        
        '''第一层池化层，卷积核为3*3，步距为2，前一层的feature map的大小为30*30，通道数为96个'''
        
        '''这样经过第一层池化层之后，得到的feature map的大小为(30-3)/2+1=14,所以feature map的维度为96*14*14'''
        
        self.pool1=nn.MaxPool2d(kernel_size=3,stride=2)
        
        '''第二层卷积层，卷积核为3*3，通道数为256，步距为1，前一层的feature map的大小为14*14，通道数为96个'''
        
        '''这样经过第一层卷积层之后，得到的feature map的大小为(14-3)/1+1=12,所以feature map的维度为256*12*12'''
        
        self.conv2=nn.Conv2d(96,256,kernel_size=3,stride=1)
        
        '''经过一次批归一化，将数据拉回到正态分布'''
        
        self.bn2=nn.BatchNorm2d(256)
        
        '''第二层池化层，卷积核为3*3，步距为2，前一层的feature map的大小为12*12，通道数为256个'''
        
        '''这样经过第二层池化层之后，得到的feature map的大小为(12-3)/2+1=5,所以feature map的维度为256*5*5'''
        
        self.pool2=nn.MaxPool2d(kernel_size=3,stride=2)
        
        '''第三层卷积层，卷积核为3*3，通道数为384，步距为1，前一层的feature map的大小为5*5，通道数为256个'''
        
        '''这样经过第一层卷积层之后，得到的feature map的大小为(5-3+2*1)/1+1=5,所以feature map的维度为384*5*5'''
        
        self.conv3=nn.Conv2d(256,384,kernel_size=3,padding=1,stride=1)
        
        '''第四层卷积层，卷积核为3*3，通道数为384，步距为1，前一层的feature map的大小为5*5，通道数为384个'''
        
        '''这样经过第一层卷积层之后，得到的feature map的大小为(5-3+2*1)/1+1=5,所以feature map的维度为384*5*5'''
        
        self.conv4=nn.Conv2d(384,384,kernel_size=3,padding=1,stride=1)
        
        '''第五层卷积层，卷积核为3*3，通道数为384，步距为1，前一层的feature map的大小为5*5，通道数为384个'''
        
        '''这样经过第一层卷积层之后，得到的feature map的大小为(5-3+2*1)/1+1=5,所以feature map的维度为256*5*5'''
        
        self.conv5=nn.Conv2d(384,256,kernel_size=3,padding=1,stride=1)
        
        '''第三层池化层，卷积核为3*3，步距为2，前一层的feature map的大小为5*5，通道数为256个'''
        
        '''这样经过第三层池化层之后，得到的feature map的大小为(5-3)/2+1=2,所以feature map的维度为256*2*2'''
        
        self.pool3=nn.MaxPool2d(kernel_size=3,stride=2)
        
        '''经过第一层全连接层'''
        
        self.linear1=nn.Linear(1024,2048)
        
        '''经过第一次DropOut层'''
        
        self.dropout1=nn.Dropout(0.5)
        
        '''经过第二层全连接层'''
        
        self.linear2=nn.Linear(2048,2048)
        
        '''经过第二层DropOut层'''
        
        self.dropout2=nn.Dropout(0.5)
        
        '''经过第三层全连接层，得到输出结果'''
        
        self.linear3=nn.Linear(2048,10)

    def forward(self, x):
        out=self.conv1(x)
        out=self.bn1(out)
        out=F.relu(out)
        out=self.pool1(out)
        
        
        out=self.conv2(out)
        out=self.bn2(out)
        out=F.relu(out)
        out=self.pool2(out)
        
        out=F.relu(self.conv3(out))
        
        out=F.relu(self.conv4(out))
        
        out=F.relu(self.conv5(out))
        
        out=self.pool3(out)
        
        out=out.reshape(-1,256*2*2)
        
        out=F.relu(self.linear1(out))
        
        out=self.dropout1(out)
        
        out=F.relu(self.linear2(out))
        
        out=self.dropout2(out)
        
        out=self.linear3(out)
        out = F.log_softmax(out, dim=1)
        return out
    # #和非分布式一致的网络
    # def __init__(self):
    #     super(Net, self).__init__()
    #     self.conv1 = nn.Conv2d(1, 32, 3, 1)
    #     self.conv2 = nn.Conv2d(32, 64, 3, 1)
    #     self.dropout1 = nn.Dropout2d(0.25)
    #     self.dropout2 = nn.Dropout2d(0.5)
    #     self.fc1 = nn.Linear(9216, 128)
    #     self.fc2 = nn.Linear(128, 10)
    #
    # def forward(self, x):
    #     x = self.conv1(x)
    #     x = F.relu(x)
    #     x = self.conv2(x)
    #     x = F.relu(x)
    #     x = F.max_pool2d(x, 2)
    #     x = self.dropout1(x)
    #     x = torch.flatten(x, 1)
    #     x = self.fc1(x)
    #     x = F.relu(x)
    #     x = self.dropout2(x)
    #     x = self.fc2(x)
    #     output = F.log_softmax(x, dim=1)
    #     return output


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
    model = Net().to(device)
    model = model
    tensor_list = [torch.zeros(3, dtype=torch.float).to(device) for _ in range(dist.get_world_size())]
    mytensor_list = [[0,0,0] for _ in range(dist.get_world_size())]
    lrlist=[]
#    model = model.cuda(rank)
    # 使用PBT优化选择不同的lr和momentum
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    #使用PBT优化选择不同的lr和momentum
    nativelr = random.uniform(0.03,0.06)
    naivemomentum = random.uniform(0.5,0.9)
    lr = nativelr
    optimizer = optim.SGD(model.parameters(), lr=nativelr, momentum=naivemomentum)
    num_batches = ceil(len(train_set.dataset) / float(bsz))
    totalbegin = time.time()
    for epoch in range(20):
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
            print("epoch loss is:",epoch_loss)
            # 防止梯度爆炸出现nan
            # loss += 1e-10
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
        sharetensor = torch.tensor([realoss,lr,naivemomentum]).to(device)
        getbesthyperparameter(tensor_list,sharetensor)
        for i in range(len(tensor_list)):
            mytensor_list[i] = tensor_list[i].tolist()
        print(mytensor_list)
        bestrank = mytensor_list.index(min(mytensor_list))
        lrlist.append(mytensor_list[bestrank][1])
        if dist.get_rank() != bestrank:
            lr = mytensor_list[bestrank][1] * random.uniform(0.8,1.2)
            naivemomentum = mytensor_list[bestrank][2] * random.uniform(0.8,1.2)
            if (naivemomentum > 1):
                naivemomentum = mytensor_list[bestrank][2] * random.uniform(0.8,0.99)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=naivemomentum)
    print("lr list is",lrlist)


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
