from data import MINSTdataSet
from net import netV1
from torch.utils.data import DataLoader
import torch, torch.optim
import time

class train:
    '''
    训练模型，梯度下降过程
    '''

    def __init__(self, root):
        '''
        模型的准备工作，如加载数据集和网络模型,定义损失，优化器，
        :param root: data中获取数据集的地址
        '''
        # 把所有数据的地址加载进来
        self.train_dataset = MINSTdataSet(root)
        # 按批次在数据集中加载数据
        self.train_dataloder = DataLoader(self.train_dataset, batch_size=100, shuffle=True, num_workers=8)
        self.net = netV1()
        self.optim = torch.optim.Adam(self.net.parameters())

    def __call__(self, *args, **kwargs):
        '''
        实例化的时候就会调用 call  如 train_moid = train()  train_moid()
        实现后向学习
        :param args:
        :param kwargs:
        :return:
        '''
        for epoch in range(10000):
            ts = time.time()
            for i, (img, tage)in enumerate(self.train_dataloder):
                input, tage = img, tage
                out = self.net(input)
                loss = torch.mean((out - tage) ** 2)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
            tn = time.time()
            print(epoch, loss.detach().item(), tn-ts)

if __name__ == '__main__':
    train = train("data/MNIST_IMG")
    train()