from data import MINSTdataSet
from net import netV1
from torch.utils.data import DataLoader
import torch, torch.optim
from torch.utils.tensorboard import SummaryWriter
import time, warnings
warnings.filterwarnings('ignore')

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

        # 加载验证集
        self.test_dataset = MINSTdataSet(root, is_train=False)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=100, shuffle=False, num_workers=0)

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
        self.summarywriter = SummaryWriter('./logs')

        for epoch in range(10000):
            train_ts = time.time()
            train_sum_loss = 0

            for i, (img, tage)in enumerate(self.train_dataloder):
                # 这一轮就把六万张图片训练完成了，
                input, tage = img, tage
                out = self.net(input)
                loss = torch.mean((out - tage) ** 2)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                train_sum_loss += loss.detach().item()
            train_avg_loss =train_sum_loss/ len(self.train_dataset)
            train_te = time.time()
            train_time = train_te - train_ts

            # 验证
            test_sum_loss = 0
            test_ts = time.time()
            score_sum = 0
            for i, (img, tage) in enumerate(self.test_dataloader):
                input, test_tage = img, tage
                test_output = self.net(img)
                loss = torch.mean((test_output - test_tage) ** 2)
                test_sum_loss += loss
                # 将输出转化为 one_hot
                # index_testout = torch.argmax(test_output, dim=1, keepdim=True)
                # test_output = test_output.scatter_(1, index_testout, 1)
                # score_sum += torch.sum(torch.eq(test_output, test_tage).float())

                pre_tage = torch.argmax(test_output, dim=1)
                label_tage = torch.argmax(test_tage,dim=1)
                score_sum +=torch.sum(torch.eq(pre_tage, label_tage).float())

            test_avg_loss = loss.item() / len(self.test_dataset)
            test_te = time.time()
            test_time = test_te - test_ts
            score_avg = score_sum / len(self.test_dataset)
            self.summarywriter.add_scalars("loss", {"train_avg_loss":train_avg_loss, "test_avg_loss":test_avg_loss}, epoch)
            self.summarywriter.add_scalar("score", score_avg, epoch)
            print(epoch, train_avg_loss, train_time, test_avg_loss, test_time, score_avg.item())
if __name__ == '__main__':
    train = train("data/MNIST_IMG")
    train()

    # x = torch.tensor([[0.2, 0.22, 0.12, 0.78],
    #                   [0.2, 0.22, 0.82, 0.78],
    #                   [0.92, 0.22, 0.12, 0.78]])
    # h = torch.tensor([[0,0,0,1],
    #                   [0,0,1,0],
    #                   [0,0,0,1]])
    #
    # y = torch.argmax(x, dim=1, keepdim=True )
    # print(y)
    # # z = torch.zeros(x.shape).scatter_(1, y, 1)
    # z = x.scatter_(1, y, 1)
    # print(z)
    #
    # score_sum = torch.sum(torch.eq(h, z).float())
    # # score_sum = torch.eq(h, z)
    # print(score_sum/3)
