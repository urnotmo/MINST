import torch.nn
import torch

class netV1(torch.nn.Module):
    '''
    定义所用的网络模型，和前向过程，继承了torch.nn.Module
    '''
    def __init__(self,):
        '''
        定义模型的层数，和参数 w, b,
        '''

        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(784,10))

    def forward(self, input):
        '''
        模型的前向过程，输出的处理
        :param input: 输入的数据 x
        :return: 模型输出的结果
        '''

        h = input @ self.w

        # 多分类问题对输出的数据进行 softmax激活
        h = torch.exp(h)
        z = torch.sum(h, dim=1, keepdim=True)
        h = h/z
        return h


class netV2(torch.nn.Module):

    def __init__(self):
        super(netV2, self).__init__()
        self.fc1 = torch.nn.Linear(784, 123)
        self.fc2 = torch.nn.Linear(123, 100)
        self.fc3 = torch.nn.Linear(100, 10)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, input):
        h = self.fc1(input)
        h = self.relu(h)
        h = self.fc2(h)
        h = self.relu(h)
        h = self.fc3(h)
        out = self.softmax(h)
        return out


class netV3(torch.nn.Module):
    def __init__(self):
        super(netV3, self).__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Linear(784, 123),
            torch.nn.ReLU(),
            torch.nn.Linear(123, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 10),
            torch.nn.Softmax(dim=1)
        )
    def forward(self, input):
        return self.layer(input)

if __name__ == '__main__':
    '''
    每完成一个模块后要测试这个模块的功能，看是否按照构建的一样运行 
    输出的形状是否正确，数据类型是否正确等
    '''
    net1 = netV1()
    h = torch.randn(6, 784)
    out = net1(h)
    print(out.shape)
    print(out.dtype)

    net2 = netV2()
    h = torch.randn(7, 784)
    out = net2(h)
    print(out.shape)
    print(out.dtype)

    net3 = netV3()
    h = torch.randn(8, 784)
    out = net3(h)
    print(out.shape)
    print(out.dtype)