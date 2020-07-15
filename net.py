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



if __name__ == '__main__':
    '''
    每完成一个模块后要测试这个模块的功能，看是否按照构建的一样运行 
    输出的形状是否正确，数据类型是否正确等
    '''
    net = netV1()
    h = torch.randn(10,784)
    print(h.shape)
    print(h.dtype)
