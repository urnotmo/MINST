from torch.utils.data import Dataset
import numpy as np
import torch, cv2, os

class MINSTdataSet(Dataset):
    '''
    处理自己的数据集，继承了torch.utils包里面的Dataset
    '''
    def __init__(self, root, is_train = True):
        '''
        封装自己的数据集,
        :param root: 数据的根目录
        :param is_train: 加载的是哪个数据集该例子中默认加载的是训练集
        '''

        self.dataset = []   # 存放所有数据的路径
        sub_dir = 'TRAIN' if is_train else 'TEST'   # 选择需要加载的数据集
        for tag in os.listdir(f'{root}/{sub_dir}'):
            img_dir = f'{root}/{sub_dir}/{tag}'
            for img_path in os.listdir(img_dir):
                img_filename = f'{img_dir}/{img_path}'
                self.dataset.append((img_filename, tag))

    def __len__(self):
        '''
        获取数据的长度
        :return: 返回数据的长度
        '''
        return(len(self.dataset))

    def __getitem__(self, index):
        '''
        对每条数据进行以便能够输入模型中
        :param index: 数据的下标索引
        :return:
        '''
        data = self.dataset[index]  # 获取每条数据的地址和标签
        img_data = cv2.imread(data[0], cv2.IMREAD_GRAYSCALE)     # cv2读取图片，读出来的类型为ndarray类型
        img_data = img_data.reshape(-1)     # 将
        img_data = img_data/255     # 将数据进行最大范数归一化

        tag_one_hot = torch.zeros(10)   # 对标签进行one_hot编码
        tag_one_hot[int(data[1])] = 1

        return img_data, tag_one_hot


if __name__ == '__main__':
    dataset = MINSTdataSet('data\MNIST_IMG', is_train=True)
    print(len(dataset))
    print(dataset[3])


# 自己瞎搞
# import os, torch, cv2
# from torch.utils.data import Dataset
# import time
#
# class MINST_DATA(Dataset):
#     '''
#     处理自己的数据以便能输入模型
#     '''
#     def __init__(self, root, is_train):
#         self.dataset = []
#         sub_dir = 'TRAIN' if is_train else 'TEST'
#         for tag in os.listdir(f'{root}/{sub_dir}'):
#             for filename in os.listdir(f'{root}/{sub_dir}/{tag}'):
#                 img_dir = f'{root}/{sub_dir}/{tag}/{filename}'
#                 self.dataset.append((img_dir, tag))
#
#     def __len__(self):
#         return len(self.dataset)
#
#     def __getitem__(self, index):
#         data = self.dataset[index]
#         img_data = cv2.imread(data[0], cv2.IMREAD_GRAYSCALE)
#         # print(img_data)
#         # print("###########")
#         img_data = img_data.reshape(-1)
#         img_data = img_data/255
#
#         tag_one_hot = torch.zeros(10)
#         tag_one_hot[int(data[1])] = 1
#
#         return img_data, tag_one_hot
#
# if __name__ == '__main__':
#     start = time.time()
#     dataset = MINST_DATA("data/MNIST_IMG", True)
#     print(dataset[2])
#     end = time.time()
#     print(end-start)

