import torch

#torch.nn 工具开发者
import torch.nn as nn

#数学工具包
import math

from torch.autograd import Variable


# Embeddings 这个类，继承自 nn.Module
class Embeddings(nn.Module):
    '''
    d_model：词向量维度
    vocab_size：最大序列长度（默认 80）
    
    '''

    def __init__(self, vocab_size, d_model):

        # 接着就是使用super的方式指明继承nn.Module的初始化函数，我们自己实现的所有层都会
        #super() 是 Python 中调用父类方法的机制，在 PyTorch 中必须通过 super().__init__() 触发 nn.Module 的内部初始化逻辑，才能正确注册参数、子模块并支持自动求导。
        super(Embeddings, self).__init__()

        # 之后就是调用nn中的预定义层Embedding把 离散的整数索引 映射成 连续的向量表示
        #Embedding 层将离散的 token id 映射到一个连续的 d_model 维向量空间中，这个向量是可学习的参数，用于表示 token 的语义信息。
        #Embedding 的作用是：输入 token id，输出对应的向量（d_model 维）
        # 查表（Look-Up Table）
        # nn.Embedding 的向量在初始化时是“随机的”，默认服从均匀分布：
        self.lut = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model


    def forward(self, x):
        # x: token id 张量
        # Embedding是一个对象，内部定义了__call__

        # 将x传给了
        return self.lut(x) * math.sqrt(self.d_model)




