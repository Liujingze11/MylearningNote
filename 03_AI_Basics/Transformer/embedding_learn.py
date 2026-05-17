import torch

#torch.nn 工具开发者
import torch.nn as nn

#数学工具包
import math

embedding = nn.Embedding(10,3)
input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])

print(embedding(input))

