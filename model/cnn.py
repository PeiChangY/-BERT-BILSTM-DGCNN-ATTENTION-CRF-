# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
# from collections import OrderedDict
# from constants import *

class IDCNN(nn.Module):
    """
      (idcnns): ModuleList(
    (0): Sequential(
      (layer0): Conv1d(10, 1, kernel_size=(3,), stride=(1,), padding=(1,))
      (layer1): Conv1d(1, 1, kernel_size=(3,), stride=(1,), padding=(1,))
      (layer2): Conv1d(1, 1, kernel_size=(3,), stride=(1,), padding=(2,), dilation=(2,))
    )
    (1): Sequential(
      (layer0): Conv1d(10, 1, kernel_size=(3,), stride=(1,), padding=(1,))
      (layer1): Conv1d(1, 1, kernel_size=(3,), stride=(1,), padding=(1,))
      (layer2): Conv1d(1, 1, kernel_size=(3,), stride=(1,), padding=(2,), dilation=(2,))
    )
    (2): Sequential(
      (layer0): Conv1d(10, 1, kernel_size=(3,), stride=(1,), padding=(1,))
      (layer1): Conv1d(1, 1, kernel_size=(3,), stride=(1,), padding=(1,))
      (layer2): Conv1d(1, 1, kernel_size=(3,), stride=(1,), padding=(2,), dilation=(2,))
    )
    (3): Sequential(
      (layer0): Conv1d(10, 1, kernel_size=(3,), stride=(1,), padding=(1,))
      (layer1): Conv1d(1, 1, kernel_size=(3,), stride=(1,), padding=(1,))
      (layer2): Conv1d(1, 1, kernel_size=(3,), stride=(1,), padding=(2,), dilation=(2,))
    )
  )
)
    """
    def __init__(self, input_size, filters, kernel_size=3, num_block=4):   #kernel_size = 3 是大小为3*3的卷积核  num_block为生成块的数量
        super(IDCNN, self).__init__()
        self.layers = [
            {"dilation": 1},
            {"dilation": 1},
            {"dilation": 2}]
        net = nn.Sequential() #一个序列容器,用于搭建神经网络的模块被按照被传入构造器的顺序添加到nn.Sequential()容器中。
        norms_1 = nn.ModuleList([LayerNorm(128) for _ in range(len(self.layers))])
        norms_2 = nn.ModuleList([LayerNorm(128) for _ in range(num_block)])
        for i in range(len(self.layers)):
            dilation = self.layers[i]["dilation"]
            single_block = nn.Conv1d(in_channels=filters,
                                     out_channels=filters,
                                     kernel_size=kernel_size,
                                     dilation=dilation,
                                     padding=kernel_size // 2 + dilation - 1)
            net.add_module("layer%d"%i, single_block)
            net.add_module("relu", nn.ReLU())
            net.add_module("layernorm", norms_1[i])

        self.linear = nn.Linear(input_size, filters)
        self.idcnn = nn.Sequential()
        self.sigmoid = nn.Sigmoid()


        for i in range(num_block):
            self.idcnn.add_module("block%i" % i, net)
            self.idcnn.add_module("relu", nn.ReLU())
            self.idcnn.add_module("layernorm", norms_2[i])

    def forward(self, embeddings, length):  #torch.Size([16, 128, 768])
        embeddings = self.linear(embeddings)
        embeddings_1 = embeddings  # torch.Size([16, 128, 120])
        embeddings = embeddings.permute(0, 2, 1)
        output = self.idcnn(embeddings).permute(0, 2, 1) # torch.Size([16, 128, 120])
        output_sig = self.sigmoid(output) #torch.Size([16, 128, 120])
        output = embeddings_1+output*output_sig
        # output = embeddings_1 + output #res
        # output = output*output_sig #gate
        return output


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x-mean) / (std + self.eps) + self.b_2




