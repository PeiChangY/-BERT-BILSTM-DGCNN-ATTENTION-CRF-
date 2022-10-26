import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=12, score_function='bi_linear', dropout=0.5):  #score_function 注意力评分函数
        ''' Attention Mechanism
        :param embed_dim:
        :param hidden_dim:
        :param out_dim:
        :param n_head: num of head (Multi-Head Attention)
        :param score_function: scaled_dot_product / mlp (concat) / bi_linear (general dot)
        :return (?, q_len, out_dim,)
        '''
        super(Attention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function
        self.w_k = nn.Linear(embed_dim, n_head * hidden_dim)  #（22,128）
        self.w_q = nn.Linear(embed_dim, n_head * hidden_dim)  #（22,128）
        self.proj = nn.Linear(n_head * hidden_dim, out_dim)   #（128,22）
        self.dropout = nn.Dropout(dropout)
        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim*2))   #转换成可以改变值的张量
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:  # dot_product / scaled_dot_product
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, k, q):
        if len(q.shape) == 2:  # q_len missing   #q为二维数组
            q = torch.unsqueeze(q, dim=1)   #维度扩张  在q中的列上加了一个维数为1的维度  三维
        if len(k.shape) == 2:  # k_len missing
            k = torch.unsqueeze(k, dim=1)
        mb_size = k.shape[0]  # ?       q,k为 16  128  22  k.shape[0]=16
        k_len = k.shape[1]  #128
        q_len = q.shape[1]   #128
        # k: (?, k_len, embed_dim,)   16  128  22
        # q: (?, q_len, embed_dim,)   16  128  22
        # kx: (n_head*?, k_len, hidden_dim)    16  128  128
        # qx: (n_head*?, q_len, hidden_dim)    16  128  128
        # score: (n_head*?, q_len, k_len,)     16  128  128
        # output: (?, q_len, out_dim,)         16  128  22
        kx = self.w_k(k).view(mb_size, k_len, self.n_head, self.hidden_dim)  #  （16，128，1，128）   view（）的作用是改变为对应维度
        kx = kx.permute(2, 0, 1, 3).contiguous().view(-1, k_len, self.hidden_dim)    #permute函数的作用是维度换位  (1,16,128,128)   contiguous()作用是拷贝张量    （16,128,128）
        qx = self.w_q(q).view(mb_size, q_len, self.n_head, self.hidden_dim)
        qx = qx.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.hidden_dim)
        if self.score_function == 'dot_product': #加性模型和点积模型的复杂度相似，但点积模型可以更好利用矩阵计算，其效率更高；
            kt = kx.permute(0, 2, 1)    #16  128   128
            score = torch.bmm(qx, kt)  # 16  128  128
        elif self.score_function == 'scaled_dot_product':  #缩放点积模型是点积模型在高维输入向量上的改进，它可以解决当输入向量的维度𝐷 较高时，点积模型计算值方差较大，从而导致Softmax函数梯度较小的问题；
            kt = kx.permute(0, 2, 1)  #  16  128  128
            qkt = torch.bmm(qx, kt)   #  16  128  128
            score = torch.div(qkt, math.sqrt(self.hidden_dim)) #math.sqrt(x) 得到x的平方根   torch.div（a,b）得到a除以b的结果
        elif self.score_function == 'mlp': #加性注意力机制，把Q和K结合起来输入一个多层感知机中   Q和K长度不同时效果较好
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1) #expand()函数的作用是把指定维度扩大为更高维  16  128 128  128
            qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)    # 16  128  128  128
            kq = torch.cat((kxx, qxx), dim=-1)  # (n_head*?, q_len, k_len, hidden_dim*2)  16  128  128  256
            # kq = torch.unsqueeze(kx, dim=1) + torch.unsqueeze(qx, dim=2)
            # score = F.tanh(torch.matmul(kq, self.weight))
            score = torch.tanh(torch.matmul(kq,self.weight))  #matmul（）作用是给矩阵做乘法     16  128  128
        elif self.score_function == 'bi_linear':  #双线性模型是一种泛化的点积模型（即：分别对𝒙 和𝒒 进行线性变换后计算点积），相比点积模型，双线性模型在计算特征权重时引入了非对称性。
            qw = torch.matmul(qx, self.weight)
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qw, kt)  #bmm计算两个tensor的矩阵乘法，torch.bmm(a,b),tensor a 的size为(b,h,w),tensor b的size为(b,w,m)  输出维度 （b,h,m）
        else:
            raise RuntimeError('invalid score_function')
        score = F.softmax(score, dim=-1)    # 16  128  128
        output = torch.bmm(score, kx)  # (n_head*?, q_len, hidden_dim)  16  128  128
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)  # (?, q_len, n_head*hidden_dim)   16  128  128
        output = self.proj(output)  # (?, q_len, out_dim)  16  128  22
        output = self.dropout(output)
        return output, score


class NoQueryAttention(Attention):
    '''q is a parameter'''
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function='dot_product', q_len=1, dropout=0):
        super(NoQueryAttention, self).__init__(embed_dim, hidden_dim, out_dim, n_head, score_function, dropout)
        self.q_len = q_len
        self.q = nn.Parameter(torch.Tensor(q_len, embed_dim))
        self.reset_q()

    def reset_q(self):
        stdv = 1. / math.sqrt(self.embed_dim)
        self.q.data.uniform_(-stdv, stdv)

    def forward(self, k, **kwargs):
        mb_size = k.shape[0]
        q = self.q.expand(mb_size, -1, -1)
        return super(NoQueryAttention, self).forward(k, q)
