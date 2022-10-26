import torch.nn as nn
import torch
from transformers import BertModel
from torchcrf import CRF
from model.attention import Attention
from model.agcn import GraphConvolution
import copy
from model.cnn import IDCNN
class BERT_BiLSTM_CRF(nn.Module):
    def __init__(self, tagset_size, embedding_dim, hidden_dim, rnn_layers, filters_number, dropout, pretrain_model_name, device):
        '''
        the model of BERT_BiLSTM_CRF
        :param bert_config:
        :param tagset_size:
        :param embedding_dim:
        :param hidden_dim:
        :param rnn_layers:
        :param lstm_dropout:
        :param dropout:
        :param use_cuda:
        :return:
        '''
        super(BERT_BiLSTM_CRF, self).__init__()
        self.tagset_size = tagset_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.rnn_layers = rnn_layers
        # self.num_gcn_layers = num_gcn_layers
        # gcn_layer = GraphConvolution(self.embedding_dim, self.embedding_dim)
        # self.gcn_layer = nn.ModuleList([copy.deepcopy(gcn_layer) for _ in range(self.num_gcn_layers)])
        self.dropout = dropout
        self.device = device
        self.word_embeds = BertModel.from_pretrained(pretrain_model_name)
        for param in self.word_embeds.parameters():
            param.requires_grad = True
        self.filters_number = filters_number
        self.idcnn = IDCNN(input_size=self.embedding_dim, filters=self.filters_number) #bilstm/idcnn  idcnn-bilstm
        # self.idcnn = IDCNN(input_size=self.hidden_dim * 2, filters=self.filters_number)  # bilstm-idcnn
        self.LSTM = nn.LSTM(self.embedding_dim,
                            self.hidden_dim,
                            num_layers=self.rnn_layers,
                            bidirectional=True,
                            batch_first=True)#bilstm/idcnn   bilstm-idcnn
        # self.LSTM = nn.LSTM(self.filters_number,
        #                     self.hidden_dim,
        #                     num_layers=self.rnn_layers,
        #                     bidirectional=True,
        #                     batch_first=True) # idcnn -bilstm
        # self.GRU = nn.GRU(self.embedding_dim,
        #                     self.hidden_dim,
        #                     num_layers=self.rnn_layers,
        #                     bidirectional=True,
        #                     batch_first=True)
        self._dropout = nn.Dropout(p=self.dropout)
        self.CRF = CRF(num_tags=self.tagset_size, batch_first=True)
        self.Liner = nn.Linear(self.hidden_dim*2, self.tagset_size)
        self.Liner_1 = nn.Linear(self.hidden_dim*6, self.tagset_size)
        self.Liner_2 = nn.Linear(self.filters_number,self.tagset_size)
        self.Liner_3 = nn.Linear(self.tagset_size*2,self.tagset_size) #拼接
        self.attention=Attention(embed_dim=self.tagset_size,hidden_dim=128,out_dim=self.tagset_size)  # embed_dim 和 out_dim为标签长度
    def _init_hidden(self, batch_size):
        """
        random initialize hidden variable
        """
        return (torch.randn(2*self.rnn_layers, batch_size, self.hidden_dim).to(self.device), \
                torch.randn(2*self.rnn_layers, batch_size, self.hidden_dim).to(self.device))


    def forward(self, sentence, attention_mask=None):
        '''
        :param sentence: sentence (batch_size, max_seq_len) : word-level representation of sentence
        :param attention_mask:
        :return: List of list containing the best tag sequence for each batch.
        '''
        batch_size = sentence.size(0)  #16
        seq_length = sentence.size(1)#128
        # embeds: [batch_size, max_seq_length, embedding_dim]
        embeds = self.word_embeds(sentence, attention_mask=attention_mask).last_hidden_state  # 16 128 768    sentence维度为16  128
        # embeds_1 = self.Liner_1(embeds) #bert+crf
        # return embeds  #bert+crf

        # idcnn-bilstm
        # idcnn_out = self.idcnn(embeds, seq_length)
        # hidden = self._init_hidden(batch_size)
        # lstm_out, hidden = self.LSTM(idcnn_out, hidden)
        # lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim*2)    # 2048  256
        # d_lstm_out = self._dropout(lstm_out)  # 2048  256
        # l_out = self.Liner(d_lstm_out)  # 2048  22
        # lstm_feats = l_out.contiguous().view(batch_size, seq_length, -1)   # 16  128 22
        # lstm_feats,_= self.attention(lstm_feats,lstm_feats)   #16  128  22
        # return lstm_feats

        # bilstm-idcnn
        # hidden = self._init_hidden(batch_size)
        # # lstm_out: [batch_size, max_seq_length, hidden_dim*2]
        # lstm_out, hidden = self.LSTM(embeds, hidden)  # 16  128  256
        # idcnn_out = self.idcnn(lstm_out,seq_length)  # BILSTM+DGCNN  16  128  120
        # idcnn_out = idcnn_out.contiguous().view(-1, self.filters_number)
        # d_idcnn_out = self._dropout(idcnn_out)
        # i_out = self.Liner_2(d_idcnn_out)  # BILSTM+DGCNN
        # idcnn_out = i_out.contiguous().view(batch_size, seq_length, -1)
        # idcnn_out, _ = self.attention(idcnn_out, idcnn_out)  # 16  128  22
        # return idcnn_out

        #bilstm/idcnn
        idcnn_out = self.idcnn(embeds,seq_length) #idcnn   torch.Size([16, 128, 64])
        idcnn_out = self._dropout(idcnn_out)
        idcnn_out = self.Liner_2(idcnn_out) # bilstm/idcnn
        hidden = self._init_hidden(batch_size)
        # lstm_out: [batch_size, max_seq_length, hidden_dim*2]
        lstm_out, hidden = self.LSTM(embeds, hidden)  # 16  128  256
        # lstm_out, hidden = self.GRU(embeds, None)   #GRU
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim*2)    # 2048  256
        d_lstm_out = self._dropout(lstm_out)  # 2048  256
        l_out = self.Liner(d_lstm_out)  # 2048  22
        lstm_feats = l_out.contiguous().view(batch_size, seq_length, -1)   # 16  128 22
        # lstm_feats,_= self.attention(lstm_feats,idcnn_out)   #16  128  22
        # lstm_feats = torch.cat((lstm_feats,idcnn_out),2)  # 拼接
        # lstm_feats = self.Liner_3(lstm_feats) #拼接
        # lstm_feats = lstm_feats + idcnn_out # 相加
        return lstm_feats

    def loss(self, feats, tags, mask):
        ''' 做训练时用
        :param feats: the output of BiLSTM and Liner
        :param tags:
        :param mask:
        :return:
        '''
        loss_value = self.CRF(emissions=feats,
                              tags=tags,
                              mask=mask,
                              reduction='mean')
        return -loss_value

    def predict(self, feats, attention_mask):
        # 做验证和测试时用
        out_path = self.CRF.decode(emissions=feats, mask=attention_mask)
        return out_path

    # def get_attention(self, val_out, dep_embed, adj):
    #     batch_size, max_len, feat_dim = val_out.shape
    #     val_us = val_out.unsqueeze(dim=2)
    #     val_us = val_us.repeat(1,1,max_len,1)
    #     val_cat = torch.cat((val_us, dep_embed), -1)
    #     atten_expand = (val_cat.float() * val_cat.float().transpose(1,2))
    #     attention_score = torch.sum(atten_expand, dim=-1)
    #     attention_score = attention_score / feat_dim ** 0.5
    #     # softmax
    #     exp_attention_score = torch.exp(attention_score)
    #     exp_attention_score = torch.mul(exp_attention_score.float(), adj.float())
    #     sum_attention_score = torch.sum(exp_attention_score, dim=-1).unsqueeze(dim=-1).repeat(1,1,max_len)
    #     attention_score = torch.div(exp_attention_score, sum_attention_score + 1e-10)
    #     return attention_score

