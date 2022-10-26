
class Config(object):
    '''配置类'''

    def __init__(self):
        # # weibo
        # self.label_file = '../dataset/weibo_data/weibo_data/tag.txt'
        # self.train_file = '../dataset/weibo_data/weibo_data/train.txt'
        # self.dev_file = '../dataset/weibo_data/weibo_data/dev.txt'
        # self.test_file = '../dataset/weibo_data/weibo_data/test.txt'

        # #msra
        self.label_file = '../dataset/msra_data/data_div/tag.txt'
        self.train_file = '../dataset/msra_data/data_div/trian.txt'
        self.dev_file = '../dataset/msra_data/data_div/dev.txt'
        self.test_file = '../dataset/msra_data/data_div/test.txt'

        # # resume
        # self.label_file = '../dataset/resume_data/resume_data/tag.txt'
        # self.train_file = '../dataset/resume_data/resume_data/train.txt'
        # self.dev_file = '../dataset/resume_data/resume_data/dev.txt'
        # self.test_file = '../dataset/resume_data/resume_data/test.txt'

        # self.label_file = '../dataset/data_yang/tag.txt'
        # self.train_file = '../dataset/data_yang/train.txt'
        # self.dev_file = '../dataset/data_yang/dev.txt'
        # self.test_file = '../dataset/data_yang/test.txt'

        self.vocab = '../dataset/bert/vocab.txt'
        self.max_length = 128
        self.use_cuda = True
        self.gpu = 0
        self.batch_size = 32
        self.rnn_hidden = 128
        self.bert_embedding = 768
        self.dropout = 0.5
        self.rnn_layer = 1
        self.num_gcn_layers = 1
        self.filters_number = 120  #卷积核个数
        self.lr = 0.00003
        self.lr_decay = 0.00001
        self.weight_decay = 0.00005
        self.checkpoint = None
        self.epochs = 64
        self.max_grad_norm = 10
        self.target_dir = '../result/RoBERTa_result'
        self.target_dir_yang = '../result'
        self.patience = 5
        # 可以换成RoBERTa的中文预训练模型（哈工大提供）
        self.pretrain_model_name = 'bert-base-chinese'
        # self.pretrain_model = [
        #     'albert-base-v2',                                   # ALBERT: （12-layer, 768-hidden, 12-heads, 11M parameters）
        #     'bert-base-chinese',                                # BERT: Google开源中文预训练模型（12-layer, 768-hidden, 12-heads, 110M parameters）
        #     'hfl/chinese-bert-wwm',                             # BERT: 中文wiki数据训练的Whole Word Mask版本（12-layer, 768-hidden, 12-heads, 110M parameters）
        #     'hfl/chinese-bert-wwm-ext',                         # BERT: 使用额外数据训练的Whole Word Mask版本（12-layer, 768-hidden, 12-heads, 110M parameters）
        #     'hfl/chinese-roberta-wwm-ext',                      # RoBERTa: 使用额外数据训练的Whole Word Mask版本（12-layer, 768-hidden, 12-heads, 110M parameters）
        #     'hfl/chinese-roberta-wwm-ext-large',                # RoBERTa: 使用额外数据训练的Whole Word Mask+Large 版本（24-layer, 1024-hidden, 16-heads, 330M parameters）
        #     'voidful/albert_chinese_base',                      # Albert: 非官方+base版（12layer）
        #     'voidful/albert_chinese_large',                     # Albert: 非官方+large版（24layer）
        #     'hfl/chinese-electra-base-discriminator',           # ELECTRA: 中文版+discriminator（12层，隐层768，12个注意力头，学习率2e-4，batch256，最大长度512，训练1M步）
        #     'hfl/chinese-electra-large-discriminator',          # ELECTRA: 中文版+discriminator+large （24层，隐层1024，16个注意力头，学习率1e-4，batch96，最大长度512，训练2M步）
        #     'hfl/chinese-electra-180g-base-discriminator',      # ELECTRA: 中文版+discriminator+大训练语料（12层，隐层768，12个注意力头，学习率2e-4，batch256，最大长度512，训练1M步）
        #     'hfl/chinese-electra-180g-large-discriminator',     # ELECTRA: 中文版+discriminator+大训练语料+large （24层，隐层1024，16个注意力头，学习率1e-4，batch96，最大长度512，训练2M步）
        # ]

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):
        return '\n'.join(['%s:%s' % item for item in self.__dict__.items()])


if __name__ == '__main__':
    con = Config()
    con.update(gpu=8)
    print(con.gpu)
    print(con)