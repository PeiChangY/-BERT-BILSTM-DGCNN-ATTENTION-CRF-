import torch
from model.BERT_BiLSTM_CRF import BERT_BiLSTM_CRF
from scripts.config import Config
from scripts.utils import load_vocab
import os
'''用于识别输入的句子（可以换成批量输入）的命名实体
    <pad>   0
    B-PER   1
    I-PER   2
    B-LOC   3
    I-LOC   4
    B-ORG   5
    I-ORG   6
    O       7
    <START> 8
    <EOS>   9
'''
# <pad>   0
# B-PER   1
# I-PER   2
# B-ECO   3
# I-ECO   4
# B-MAG   5
# I-MAG   6
# B-PTN   7
# I-PTN   8
# B-MAK   9
# I-MAK   10
# B-PLR   11
# I-PLR   12
# B-ETF   13
# I-ETF   14
# B-PPR   15
# I-PPR   16
# B-INS   17
# I-INS   18
# O       19
# <START> 20
# <EOS>   21
# tags = [(1, 2), (3, 4), (5, 6)]
tags = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18)]
def predict(input_seq, max_length=128):
    '''
    :param input_seq: 输入一句话
    :return:
    '''
    config = Config()
    vocab = load_vocab(config.vocab)  #导入bert词典
    label_dic = load_vocab(config.label_file)  #导入tag标签
    tagset_size = len(label_dic)  #计算标签个数
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = BERT_BiLSTM_CRF(tagset_size,
                            config.bert_embedding,
                            config.rnn_hidden,
                            config.rnn_layer,
                            # config.dropout_ratio,
                            config.dropout,
                            config.pretrain_model_name,
                            device).to(device)

    # checkpoint = torch.load(os.path.join(config.target_dir, "RoBERTa_best.pth.tar"))
    checkpoint = torch.load(os.path.join(config.target_dir_yang, "RoBERTa_best.pth.tar"))
    model.load_state_dict(checkpoint["model"])

    # 构造输入
    input_list = []
    for i in range(len(input_seq)):
        input_list.append(input_seq[i])   #append，引用追加对象的地址，把列表连接起来  可以追加列表对象  比如[1, 2, 3, [‘a’, ‘b’, ‘c’]]

    #当输入句子长度大于最大序列长度减2（126），则把句子中前126个字作为新的输入
    if len(input_list) > max_length - 2:
        input_list = input_list[0:(max_length - 2)]
    input_list = ['[CLS]'] + input_list + ['[SEP]']

    input_ids = [int(vocab[word]) if word in vocab else int(vocab['[UNK]']) for word in input_list] #以列表的形式得到输入的字符在BERT字典里的位置
    input_mask = [1] * len(input_ids)    #以列表的形式得到输入的字符在BERT字典能找到的个数

    if len(input_ids) < max_length:
        input_ids.extend([0] * (max_length - len(input_ids)))   #extend通过复制第二个数组的值，把数组进行拼接
        input_mask.extend([0] * (max_length - len(input_mask)))
    assert len(input_ids) == max_length
    assert len(input_mask) == max_length

    # 变为tensor并放到GPU上, 二维, 这里mask在CRF中必须为unit8类型或者bool类型
    input_ids = torch.LongTensor([input_ids]).to(device)
    input_mask = torch.ByteTensor([input_mask]).to(device)

    feats = model(input_ids, input_mask)
    # out_path是一条预测路径（数字列表）, [1:-1]表示去掉一头一尾, <START>和<EOS>标志
    out_path = model.predict(feats, input_mask)[0][1:-1]
    res = find_all_tag(out_path)

    # PER = []
    # LOC = []
    # ORG = []

    # for name in res:
    #     if name == 1:
    #         for i in res[name]:
    #             PER.append(input_seq[i[0]:(i[0]+i[1])])
    #     if name == 2:
    #         for j in res[name]:
    #             LOC.append(input_seq[j[0]:(j[0]+j[1])])
    #     if name == 3:
    #         for k in res[name]:
    #             ORG.append(input_seq[k[0]:(k[0]+k[1])])

    # 输出结果
    # print('预测结果:', '\n', 'PER:', PER, '\n', 'ORG:', ORG, '\n', 'LOC:', LOC)

    PER = []
    ECO = []
    MAG = []
    PTN = []
    MAK = []
    PLR = []
    ETF = []
    PPR = []
    INS = []
    for name in res:
        if name == 1:
            for i in res[name]:
                PER.append(input_seq[i[0]:(i[0]+i[1])])
        if name == 2:
            for j in res[name]:
                ECO.append(input_seq[j[0]:(j[0]+j[1])])
        if name == 3:
            for k in res[name]:
                MAG.append(input_seq[k[0]:(k[0]+k[1])])
        if name == 4:
            for k in res[name]:
                PTN.append(input_seq[k[0]:(k[0]+k[1])])
        if name == 5:
            for k in res[name]:
                MAK.append(input_seq[k[0]:(k[0]+k[1])])
        if name == 6:
            for k in res[name]:
                PLR.append(input_seq[k[0]:(k[0]+k[1])])
        if name == 7:
            for k in res[name]:
                ETF.append(input_seq[k[0]:(k[0]+k[1])])
        if name == 8:
            for k in res[name]:
                PPR.append(input_seq[k[0]:(k[0]+k[1])])
        if name == 9:
            for k in res[name]:
                INS.append(input_seq[k[0]:(k[0]+k[1])])

    # 输出结果
    print('预测结果:', '\n', 'PER:', PER, '\n', 'ECO:', ECO, '\n', 'MAG:', MAG , '\n', 'PTN:', PTN , '\n', 'MAK:', MAK , '\n', 'PLR:', PLR
           , '\n', 'ETF:', ETF , '\n', 'PPR:', PPR , '\n', 'INS:', INS)


def find_tag(out_path, B_label_id=1, I_label_id=2):
    '''
    找到指定的label
    :param out_path: 模型预测输出的路径 shape = [1, rel_seq_len]
    :param B_label_id:
    :param I_label_id:
    :return:
    '''
    sentence_tag = []
    for num in range(len(out_path)):
        if out_path[num] == B_label_id:
            start_pos = num
        if out_path[num] == I_label_id and out_path[num-1] == B_label_id:
            length = 2
            for num2 in range(num, len(out_path)):
                if out_path[num2] == I_label_id and out_path[num2-1] == I_label_id:
                    length += 1
                    if num2 == len(out_path)-1:  # 如果已经到达了句子末尾
                        sentence_tag.append((start_pos, length))
                        return sentence_tag
                if out_path[num2] == 19:  #7改为19
                    sentence_tag.append((start_pos, length))
                    break
    return sentence_tag

def find_all_tag(out_path):
    num = 1  # 1: PER、 2: LOC、3: ORG
    result = {}
    for tag in tags:
        res = find_tag(out_path, B_label_id=tag[0], I_label_id=tag[1])
        result[num] = res
        num += 1
    return result

if __name__ == "__main__":
    while True:
        input_seq = input("输入:")
        predict(input_seq)


