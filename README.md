# BERT-Chinese-NER-pytorch
基于BERT+BiLSTM/DGCNN+Attention+CRF的中文命名实体识别 (pytorch实现)<br>

使用注意力机制融合BILSTM和DGCNN的输出，在公共数据集上表现优异，可以参考我们的论文《基于门控空洞卷积特征融合的中文命名实体识别》[https://chn.oversea.cnki.net/KCMS/detail/detail.aspx?sfield=fn&QueryID=0&CurRec=11&recid=&FileName=JSJC20221028000&DbName=CAPJLAST&DbCode=CAPJ&yx=Y&pr=&URLID=31.1289.TP.20221031.0907.001]
<hr>
基本环境：<br>
python 3.8 <br>
pytorch 1.7.1 + cu110 <br>
pytorch-crf 0.7.2 <br>
