# -*- coding: utf-8 -*-

# tags
l2i_dic = {"O": 0, u'B-LOC': 1, u'B-PER': 2, u'B-ORG': 3, u'I-LOC': 4, u'I-PER': 5, u'I-ORG': 6, u'E-LOC': 7, u'E-PER': 8, u'E-ORG': 9, "<pad>": 10, "<start>": 11, "<eos>": 12}
#
i2l_dic = {0:"O",1:u'B-LOC', 2:u'B-PER', 3:u'B-ORG',4:u'I-LOC',5:u'I-PER',6:u'I-ORG', 7: u'E-LOC', 8: u'E-PER' , 9: u'E-ORG', 10: "<pad>", 11: "<start>", 12: "<eos>"}


train_file = './data/train.txt'
dev_file = './data/test.txt'
vocab_file = './data/bert/vocab.txt'

save_model_dir =  './data/model/idcnn_lstm_'
bert_model_dir = './data/bert/'

model_path = './data/model/idcnn_lstm_1.pkl'

max_length = 256
batch_size = 8
epochs = 5
tagset_size = len(l2i_dic)
use_cuda = True

