# -*- coding: utf-8 -*-

import torch.nn as nn
from pytorch_pretrained_bert import BertModel
from model import CRF
from torch.autograd import Variable
from model.cnn import IDCNN 
import torch

class BERT_LSTM_CRF(nn.Module):
    def __init__(self, bert_config, tagset_size, embedding_dim, hidden_dim, rnn_layers, dropout_ratio, dropout1, use_cuda):
        super(BERT_LSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.word_embeds = BertModel.from_pretrained(bert_config)

        self.idcnn = IDCNN(input_size=embedding_dim, filters=64)
        self.linear = nn.Linear(64, 256)

        self.lstm = nn.LSTM(64, hidden_dim,
                            num_layers=rnn_layers, bidirectional=True, dropout=dropout_ratio, batch_first=True)
        self.rnn_layers = rnn_layers
        self.dropout1 = nn.Dropout(p=dropout1)
        self.crf = CRF(target_size=tagset_size, average_batch=True, use_cuda=use_cuda)
        self.liner = nn.Linear(hidden_dim*2, tagset_size+2)
        self.tagset_size = tagset_size
        self.use_cuda =  use_cuda

    def rand_init_hidden(self, batch_size):
        if self.use_cuda:
            return Variable(
                torch.randn(2 * self.rnn_layers, batch_size, self.hidden_dim)).cuda(), Variable(
                torch.randn(2 * self.rnn_layers, batch_size, self.hidden_dim)).cuda()
        else:
            return Variable(
                torch.randn(2 * self.rnn_layers, batch_size, self.hidden_dim)), Variable(
                torch.randn(2 * self.rnn_layers, batch_size, self.hidden_dim))

    def get_output_score(self, sentence, attention_mask=None):
        batch_size = sentence.size(0)
        seq_length = sentence.size(1)
        embeds, _ = self.word_embeds(sentence, attention_mask=attention_mask, output_all_encoded_layers=False)

        out = self.idcnn(embeds, seq_length)
        # out = self.linear(out)
        hidden = self.rand_init_hidden(batch_size)
        # if embeds.is_cuda:
        #     hidden = (i.cuda() for i in hidden)
        lstm_out, hidden = self.lstm(out, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim * 2)
        d_lstm_out = self.dropout1(lstm_out)
        l_out = self.liner(d_lstm_out)
        lstm_feats = l_out.contiguous().view(batch_size, seq_length, -1)
        return lstm_feats

    def forward(self, sentence, masks):
        lstm_feats = self.get_output_score(sentence)
        scores, tag_seq = self.crf._viterbi_decode(lstm_feats, masks.byte())
        return tag_seq

    def neg_log_likelihood_loss(self, sentence, mask, tags):
        lstm_feats = self.get_output_score(sentence)
        loss_value = self.crf.neg_log_likelihood_loss(lstm_feats, mask, tags)
        batch_size = lstm_feats.size(0)
        loss_value /= float(batch_size)
        return loss_value




