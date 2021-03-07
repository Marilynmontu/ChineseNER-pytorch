# -*- coding: utf-8 -*-

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from utils import load_vocab, load_data, recover_label, get_ner_fmeasure, save_model, load_model
from constants import *
from model import BERT_LSTM_CRF


if torch.cuda.is_available():
    device = torch.device("cuda", 0)
    print('device',device)
    use_cuda = True
else:
    device = torch.device("cpu")
    use_cuda = False
vocab = load_vocab(vocab_file)
####读取训练集
print('max_length',max_length)
train_data = load_data(train_file, max_length=max_length, label_dic=l2i_dic, vocab=vocab)

train_ids = torch.LongTensor([temp.input_id for temp in train_data])
train_masks = torch.LongTensor([temp.input_mask for temp in train_data])
train_tags = torch.LongTensor([temp.label_id for temp in train_data])

train_dataset = TensorDataset(train_ids, train_masks, train_tags)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

#######读取测试集
dev_data = load_data(dev_file, max_length=max_length, label_dic=l2i_dic, vocab=vocab)

dev_ids = torch.LongTensor([temp.input_id for temp in dev_data])
dev_masks = torch.LongTensor([temp.input_mask for temp in dev_data])
dev_tags = torch.LongTensor([temp.label_id for temp in dev_data])

dev_dataset = TensorDataset(dev_ids, dev_masks, dev_tags)
dev_loader = DataLoader(dev_dataset, shuffle=True, batch_size=batch_size)

######测试函数
def evaluate(medel, dev_loader):
    medel.eval()
    pred = []
    gold = []
    print('evaluate')
    for i, dev_batch in enumerate(dev_loader):
        sentence, masks, tags = dev_batch
        sentence, masks, tags = Variable(sentence), Variable(masks), Variable(tags)
        if use_cuda:
            sentence = sentence.cuda()
            masks = masks.cuda()
            tags = tags.cuda()
        predict_tags = medel(sentence, masks)
        pred.extend([t for t in predict_tags.tolist()])
        gold.extend([t for t in tags.tolist()])
    pred_label,gold_label = recover_label(pred, gold, l2i_dic,i2l_dic)
    acc, p, r, f = get_ner_fmeasure(gold_label,pred_label)
    print('p: {}，r: {}, f: {}'.format(p, r, f))
    model.train()
    return acc, p, r, f

########加载模型
model = BERT_LSTM_CRF(bert_model_dir, tagset_size, 768, 200, 1,
                      dropout_ratio=0.4, dropout1=0.4, use_cuda = use_cuda)
if use_cuda:
    model = model.cuda()
    model.load_state_dict(torch.load(model_path))
model.train()
optimizer = getattr(optim, 'Adam')
optimizer = optimizer(model.parameters(), lr=0.00001, weight_decay=0.00005)

best_f = -100

for epoch in range(epochs):
    print('epoch: {}，train'.format(epoch))
    for i, train_batch in enumerate(tqdm(train_loader)):
        model.zero_grad()
        sentence, masks, tags = train_batch
        sentence, masks, tags = Variable(sentence), Variable(masks), Variable(tags)
        if use_cuda:
            sentence = sentence.cuda()
            masks = masks.cuda()
            tags = tags.cuda()
        loss = model.neg_log_likelihood_loss(sentence, masks, tags)
        loss.backward()
        optimizer.step()
    print('epoch: {}，loss: {}'.format(epoch, loss.item()))
    acc, p, r, f = evaluate(model,dev_loader)
    if f > best_f:
        model_name = model_name = save_model_dir + str(epoch) + ".pkl"
        torch.save(model.state_dict(), model_name)
        best_f = f













