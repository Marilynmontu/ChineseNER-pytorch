import sys
import torch
from model import BERT_LSTM_CRF
from constants import *
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils import load_vocab, load_data, recover_label, get_ner_BMES

model_file = "idcnn_1e5_3.pkl"

input = sys.argv[1]
f = open("./data/input.txt", "w")
for each_word in list(input):
    f.write(each_word + " O\n")
f.write("\n")
f.close()

#model = torch.load(model_file)
# 用参数文件也可以运行，取消下面两条注释，注释上面一行。
model = BERT_LSTM_CRF(bert_model_dir, tagset_size, 768, 200, 1,
                       dropout_ratio=0.4, dropout1=0.4, use_cuda=False)
model.load_state_dict(torch.load(model_file, map_location='cpu'))
model.eval()

vocab = load_vocab(vocab_file)
dev_data = load_data("./data/input.txt", max_length=max_length, label_dic=l2i_dic, vocab=vocab)
dev_ids = torch.LongTensor([temp.input_id for temp in dev_data])
dev_masks = torch.LongTensor([temp.input_mask for temp in dev_data])
dev_tags = torch.LongTensor([temp.label_id for temp in dev_data])
dev_dataset = TensorDataset(dev_ids, dev_masks, dev_tags)
dev_loader = DataLoader(dev_dataset, shuffle=True, batch_size=batch_size)

pred = []
gold = []

for i, dev_batch in enumerate(dev_loader):
    sentence, masks, tags = dev_batch
    sentence, masks, tags = Variable(sentence), Variable(masks), Variable(tags)
    predict_tags = model(sentence, masks)
    break
try:
    pred.extend([t for t in predict_tags.tolist()])
    gold.extend([t for t in tags.tolist()])
    pred_label, gold_label = recover_label(pred, gold, l2i_dic, i2l_dic)
    pred_label = pred_label[0][1:]
except:
    print("Input Error.")
    sys.exit(-1)

in_file = open("./data/input.txt")
out_file = open("./data/output.txt", "w+")
lines = in_file.readlines()
new_lines = []
for index in range(len(lines)):
    if len(lines[index]) > 2:
        new_lines.append(lines[index][:2] + pred_label[index] + "\n")
out_file.writelines(new_lines)
in_file.close()
out_file.close()

# print(get_ner_BMES(pred_label))

all = []
index = 0
while index != len(new_lines):
    if new_lines[index][2] == "B":
        tag = new_lines[index][4:7]
        # print(tag)
        for j in range(index + 1, len(new_lines)):
            if new_lines[j][2] == "I":
                pass
            elif new_lines[j][2] == "O" or new_lines[j][2] == "B":
                index = j + 1
                break
            else:
                # print(str([index, j]))
                # print(input[index:j+1])
                all.append((tag, input[index:j + 1]))
                index = j + 1
                break
    else:
        index = index + 1
print(all)
print("done")
