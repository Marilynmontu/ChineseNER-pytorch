# BERT-IDCNN-BILSTM-CRF

基于Pytorch的BERT-IDCNN-BILSTM-CRF中文实体识别实现。

## 文件描述

- model/: 模型代码
  - bert_lstm_crf.py
  - cnn.py
  - crf.py
- data/: 数据集存放
  - train.txt: 训练集
  - test.txt: 测试集
- data/bert/: bert模型存放
  - bert_config.json: bert配置文件
  - pytorch_model.bin: bert中文预训练模型pytorch版（详情参考：https://github.com/maknotavailable/pytorch-pretrained-BERT）
  - vocab.txt: 词表文件
- constants.py: 模型配置：标注，数据集，最大长度，batch_size, epoch等
- train.py: 训练模型
- SaveModel.py: 从模型参数保存完整模型
- Wrapper.py: 执行单次NER
- utils.py: 数据处理相关

## constants.py

- bert_model_dir: bert目录，例如`data/bert`
- vocab_file: bert词表文件，例如`data/bert/vocab.txt`
- train_file: 训练集，例如`data/train.txt`
- dev_file: 测试集，例如`data/test.txt`
- model_path: 载入已有模型参数文件，指定文件名，例如`data/model/idcnn_lstm_1.pkl`
- save_model_dir: 模型保存文件路径及文件名前缀，例如`data/model/idcnn_lstm_`
- max_length: 最大句子长度
- batch_size: batch大小
- epochs: 训练轮数
- tagset_size: 标签数目
- use_cuda: 是否使用cuda

## 资源地址
- 数据集、训练好的BERT_IDCNN_LSTM_CRF模型文件以及中文版BERT预训练模型下载地址：链接:https://pan.baidu.com/s/196NCvVZhYyfZe8yo_-fIVw  密码:kw0e

## 模型训练（可选）
1. 下载pytorch_model.bin到data/bert
2. 下载训练集和测试集到data/
3. 检查配置constants.py
4. 执行train.py，命令为 `python train.py`

## 中文命名实体识别系统运行步骤

1. 已训练好的BERT_IDCNN_LSTM_CRF模型（如果有），下载到data/model
2. 检查配置constants.py
3. 单次运行系统，执行Wrapper.py，命令为 `Wrapper.py "新华网1950年10月1日电(中央人民广播电台记者刘振英、新华社记者张宿堂)中国科学院成立了。"`
4. 若想多次运行系统，则执行ChineseNer.sh，命令为`./ChineseNer.sh`

## 依赖

```
python >= 3.5
torch = 0.4.0
pytorch-pretrained-bert
tqdm
numpy
...
```

## 数据集示例

```
俄	B-ORG
罗	I-ORG
斯	I-ORG
国	I-ORG
家	I-ORG
杜	I-ORG
马	I-ORG
国	I-ORG
防	I-ORG
委	I-ORG
员	I-ORG
会	E-ORG
会	O
员	O
、	O
宇	O
航	O
员	O
萨	B-PER
维	I-PER
茨	I-PER
卡	I-PER
亚	E-PER
说	O
，	O
迄	O
今	O
为	O
止	O
，	O
俄	B-LOC
罗	I-LOC
斯	E-LOC
航	O
天	O
部	O
门	O
...
```

## 联系方式

若有疑问，可以通过以下方式联系：
- 邮箱：lianghui@iie.ac.cn



