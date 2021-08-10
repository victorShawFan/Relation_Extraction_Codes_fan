# 1 导入必要的库
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm
# from pytorch_transformers import BertModel, BertConfig, BertTokenizer
from transformers import (AdamW, BertConfig, BertModel, BertTokenizer,
                          get_cosine_schedule_with_warmup)

warnings.filterwarnings('ignore')
print('current file path : ', os.getcwd())
print(
    f'gpu info : cuda_available: {torch.cuda.is_available()}, gpu nums : {torch.cuda.device_count()}, names : {torch.cuda.get_device_name(0)}'
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bert_path = "./bert_model/"  # 该文件夹下存放三个文件（'vocab.txt', 'pytorch_model.bin', 'config.json'）
tokenizer = BertTokenizer.from_pretrained(bert_path,
                                          local_files_only=True)  # 初始化分词器
'''# 2 预处理数据集'''
input_ids, input_masks, input_types, = [], [], []  # input char ids, segment type ids,  attention mask
labels = []  # 标签
maxlen = 30  # 取30即可覆盖99%

with open("./news_title_dataset.csv", encoding='utf-8') as f:
    for i, line in tqdm(enumerate(f)):
        title, y = line.strip().split('\t')

        # encode_plus会输出一个字典，分别为'input_ids', 'token_type_ids', 'attention_mask'对应的编码
        # 根据参数会短则补齐，长则切断
        encode_dict = tokenizer.encode_plus(text=title,
                                            max_length=maxlen,
                                            padding='max_length',
                                            truncation=True)

        input_ids.append(encode_dict['input_ids'])
        input_types.append(encode_dict['token_type_ids'])
        input_masks.append(encode_dict['attention_mask'])

        labels.append(int(y))

input_ids, input_types, input_masks = np.array(input_ids), np.array(
    input_types), np.array(input_masks)
labels = np.array(labels)
print(input_ids.shape, input_types.shape, input_masks.shape, labels.shape)
'''# 3 切分训练集、验证集和测试集'''
# 随机打乱索引
idxes = np.arange(input_ids.shape[0])
np.random.seed(2019)  # 固定种子
np.random.shuffle(idxes)
print(idxes.shape, idxes[:10])

# 8:1:1 划分训练集、验证集、测试集
input_ids_train, input_ids_valid, input_ids_test = input_ids[
    idxes[:80000]], input_ids[idxes[80000:90000]], input_ids[idxes[90000:]]
input_masks_train, input_masks_valid, input_masks_test = input_masks[
    idxes[:80000]], input_masks[idxes[80000:90000]], input_masks[idxes[90000:]]
input_types_train, input_types_valid, input_types_test = input_types[
    idxes[:80000]], input_types[idxes[80000:90000]], input_types[idxes[90000:]]

y_train, y_valid, y_test = labels[idxes[:80000]], labels[
    idxes[80000:90000]], labels[idxes[90000:]]

print(input_ids_train.shape, y_train.shape, input_ids_valid.shape,
      y_valid.shape, input_ids_test.shape, y_test.shape)
'''# 4加载到高效的DataLoader'''
BATCH_SIZE = 64  # 如果会出现OOM问题，减小它
# 训练集
train_data = TensorDataset(torch.LongTensor(input_ids_train),
                           torch.LongTensor(input_masks_train),
                           torch.LongTensor(input_types_train),
                           torch.LongTensor(y_train))
train_sampler = RandomSampler(train_data)
train_loader = DataLoader(train_data,
                          sampler=train_sampler,
                          batch_size=BATCH_SIZE)

# 验证集
valid_data = TensorDataset(torch.LongTensor(input_ids_valid),
                           torch.LongTensor(input_masks_valid),
                           torch.LongTensor(input_types_valid),
                           torch.LongTensor(y_valid))
valid_sampler = SequentialSampler(valid_data)
valid_loader = DataLoader(valid_data,
                          sampler=valid_sampler,
                          batch_size=BATCH_SIZE)

# 测试集（是没有标签的）
test_data = TensorDataset(torch.LongTensor(input_ids_test),
                          torch.LongTensor(input_masks_test),
                          torch.LongTensor(input_types_test))
test_sampler = SequentialSampler(test_data)
test_loader = DataLoader(test_data,
                         sampler=test_sampler,
                         batch_size=BATCH_SIZE)
'''5 定义bert模型'''


# 定义model
class Bert_Model(nn.Module):
    def __init__(self, bert_path, classes=10):
        super(Bert_Model, self).__init__()
        self.config = BertConfig.from_pretrained(bert_path)  # 导入模型超参数
        self.bert = BertModel.from_pretrained(bert_path)  # 加载预训练模型权重
        self.fc = nn.Linear(self.config.hidden_size, classes)  # 直接分类

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        out_pool = outputs[1]  # 池化后的输出 [bs, config.hidden_size]
        logit = self.fc(out_pool)  #  [bs, classes]
        return logit


# 可以发现，bert模型的定义由于高效简易的封装库存在，使得定义模型较为容易，如果想要在bert之后加入cnn/lstm等层，可在这里定义。
'''6 实例化bert模型'''


def get_parameter_number(model):
    #  打印模型参数量
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters()
                        if p.requires_grad)
    return 'Total parameters: {}, Trainable parameters: {}'.format(
        total_num, trainable_num)


EPOCHS = 5
model = Bert_Model(bert_path).to(DEVICE)
print(get_parameter_number(model))
'''7 定义优化器'''

optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)  #AdamW优化器
scheduler = get_cosine_schedule_with_warmup(optimizer,
                                            num_warmup_steps=len(train_loader),
                                            num_training_steps=EPOCHS *
                                            len(train_loader))
# 学习率先线性warmup一个epoch，然后cosine式下降。
# 这里给个小提示，一定要加warmup（学习率从0慢慢升上去），要不然你把它去掉试试，基本上收敛不了。
'''8 定义训练函数和验证测试函数'''


# 评估模型性能，在验证集上
def evaluate(model, data_loader, device):
    model.eval()
    val_true, val_pred = [], []
    with torch.no_grad():
        for idx, (ids, att, tpe, y) in (enumerate(data_loader)):
            y_pred = model(ids.to(device), att.to(device), tpe.to(device))
            y_pred = torch.argmax(y_pred,
                                  dim=1).detach().cpu().numpy().tolist()
            val_pred.extend(y_pred)
            val_true.extend(y.squeeze().cpu().numpy().tolist())

    return accuracy_score(val_true, val_pred)  #返回accuracy


# 测试集没有标签，需要预测提交
def predict(model, data_loader, device):
    model.eval()
    val_pred = []
    with torch.no_grad():
        for idx, (ids, att, tpe) in tqdm(enumerate(data_loader)):
            y_pred = model(ids.to(device), att.to(device), tpe.to(device))
            y_pred = torch.argmax(y_pred,
                                  dim=1).detach().cpu().numpy().tolist()
            val_pred.extend(y_pred)
    return val_pred


def train_and_eval(model, train_loader, valid_loader, optimizer, scheduler,
                   device, epoch):
    best_acc = 0.0
    patience = 0
    criterion = nn.CrossEntropyLoss()
    for i in range(epoch):
        """训练模型"""
        start = time.time()
        model.train()
        print("***** Running training epoch {} *****".format(i + 1))
        train_loss_sum = 0.0
        for idx, (ids, att, tpe, y) in enumerate(train_loader):
            ids, att, tpe, y = ids.to(device), att.to(device), tpe.to(
                device), y.to(device)
            y_pred = model(ids, att, tpe)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()  # 学习率变化

            train_loss_sum += loss.item()
            if (idx + 1) % (len(train_loader) // 5) == 0:  # 只打印五次结果
                print(
                    "Epoch {:04d} | Step {:04d}/{:04d} | Loss {:.4f} | Time {:.4f}"
                    .format(i + 1, idx + 1, len(train_loader),
                            train_loss_sum / (idx + 1),
                            time.time() - start))
                print("Learning rate = {}".format(
                    optimizer.state_dict()['param_groups'][0]['lr']))
        """验证模型"""
        model.eval()
        acc = evaluate(model, valid_loader, device)  # 验证模型的性能
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_bert_model.pth")

        print("current acc is {:.4f}, best acc is {:.4f}".format(
            acc, best_acc))
        print("time costed = {}s \n".format(round(time.time() - start, 5)))


'''9 开始训练和验证模型'''
# 训练和验证评估
if 'train' in sys.argv:
    train_and_eval(model, train_loader, valid_loader, optimizer, scheduler,
                   DEVICE, EPOCHS)
'''10 加载最优模型进行测试'''
# 加载最优权重对测试集测试
model.load_state_dict(torch.load("best_bert_model.pth"))
pred_test = predict(model, test_loader, DEVICE)
print("\n Test Accuracy = {} \n".format(accuracy_score(y_test, pred_test)))
print(classification_report(y_test, pred_test, digits=4))
