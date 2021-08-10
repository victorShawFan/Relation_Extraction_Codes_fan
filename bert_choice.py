import json
import psutil
from tqdm import tqdm, trange


import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertForMultipleChoice, AdamW, get_linear_schedule_with_warmup
from transformers import BertTokenizer

train = json.load(open('/mnt/xfan/re_datasets/c3_public/d-train.json')) + json.load(
    open('/mnt/xfan/re_datasets/c3_public/m-train.json'))
val = json.load(open('/mnt/xfan/re_datasets/c3_public/m-dev.json')) + json.load(
    open('/mnt/xfan/re_datasets/c3_public/d-dev.json'))

train_label = [x[1][0]['choice'].index(x[1][0]['answer']) for x in train]
val_label = [x[1][0]['choice'].index(x[1][0]['answer']) for x in val]

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', num_choices=4)
# train_encoding = tokenizer(list(train_lines), truncation=True, padding=True, max_length=64)
# val_encoding = tokenizer(list(val_lines), truncation=True, padding=True, max_length=64)

def collate_fn(data):  # 将文章问题选项拼在一起后，得到分词后的数字id，输出的size是(batch, n_choices, max_len)
    input_ids, attention_mask, token_type_ids = [], [], []
    for x in data:
        text = tokenizer(x[1], text_pair=x[0], padding='max_length', truncation=True,
                         max_length=128, return_tensors='pt')
        input_ids.append(text['input_ids'].tolist())
        attention_mask.append(text['attention_mask'].tolist())
        token_type_ids.append(text['token_type_ids'].tolist())
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    token_type_ids = torch.tensor(token_type_ids)
    label = torch.tensor([x[-1] for x in data])
    return input_ids, attention_mask, token_type_ids, label


class TextDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, idx):
        label = self.labels[idx]
        question = self.data[idx][1][0]['question']
        content = '。'.join(self.data[idx][0])
        choice = self.data[idx][1][0]['choice']
        if len(choice) < 4:  # 如果选项不满四个，就补“不知道”
            for i in range(4 - len(choice)):
                choice.append('不知道')

        content = [content for i in range(len(choice))]
        pair = [question + ' ' + i for i in choice]

        return content, pair, label

    def __len__(self):
        return len(self.labels)


train_dataset = TextDataset(train, train_label)
test_dataset = TextDataset(val, val_label)

model = BertForMultipleChoice.from_pretrained('bert-base-chinese')

# device = 'cpu'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

cpu_nums = psutil.cpu_count()
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=cpu_nums)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=cpu_nums)

optim = AdamW(model.parameters(), lr=5e-5)
total_steps = len(train_loader) * 1
scheduler = get_linear_schedule_with_warmup(optim,
                                            num_warmup_steps=0,  # Default value in run_glue.py
                                            num_training_steps=total_steps)


def train():
    model.train()
    total_train_loss = 0
    iter_num = 0
    total_iter = len(train_loader)
    for idx, (input_ids, attention_mask, token_type_ids, labels) in enumerate(train_loader):
        optim.zero_grad()

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        # print(labels)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        # loss = outputs[0]

        # print(outputs[1])
        loss = outputs.loss

        if idx % 10 == 0:
            with torch.no_grad():
                # 64 * 7
                print((outputs[1].argmax(1).data == labels.data).float().mean().item(), loss.item())

        total_train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        scheduler.step()

        iter_num += 1
        if (iter_num % 100 == 0):
            print("epoth: %d, iter_num: %d, loss: %.4f, %.2f%%" % (
                epoch, iter_num, loss.item(), iter_num / total_iter * 100))

    print("Epoch: %d, Average training loss: %.4f" % (epoch, total_train_loss / len(train_loader)))


def validation():
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    for (input_ids, attention_mask, token_type_ids, labels) in test_dataloader:
        with torch.no_grad():
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs[1]

        total_eval_loss += loss.item()
        logits = logits.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()
        total_eval_accuracy += (outputs[1].argmax(1).data == labels.data).float().mean().item()

    avg_val_accuracy = total_eval_accuracy / len(test_dataloader)
    print("Accuracy: %.4f" % (avg_val_accuracy))
    print("Average testing loss: %.4f" % (total_eval_loss / len(test_dataloader)))
    print("-------------------------------")


for epoch in trange(4):
    print("------------Epoch: %d ----------------" % epoch)
    validation()
    train()
