import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from string import punctuation
from collections import Counter
import multiprocessing
from multiprocessing import freeze_support

# 定义网络模型结构
class Sentiment(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_dim, output_size, num_layers, dropout=0.5):
        super(Sentiment, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):
        '''
        x shape : (batch_size, seq_len, features)
        '''
        batch_size = x.size(0)
        x = x.long()
        embeds = self.embedding(x)
        out, hidden = self.lstm(embeds, hidden)
        out = out.reshape(-1, self.hidden_dim)
        out = self.linear(out)
        sigmoid_out = self.sigmoid(out)
        sigmoid_out = sigmoid_out.reshape(batch_size, -1)
        sigmoid_out = sigmoid_out[:, -1]
        return sigmoid_out, hidden

    def init_hidden(self, batch_size, device):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.num_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.num_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden

# 重置文本长度的函数
def reset_text(text, seq_len):
    dataset = np.zeros((len(text), seq_len))
    for index, sentence in enumerate(text):
        if len(sentence) < seq_len:
            dataset[index, :len(sentence)] = sentence
        else:
            dataset[index, :] = sentence[:seq_len]
    return dataset

# 文本预处理函数
def converts(text, word_int):
    # 去除标点符号
    new_text = ''.join([char for char in text if char not in punctuation])
    print("new text :\n", new_text)
    # 文本映射为索引
    text_ints = [word_int[word.lower()] for word in new_text.split()]
    print("文本映射为索引：\n", text_ints)
    return text_ints

# 训练模型函数
def train_model(model, device, data_loader, criterion, optimizer, num_epochs, val_loader, batch_size):
    history = list()
    for epoch in range(num_epochs):
        hs = model.init_hidden(batch_size, device)
        train_loss = []
        train_cor = 0.0
        model.train()
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output, hs = model(data, hs)
            hs = tuple([h.data for h in hs])
            loss = criterion(output, target.float())
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            train_cor += torch.sum(output==target)
        model.eval()
        hs = model.init_hidden(batch_size, device)
        val_loss = []
        val_cor = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(device)
                target = target.to(device)
                preds, hs = model(data, hs)
                hs = tuple([h.data for h in hs])
                loss = criterion(preds, target.float())
                val_loss.append(loss.item())
                val_cor += torch.sum(preds==target).item()

        print(
            f'Epoch {epoch}/{num_epochs} --- train loss {np.round(np.mean(train_loss), 5)} --- val loss {np.round(np.mean(val_loss), 5)}')

# 测试模型函数
def test_model(model, data_loader, device, criterion, batch_size):
    test_loss = []
    num_cor = 0.0
    hs = model.init_hidden(batch_size, device)
    model.eval()
    for i, dataset in enumerate(data_loader):
        data = dataset[0].to(device)
        target = dataset[1].to(device)
        output, hs = model(data, hs)
        loss = criterion(output, target.float())
        pred = torch.round(output)
        test_loss.append(loss.item())
        cor_tensor = pred.eq(target.float().view_as(pred))
        cor = cor_tensor.cpu().numpy()
        result = np.sum(cor)
        num_cor += result
        print(f'Batch {i}')
        print(f'loss : {np.round(np.mean(loss.item()), 3)}')
        print(f'accuracy : {np.round(result / len(data), 3) * 100} %')
        print()
    print("总的测试损失 test loss : {:.2f}".format(np.mean(test_loss)))
    print("总的测试准确率 test accuracy : {:.2f}".format(np.mean(num_cor / len(data_loader.dataset))))

# 预测函数
def predict(model, text_tensor, device):
    batch_size = text_tensor.size(0) # 这里是1
    hs = model.init_hidden(batch_size, device) # 初始化隐藏状态
    text_tensor = text_tensor.to(device)
    pred, hs = model(text_tensor, hs) # 判断
    print("概率值：", pred.item())
    # 将pred概率值转换为0或1
    pred = torch.round(pred)
    print("类别值：", pred.item())
    # 判断
    if pred.data == 1:
        print("评论正面")
    else:
        print("评论反面")

def main():
    # 读取文本数据
    with open("reviews.txt", "r") as file:
        text = file.read()

    print(len(text))
    print(type(text))
    print(text[:10])

    with open('labels.txt', 'r') as file:
        labels = file.read()

    print(len(labels))
    print(type(labels))
    print(labels[:10])

    # 清理无用的标点符号
    print("标点符号：", punctuation)

    # 遍历
    clean_text = ''.join([char for char in text if char not in punctuation])
    print(len(clean_text)) # 新的文本字符个数

    # 根据换行符 \n 分割
    clean_text = clean_text.split('\n')
    print(len(clean_text))
    print(clean_text[0])

    # 标签 根据 \n 分割
    labels = labels.split('\n')

    # 字典： 单词 --> 索引
    # 获取所有评论中的每个单词
    words = [word.lower() for sentence in clean_text for word in sentence.split(' ')]
    print(words[:10])
    various_words = list(set(words))
    various_words.remove('')
    print(len(various_words))

    # 创建字典，格式： 单词 ： 整数
    int_word = dict(enumerate(various_words, 1))
    print(int_word)

    # 字典，格式： 整数 ： 单词
    word_int = {w:int(i) for i, w in int_word.items()}
    print(word_int)

    # 标签 --> 1， 0 转换
    label_int = np.array([1 if x == 'positive' else 0 for x in labels])
    print(len(label_int))

    Counter(label_int)

    # 清理文本太短以及过长的样本
    # 统计文本中， 每条评论的长度
    sentence_length = [len(sentence.split()) for sentence in clean_text]
    counts = Counter(sentence_length) # 统计不同长度的评论

    # 最小评论长度
    min_sen = min(sorted(counts.items()))
    print(min_sen)
    # 最大
    max_sen = max(sorted(counts.items()))

    # 获取对应索引
    min_index = [i for i, length in enumerate(sentence_length) if length == min_sen[0]]
    max_index = [i for i, length in enumerate(sentence_length) if length == max_sen[0]]

    # 根据索引删除文本中过短或过长的评论
    new_text = np.delete(clean_text, min_index)
    new_text = np.delete(new_text, max_index)

    # 同样需要在标签集中根据索引删除对应的标签
    new_labels = np.delete(label_int, min_index)
    new_labels = np.delete(new_labels, max_index)

    print(new_text[0])

    # 将单词映射为整形
    text_ints = []
    for sentence in new_text:
        sample = list()
        for word in sentence.split():
            int_value = word_int[word]
            sample.append(int_value)
        text_ints.append(sample)
    print(text_ints[0]) # 第一条评论
    print(len(text_ints)) # 评论总数

    # 设定统一的文本长度，对整个文本数据中的每条评论进行填充或截断
    # 设定每条评论固定长度为200个单词，不足的评论用0填充，超过的直接截断
    dataset = reset_text(text_ints, seq_len=200)
    print(dataset.shape)

    # 数据类型的转换
    dataset_tensor = torch.from_numpy(dataset)
    label_tensor = torch.from_numpy(new_labels)

    # 数据分割
    all_samples = len(dataset_tensor)
    ratio = 0.8 # 设置比例
    train_size = int(all_samples * ratio)
    rest_size = all_samples - train_size
    val_size = int(rest_size * 0.5)
    test_size = int(rest_size * 0.5)

    # 获取train, val, test 样本
    train = dataset_tensor[:train_size]
    train_labels = label_tensor[:train_size]
    rest_samples = dataset_tensor[train_size:] # 剩余样本
    rest_labels = label_tensor[train_size:] # 剩余标签
    val = rest_samples[:val_size]
    val_labels = rest_labels[:val_size]
    test = rest_samples[val_size:]
    test_labels = rest_labels[val_size:]

    # 通过DataLoader按批处理数据
    train_dataset = TensorDataset(train, train_labels)
    val_dataset = TensorDataset(val, val_labels)
    test_dataset = TensorDataset(test, test_labels)

    batch_size = 128

    # 设置num_workers=0以避免多进程问题
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

    # 获取train中的一批数据
    data, label = next(iter(train_loader))
    print(data.shape)
    print(label.shape)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化超参数
    input_size = len(word_int) + 1 # 输入（不同的单词个数）
    output_size = 1 # 输出
    embedding_dim = 400 # 词嵌入维度
    hidden_dim = 128 # 隐藏层节点个数
    num_layers = 2 # lstm的层数

    # 创建模型
    model = Sentiment(input_size, embedding_dim, hidden_dim, output_size, num_layers)
    model = model.to(device)
    print(model)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10
    # num_epochs = 50

    # 训练模型
    train_model(model, device, train_loader, criterion, optimizer, num_epochs, val_loader, batch_size)

    # 测试模型
    test_model(model, test_loader, device, criterion, batch_size)

    # 预测
    # 案例1
    text = 'this movie is so amazing. the plot is attractive. and I really like it.'
    text_ints = converts(text, word_int)
    new_text_ints = reset_text([text_ints], seq_len=200)  # 注意这里要添加一个[]，因为reset_text处理的二维数据
    text_tensor = torch.from_numpy(new_text_ints)
    print(text_tensor.shape)
    predict(model, text_tensor, device)


if __name__ == "__main__":
    # 解决Windows下多进程问题
    freeze_support()
    main()