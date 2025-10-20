import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from string import punctuation
from collections import Counter
import multiprocessing
from multiprocessing import freeze_support
import re
import nltk
from nltk.stem import PorterStemmer
from sklearn.utils import class_weight


# 定义网络模型结构（修复递归错误）
class SentimentImproved(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_dim, output_size, num_layers, dropout=0.3):
        super(SentimentImproved, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.num_layers = num_layers

        # 嵌入层
        self.embedding = nn.Embedding(input_size, embedding_dim)

        # 嵌入正则化 - 防止过拟合
        self.embed_dropout = nn.Dropout(0.2)

        # 双向LSTM
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            dropout=dropout, batch_first=True,
                            bidirectional=True)

        # 注意力层 - 修改为矢量形式防止递归错误
        self.attention_weights = nn.Parameter(torch.rand(hidden_dim * 2))

        # BatchNorm和全连接层
        self.bn = nn.BatchNorm1d(hidden_dim * 2)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # 第二个BatchNorm - 增强正则化
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()

    def attention_net(self, lstm_output):
        """
        重新实现注意力机制，避免递归栈溢出
        lstm_output: [batch_size, seq_len, hidden_dim*2]
        """
        # 非递归方式计算注意力权重
        batch_size, seq_len, hidden_size = lstm_output.size()

        # 展平注意力权重以便与每个时间步的输出进行点积
        attn_weights = torch.bmm(
            lstm_output,  # [batch_size, seq_len, hidden_dim*2]
            self.attention_weights.unsqueeze(0).unsqueeze(2).expand(batch_size, -1, 1)  # [batch_size, hidden_dim*2, 1]
        ).squeeze(2)  # [batch_size, seq_len]

        # Softmax归一化
        soft_attn_weights = F.softmax(attn_weights, dim=1).unsqueeze(2)  # [batch_size, seq_len, 1]

        # 计算上下文向量
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights).squeeze(2)  # [batch_size, hidden_dim*2]

        return context

    def forward(self, x, hidden):
        '''
        x shape : (batch_size, seq_len, features)
        '''
        batch_size = x.size(0)
        x = x.long()

        # 嵌入层
        embeds = self.embedding(x)
        embeds = self.embed_dropout(embeds)  # 添加嵌入层Dropout

        # LSTM层
        lstm_out, hidden = self.lstm(embeds, hidden)

        # 注意力机制
        attn_output = self.attention_net(lstm_out)

        # BatchNorm和全连接层
        norm_output = self.bn(attn_output)
        fc1_output = self.fc1(norm_output)
        fc1_output = self.dropout(fc1_output)
        fc1_output = self.relu(fc1_output)

        # 第二个BatchNorm
        fc1_output = self.bn2(fc1_output)

        fc2_output = self.fc2(fc1_output)

        # 输出概率
        sigmoid_out = self.sigmoid(fc2_output)

        return sigmoid_out, hidden

    def init_hidden(self, batch_size, device):
        # 由于使用了双向LSTM，隐藏状态需要乘以2
        weight = next(self.parameters()).data
        hidden = (weight.new(self.num_layers * 2, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.num_layers * 2, batch_size, self.hidden_dim).zero_().to(device))
        return hidden


# 改进的文本清洗函数
def clean_text(text):
    """更全面的文本清洗函数"""
    # 转换为小写
    text = text.lower()

    # 替换URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' URL ', text)

    # 替换表情符号
    text = re.sub(
        r':\)|;\)|:-\)|\(-:|:-D|:D|=-\)|=\)|:-\(|:\(|:-o|:o|:-O|:O|:-0|8-\)|8\)|:-\||:\||:P|:-P|:p|:-p|:b|:-b',
        ' EMOJI ', text)

    # 替换特殊字符
    text = re.sub(r'[^\w\s]', ' ', text)

    # 替换数字
    text = re.sub(r'\d+', ' NUM ', text)

    # 替换多个空格为单个空格
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# 词形还原/词干提取函数
def stemming(text):
    """对文本进行词干提取"""
    ps = PorterStemmer()
    words = text.split()
    stemmed_words = [ps.stem(word) for word in words]
    return ' '.join(stemmed_words)


# 重置文本长度的函数
def reset_text(text, seq_len):
    dataset = np.zeros((len(text), seq_len))
    for index, sentence in enumerate(text):
        if len(sentence) < seq_len:
            dataset[index, :len(sentence)] = sentence
        else:
            dataset[index, :] = sentence[:seq_len]
    return dataset


# 文本预处理函数（改进版）
def converts_improved(text, word_int, unknown_token=0):
    # 清洗文本
    clean = clean_text(text)

    # 词干提取
    stemmed = stemming(clean)

    # 文本映射为索引，处理未知词
    text_ints = []
    for word in stemmed.split():
        if word in word_int:
            text_ints.append(word_int[word])
        else:
            text_ints.append(unknown_token)  # 使用0表示未知词

    return text_ints


# 优化的训练函数（防止递归溢出）
def train_model_improved(model, device, data_loader, criterion, optimizer, num_epochs, val_loader, batch_size,
                         patience=5):
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    # 使用更加灵活的学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2,
        verbose=True, min_lr=1e-6
    )

    # 早停变量
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    # 动态梯度裁剪阈值
    initial_max_norm = 5.0
    min_max_norm = 1.0
    max_norm = initial_max_norm

    # 主训练循环
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = []
        train_correct = 0
        total_train = 0

        # 动态调整梯度裁剪阈值 - 随着训练进行逐渐降低阈值
        if epoch > num_epochs // 3:
            decay_factor = 0.95
            max_norm = max(min_max_norm, max_norm * decay_factor)

        # 初始化隐藏状态
        hs = model.init_hidden(batch_size, device)

        # 训练批次循环
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)

            # 清除梯度
            optimizer.zero_grad()

            # 前向传播 - 明确分离隐藏状态以防递归
            output, hs = model(data, hs)

            # 分离隐藏状态，断开计算图以防止梯度累积和递归问题
            hs = (hs[0].detach(), hs[1].detach())

            # 计算损失
            loss = criterion(output, target.float().view(-1, 1))

            # 添加L2正则化损失（权重衰减）- 避免直接迭代model.parameters()
            l2_lambda = 1e-5
            l2_reg = torch.tensor(0.0).to(device)
            for name, param in model.named_parameters():
                if 'weight' in name:  # 只对权重应用L2正则化，不对偏置应用
                    l2_reg += torch.norm(param, 2)
            loss += l2_lambda * l2_reg

            # 记录损失
            train_loss.append(loss.item())

            # 反向传播
            loss.backward()

            # 梯度裁剪（防止梯度爆炸和过拟合）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

            # 更新参数
            optimizer.step()

            # 计算准确率
            predicted = torch.round(output)
            train_correct += (predicted == target.float().view(-1, 1)).sum().item()
            total_train += target.size(0)

        # 计算训练集平均损失和准确率
        avg_train_loss = np.mean(train_loss)
        train_accuracy = train_correct / total_train
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_accuracy)

        # 验证阶段
        model.eval()
        val_loss = []
        val_correct = 0
        total_val = 0

        # 重新初始化隐藏状态用于验证
        hs = model.init_hidden(batch_size, device)

        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(device)
                target = target.to(device)

                # 前向传播
                output, hs = model(data, hs)

                # 验证时不需要追踪梯度，但仍分离隐藏状态以防内存问题
                hs = (hs[0].detach(), hs[1].detach())

                # 计算损失
                loss = criterion(output, target.float().view(-1, 1))
                val_loss.append(loss.item())

                # 计算准确率
                predicted = torch.round(output)
                val_correct += (predicted == target.float().view(-1, 1)).sum().item()
                total_val += target.size(0)

        # 计算验证集平均损失和准确率
        avg_val_loss = np.mean(val_loss)
        val_accuracy = val_correct / total_val
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_accuracy)

        # 调整学习率
        scheduler.step(avg_val_loss)

        # 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # 保存最佳模型状态
            best_model_state = {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1

        # 打印当前梯度裁剪阈值
        print(f'Current gradient clipping max_norm: {max_norm:.2f}')

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            # 恢复最佳模型
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
                print("Restored best model state")
            break

        print(f'Epoch {epoch + 1}/{num_epochs} --- '
              f'Train Loss: {avg_train_loss:.5f} --- '
              f'Train Acc: {train_accuracy:.5f} --- '
              f'Val Loss: {avg_val_loss:.5f} --- '
              f'Val Acc: {val_accuracy:.5f}')

    # 如果没有触发早停，也要恢复最佳模型
    if best_model_state is not None and epochs_no_improve < patience:
        model.load_state_dict(best_model_state)
        print("Training complete. Restored best model state.")

    return history


# 测试模型函数（优化版，防止递归错误）
def test_model_improved(model, data_loader, device, criterion, batch_size):
    test_loss = []
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []

    # 初始化隐藏状态
    hs = model.init_hidden(batch_size, device)
    model.eval()

    with torch.no_grad():
        for i, (data, target) in enumerate(data_loader):
            data = data.to(device)
            target = target.to(device)

            # 前向传播
            output, hs = model(data, hs)

            # 分离隐藏状态
            hs = (hs[0].detach(), hs[1].detach())

            # 计算损失
            loss = criterion(output, target.float().view(-1, 1))
            test_loss.append(loss.item())

            # 计算准确率
            predicted = torch.round(output)
            correct += (predicted == target.float().view(-1, 1)).sum().item()
            total += target.size(0)

            # 收集预测和目标值用于评估
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            print(f'Batch {i + 1}')
            print(f'Loss: {loss.item():.3f}')
            batch_accuracy = ((predicted == target.float().view(-1, 1)).sum().item() / target.size(0)) * 100
            print(f'Accuracy: {batch_accuracy:.2f}%')
            print()

    # 计算总体测试结果
    avg_test_loss = np.mean(test_loss)
    test_accuracy = correct / total

    print(f"总的测试损失 Test Loss: {avg_test_loss:.5f}")
    print(f"总的测试准确率 Test Accuracy: {test_accuracy:.5f} ({correct}/{total})")

    # 绘制混淆矩阵
    try:
        from sklearn.metrics import confusion_matrix, classification_report
        # 转换预测和目标为一维数组
        flat_predictions = np.array(all_predictions).flatten()
        flat_targets = np.array(all_targets).flatten()

        cm = confusion_matrix(flat_targets, flat_predictions)
        print("混淆矩阵:")
        print(cm)
        print("\n分类报告:")
        print(classification_report(flat_targets, flat_predictions, target_names=['Negative', 'Positive']))

        # 可视化混淆矩阵
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        print("混淆矩阵图已保存为 'confusion_matrix.png'")

    except Exception as e:
        print(f"无法显示混淆矩阵和分类报告: {e}")


# 预测函数（优化版，防止递归错误）
def predict_improved(model, text, word_int, device, seq_len=200):
    model.eval()

    # 预处理文本
    text_ints = converts_improved(text, word_int)

    # 填充或截断
    new_text_ints = reset_text([text_ints], seq_len=seq_len)
    text_tensor = torch.from_numpy(new_text_ints)

    # 转移到设备
    text_tensor = text_tensor.to(device)

    # 初始化隐藏状态
    batch_size = text_tensor.size(0)
    hs = model.init_hidden(batch_size, device)

    # 预测
    with torch.no_grad():
        pred, _ = model(text_tensor, hs)

    # 输出结果
    print("文本:", text)
    print("概率值:", pred.item())
    print("预测类别:", "正面" if torch.round(pred).item() == 1 else "负面")

    return pred.item()


# FocalLoss类定义
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, inputs, targets):
        BCE_loss = self.bce(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


def main():
    try:
        # 下载必要的NLTK包
        nltk.download('punkt')
    except Exception as e:
        print(f"无法下载NLTK包，词干提取可能无法工作: {e}")

    # 读取文本数据
    try:
        with open("reviews.txt", "r", encoding="utf-8") as file:
            text = file.read()
    except Exception as e:
        print(f"无法读取reviews.txt: {e}")
        return

    print(f"总文本长度: {len(text)}")
    print(f"文本类型: {type(text)}")
    print(f"文本前10个字符: {text[:10]}")

    try:
        with open('labels.txt', 'r', encoding="utf-8") as file:
            labels = file.read()
    except Exception as e:
        print(f"无法读取labels.txt: {e}")
        return

    print(f"标签长度: {len(labels)}")
    print(f"标签类型: {type(labels)}")
    print(f"标签前10个字符: {labels[:10]}")

    # 清理数据（改进版）
    print("标点符号：", punctuation)

    # 改进的文本清洗
    clean_reviews = []
    for review in text.split('\n'):
        cleaned = clean_text(review)
        if cleaned:  # 确保不是空字符串
            clean_reviews.append(cleaned)

    # 显示清洗后的示例
    print(f"清洗后的评论数量: {len(clean_reviews)}")
    if clean_reviews:
        print(f"清洗后的第一条评论: {clean_reviews[0]}")

    # 标签处理
    labels = labels.split('\n')
    label_int = np.array([1 if x == 'positive' else 0 for x in labels[:len(clean_reviews)]])

    # 检查类别平衡情况
    positive_count = sum(label_int)
    negative_count = len(label_int) - positive_count
    print(f"正面评论数量: {positive_count}, 负面评论数量: {negative_count}")

    # 计算类别权重（用于处理不平衡）
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(label_int), y=label_int)
    print(f"类别权重: {class_weights}")

    # 词干提取
    stemmed_reviews = [stemming(review) for review in clean_reviews]

    # 创建词汇表
    words = []
    for review in stemmed_reviews:
        words.extend(review.split())

    # 创建词频统计
    word_counts = Counter(words)
    print(f"词汇总数: {len(word_counts)}")

    # 移除低频词（出现次数少于3次的词）
    min_word_count = 3
    filtered_words = {word for word, count in word_counts.items() if count >= min_word_count}
    print(f"过滤后的词汇量: {len(filtered_words)}")

    # 创建词汇映射
    word_int = {word: i + 1 for i, word in enumerate(filtered_words)}
    int_word = {i + 1: word for i, word in enumerate(filtered_words)}

    # 将评论转换为整数序列
    text_ints = []
    for review in stemmed_reviews:
        sample = []
        for word in review.split():
            if word in word_int:
                sample.append(word_int[word])
            else:
                sample.append(0)  # 0表示未知词
        if sample:  # 确保不是空列表
            text_ints.append(sample)

    print(f"处理后的评论数量: {len(text_ints)}")
    if text_ints:
        print(f"第一条评论的整数表示: {text_ints[0][:10]}...")

    # 设定统一的文本长度
    # seq_len = 400
    seq_len = 400
    dataset = reset_text(text_ints, seq_len=seq_len)
    print(f"数据集形状: {dataset.shape}")

    # 数据类型的转换
    dataset_tensor = torch.from_numpy(dataset)
    label_tensor = torch.from_numpy(label_int[:len(dataset)])

    # 数据分割
    all_samples = len(dataset_tensor)
    ratio = 0.8
    train_size = int(all_samples * ratio)
    rest_size = all_samples - train_size
    val_size = int(rest_size * 0.5)
    test_size = rest_size - val_size

    # 获取train, val, test 样本
    train = dataset_tensor[:train_size]
    train_labels = label_tensor[:train_size]
    rest_samples = dataset_tensor[train_size:]
    rest_labels = label_tensor[train_size:]
    val = rest_samples[:val_size]
    val_labels = rest_labels[:val_size]
    test = rest_samples[val_size:]
    test_labels = rest_labels[val_size:]

    # 创建数据加载器
    train_dataset = TensorDataset(train, train_labels)
    val_dataset = TensorDataset(val, val_labels)
    test_dataset = TensorDataset(test, test_labels)

    batch_size = 128  # 减小批量大小以减少内存使用
    # batch_size = 128

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                            drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             drop_last=True)

    # 获取一批数据查看形状
    try:
        data, label = next(iter(train_loader))
        print(f"输入形状: {data.shape}")
        print(f"标签形状: {label.shape}")
    except Exception as e:
        print(f"无法获取数据批次: {e}")

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 模型参数
    input_size = len(word_int) + 1  # +1 表示未知词的索引0
    output_size = 1
    embedding_dim = 256  # 减小嵌入维度
    hidden_dim = 128
    num_layers = 2
    dropout = 0.4  # 减小dropout

    # 创建改进的模型
    model = SentimentImproved(input_size, embedding_dim, hidden_dim, output_size, num_layers, dropout)
    model = model.to(device)
    print(model)

    # 使用FocalLoss或BCELoss
    use_focal_loss = True
    if use_focal_loss:
        criterion = FocalLoss(alpha=0.25, gamma=2)
    else:
        criterion = nn.BCELoss()  # 二元交叉熵损失

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.05, weight_decay=1e-5)

    # 训练模型
    num_epochs = 50  # 减少训练轮数，配合早停机制
    history = train_model_improved(model, device, train_loader, criterion, optimizer, num_epochs, val_loader,
                                   batch_size, patience=7)  # 增加早停的耐心

    # 绘制训练历史
    try:
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Training Loss And Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Training Accuracy')
        plt.plot(history['val_acc'], label='Validation Accuracy')
        plt.title('Training Accuracy And Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.savefig('training_history.png')
        print("训练历史图表已保存为 'training_history.png'")
    except Exception as e:
        print(f"无法绘制训练历史: {e}")

    # 测试模型
    test_model_improved(model, test_loader, device, criterion, batch_size)

    # 保存模型
    try:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'vocab': word_int,
            'config': {
                'input_size': input_size,
                'embedding_dim': embedding_dim,
                'hidden_dim': hidden_dim,
                'output_size': output_size,
                'num_layers': num_layers,
                'dropout': dropout
            }
        }, 'sentiment_model_improved.pth')
        print("模型已保存为 'sentiment_model_improved.pth'")
    except Exception as e:
        print(f"无法保存模型: {e}")

    # 预测示例
    print("\n示例预测:")

    # 积极评论示例
    positive_example = "This movie is so amazing. The plot is attractive and I really like it."
    predict_improved(model, positive_example, word_int, device)

    # 消极评论示例
    negative_example = "The movie was terrible. I hated every minute of it. The acting was poor."
    predict_improved(model, negative_example, word_int, device)

    # 中性评论示例
    neutral_example = "The movie had some good parts and some bad parts. The acting was okay."
    predict_improved(model, neutral_example, word_int, device)


if __name__ == "__main__":
    # 解决Windows下多进程问题
    freeze_support()
    main()