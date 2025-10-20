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
from nltk.corpus import stopwords
from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedKFold
import random
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


# 设置随机种子以确保结果可复现
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 定义网络模型结构（改进版）
class SentimentImproved(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_dim, output_size, num_layers, dropout=0.3):
        super(SentimentImproved, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.num_layers = num_layers

        # 嵌入层 - 增加初始化范围控制
        self.embedding = nn.Embedding(input_size, embedding_dim)
        # 使用Xavier初始化提高收敛性能
        nn.init.xavier_uniform_(self.embedding.weight)

        # 嵌入正则化 - 防止过拟合
        self.embed_dropout = nn.Dropout(0.2)

        # 双向LSTM
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            dropout=dropout, batch_first=True,
                            bidirectional=True)

        # 残差连接用的投影层
        self.projection = nn.Linear(embedding_dim, hidden_dim * 2)

        # 注意力层
        self.attention_weights = nn.Parameter(torch.rand(hidden_dim * 2))

        # 添加Layer Normalization
        self.layer_norm1 = nn.LayerNorm(hidden_dim * 2)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        # BatchNorm和全连接层
        self.bn = nn.BatchNorm1d(hidden_dim * 2)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)

        # 给全连接层添加权重初始化
        nn.init.kaiming_normal_(self.fc1.weight)

        self.dropout = nn.Dropout(dropout)
        # 使用ELU激活函数，缓解死神经元问题
        self.elu = nn.ELU(inplace=True)

        # 第二个BatchNorm - 增强正则化
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, output_size)
        nn.init.xavier_normal_(self.fc2.weight)

        self.sigmoid = nn.Sigmoid()

    def attention_net(self, lstm_output):
        """
        改进的注意力机制
        lstm_output: [batch_size, seq_len, hidden_dim*2]
        """
        # 计算注意力分数
        batch_size, seq_len, hidden_size = lstm_output.size()

        # 注意力分数计算
        attn_weights = torch.bmm(
            lstm_output,  # [batch_size, seq_len, hidden_dim*2]
            self.attention_weights.unsqueeze(0).unsqueeze(2).expand(batch_size, -1, 1)  # [batch_size, hidden_dim*2, 1]
        ).squeeze(2)  # [batch_size, seq_len]

        # Softmax归一化 - 添加温度参数使分布更平滑
        temp = 1.0
        soft_attn_weights = F.softmax(attn_weights / temp, dim=1).unsqueeze(2)  # [batch_size, seq_len, 1]

        # 计算上下文向量
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights).squeeze(2)  # [batch_size, hidden_dim*2]

        return context, soft_attn_weights.squeeze(2)

    def forward(self, x, hidden):
        '''
        x shape : (batch_size, seq_len, features)
        '''
        batch_size = x.size(0)
        x = x.long()

        # 嵌入层
        embeds = self.embedding(x)
        embeds = self.embed_dropout(embeds)  # 添加嵌入层Dropout

        # 保存嵌入输出用于残差连接
        residual = self.projection(embeds.mean(dim=1))  # 对序列维度求平均

        # LSTM层
        lstm_out, hidden = self.lstm(embeds, hidden)

        # 注意力机制
        attn_output, attention_weights = self.attention_net(lstm_out)

        # 残差连接 - 将嵌入层的输出与LSTM输出结合，减轻梯度消失问题
        combined = attn_output + residual

        # Layer Normalization - 稳定训练
        normalized = self.layer_norm1(combined)

        # BatchNorm和全连接层
        norm_output = self.bn(normalized)
        fc1_output = self.fc1(norm_output)
        fc1_output = self.dropout(fc1_output)
        fc1_output = self.elu(fc1_output)  # 使用ELU替代ReLU

        # 第二个Layer Normalization
        fc1_output = self.layer_norm2(fc1_output)

        # 第二个BatchNorm
        fc1_output = self.bn2(fc1_output)

        fc2_output = self.fc2(fc1_output)

        # 输出概率
        sigmoid_out = self.sigmoid(fc2_output)

        return sigmoid_out, hidden, attention_weights

    def init_hidden(self, batch_size, device):
        # 由于使用了双向LSTM，隐藏状态需要乘以2
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(device)

        # 使用正态分布初始化隐藏状态，有助于打破对称性
        nn.init.normal_(h0, mean=0, std=0.01)
        nn.init.normal_(c0, mean=0, std=0.01)

        return (h0, c0)


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


# 移除停用词
def remove_stopwords(text):
    """移除停用词"""
    try:
        stop_words = set(stopwords.words('english'))
        word_tokens = text.split()
        filtered_text = [word for word in word_tokens if word not in stop_words]
        return ' '.join(filtered_text)
    except:
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
        word_tokens = text.split()
        filtered_text = [word for word in word_tokens if word not in stop_words]
        return ' '.join(filtered_text)


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

    # 移除停用词
    clean = remove_stopwords(clean)

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


# 数据增强函数
def augment_data(texts, labels, augmentation_factor=0.3):
    """
    通过随机删除、替换和重排单词来增强文本数据
    """
    augmented_texts = []
    augmented_labels = []

    for i, (text, label) in enumerate(zip(texts, labels)):
        # 只对一部分数据做增强
        if random.random() > augmentation_factor:
            continue

        words = text.split()

        if len(words) <= 3:  # 太短的文本跳过增强
            continue

        # 随机删除部分单词
        if random.random() < 0.5 and len(words) > 5:
            drop_indices = random.sample(range(len(words)), int(len(words) * 0.1))
            new_words = [words[i] for i in range(len(words)) if i not in drop_indices]
            augmented_texts.append(' '.join(new_words))
            augmented_labels.append(label)

        # 随机交换相邻单词的位置
        if random.random() < 0.5 and len(words) > 3:
            for _ in range(int(len(words) * 0.1)):
                idx = random.randint(0, len(words) - 2)
                words[idx], words[idx + 1] = words[idx + 1], words[idx]
            augmented_texts.append(' '.join(words))
            augmented_labels.append(label)

    return augmented_texts, augmented_labels


# 混合精度训练函数 - 提高训练速度，减少内存使用
class MixedPrecisionTrainer:
    def __init__(self, model, optimizer, scaler=None):
        self.model = model
        self.optimizer = optimizer

        # 更新GradScaler以使用新语法
        self.scaler = scaler if scaler is not None else torch.amp.GradScaler('cuda')

    def train_step(self, data, target, criterion, device):
        # 将数据移至设备
        data = data.to(device)
        target = target.to(device)

        # 清除梯度
        self.optimizer.zero_grad()

        # 初始化隐藏状态
        hs = self.model.init_hidden(data.size(0), device)

        # 使用更新的语法的autocast
        with torch.amp.autocast('cuda'):
            # 前向传播
            output, hs, _ = self.model(data, hs)

            # 计算损失
            loss = criterion(output, target.float().view(-1, 1))

        # 缩放损失并反向传播
        self.scaler.scale(loss).backward()

        # 取消优化器的缩放
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # 使用scaler更新参数
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.item(), output

# 优化的训练函数
def train_model_improved(model, device, data_loader, criterion, optimizer, num_epochs, val_loader, batch_size,
                         patience=5, fold=None):
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [],
               'train_f1': [], 'val_f1': [], 'train_auc': [], 'val_auc': []}

    # 使用余弦退火学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )

    # 早停变量
    best_val_f1 = 0  # 使用F1作为早停指标，更适合不平衡数据
    epochs_no_improve = 0
    best_model_state = None

    # 混合精度训练器
    if torch.cuda.is_available():
        mp_trainer = MixedPrecisionTrainer(model, optimizer)

    # 主训练循环
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = []
        train_preds = []
        train_targets = []

        # 训练批次循环
        for data, target in data_loader:
            # 使用混合精度训练
            if torch.cuda.is_available():
                loss, output = mp_trainer.train_step(data, target, criterion, device)
                train_loss.append(loss)

                # 收集预测和目标
                predicted = torch.round(output)
                train_preds.extend(predicted.cpu().detach().numpy())
                train_targets.extend(target.cpu().numpy())
            else:
                # 标准训练过程（无混合精度）
                data = data.to(device)
                target = target.to(device)

                # 清除梯度
                optimizer.zero_grad()

                # 初始化隐藏状态
                hs = model.init_hidden(data.size(0), device)

                # 前向传播
                output, hs, _ = model(data, hs)

                # 计算损失
                loss = criterion(output, target.float().view(-1, 1))
                train_loss.append(loss.item())

                # 反向传播
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # 更新参数
                optimizer.step()

                # 收集预测和目标
                predicted = torch.round(output)
                train_preds.extend(predicted.cpu().detach().numpy())
                train_targets.extend(target.cpu().numpy())

        # 计算训练集评估指标
        train_preds = np.array(train_preds).flatten()
        train_targets = np.array(train_targets).flatten()

        avg_train_loss = np.mean(train_loss)
        train_accuracy = np.mean(train_preds == train_targets)
        train_f1 = f1_score(train_targets, train_preds, zero_division=0)

        # 计算AUC (如果预测包含两个类别)
        if len(np.unique(train_targets)) > 1:
            train_auc = roc_auc_score(train_targets, train_preds)
        else:
            train_auc = 0.5

        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_accuracy)
        history['train_f1'].append(train_f1)
        history['train_auc'].append(train_auc)

        # 验证阶段
        model.eval()
        val_loss = []
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(device)
                target = target.to(device)

                # 初始化隐藏状态
                hs = model.init_hidden(data.size(0), device)

                # 前向传播
                output, hs, _ = model(data, hs)

                # 计算损失
                loss = criterion(output, target.float().view(-1, 1))
                val_loss.append(loss.item())

                # 收集预测和目标
                predicted = torch.round(output)
                val_preds.extend(predicted.cpu().numpy())
                val_targets.extend(target.cpu().numpy())

        # 计算验证集评估指标
        val_preds = np.array(val_preds).flatten()
        val_targets = np.array(val_targets).flatten()

        avg_val_loss = np.mean(val_loss)
        val_accuracy = np.mean(val_preds == val_targets)
        val_f1 = f1_score(val_targets, val_preds, zero_division=0)

        # 计算AUC
        if len(np.unique(val_targets)) > 1:
            val_auc = roc_auc_score(val_targets, val_preds)
        else:
            val_auc = 0.5

        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_accuracy)
        history['val_f1'].append(val_f1)
        history['val_auc'].append(val_auc)

        # 调整学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # 早停检查 - 使用F1分数作为指标
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            # 保存最佳模型状态
            best_model_state = {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            # 恢复最佳模型
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
                print("Restored best model state")
            break

        # 打印训练信息
        fold_info = f"Fold {fold} " if fold is not None else ""
        print(f'{fold_info}Epoch {epoch + 1}/{num_epochs} --- '
              f'LR: {current_lr:.6f} --- '
              f'Train Loss: {avg_train_loss:.5f} --- '
              f'Train Acc: {train_accuracy:.5f} --- '
              f'Train F1: {train_f1:.5f} --- '
              f'Val Loss: {avg_val_loss:.5f} --- '
              f'Val Acc: {val_accuracy:.5f} --- '
              f'Val F1: {val_f1:.5f}')

    # 如果没有触发早停，也要恢复最佳模型
    if best_model_state is not None and epochs_no_improve < patience:
        model.load_state_dict(best_model_state)
        print("Training complete. Restored best model state.")

    return history


# FocalLoss类定义 - 更适合不平衡数据集
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        # 用BCEWithLogitsLoss替换BCELoss
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, targets):
        # 如果模型已经输出概率（经过sigmoid后），使用这个：
        BCE_loss = -targets * torch.log(inputs + 1e-7) - (1 - targets) * torch.log(1 - inputs + 1e-7)

        # 或者，如果模型输出的是logits，使用BCEWithLogitsLoss：
        # BCE_loss = self.bce_with_logits(inputs, targets)

        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()

# 测试模型函数（优化版）
def test_model_improved(model, data_loader, device, criterion, batch_size, threshold=0.5):
    test_loss = []
    all_predictions = []
    all_raw_predictions = []  # 存储原始的预测概率
    all_targets = []

    model.eval()

    with torch.no_grad():
        for i, (data, target) in enumerate(data_loader):
            data = data.to(device)
            target = target.to(device)

            # 初始化隐藏状态
            hs = model.init_hidden(data.size(0), device)

            # 前向传播
            output, _, attention_weights = model(data, hs)

            # 计算损失
            loss = criterion(output, target.float().view(-1, 1))
            test_loss.append(loss.item())

            # 存储原始预测概率
            all_raw_predictions.extend(output.cpu().numpy())

            # 应用阈值进行二分类
            predicted = (output > threshold).float()

            # 收集预测和目标值用于评估
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            if i % 10 == 0:  # 每10个批次打印一次
                print(f'Batch {i + 1}/{len(data_loader)}')
                print(f'Loss: {loss.item():.3f}')
                batch_accuracy = ((predicted == target.float().view(-1, 1)).sum().item() / target.size(0)) * 100
                print(f'Accuracy: {batch_accuracy:.2f}%')

    # 转换为NumPy数组，方便计算
    all_predictions = np.array(all_predictions).flatten()
    all_raw_predictions = np.array(all_raw_predictions).flatten()
    all_targets = np.array(all_targets).flatten()

    # 计算总体测试结果
    avg_test_loss = np.mean(test_loss)
    test_accuracy = np.mean(all_predictions == all_targets)

    # 计算更全面的评估指标
    precision = precision_score(all_targets, all_predictions, zero_division=0)
    recall = recall_score(all_targets, all_predictions, zero_division=0)
    f1 = f1_score(all_targets, all_predictions, zero_division=0)

    # 计算AUC
    if len(np.unique(all_targets)) > 1:
        auc = roc_auc_score(all_targets, all_raw_predictions)
    else:
        auc = 0.5

    print("\n===== 测试结果 =====")
    print(f"测试损失 (Loss): {avg_test_loss:.5f}")
    print(f"准确率 (Accuracy): {test_accuracy:.5f}")
    print(f"精确率 (Precision): {precision:.5f}")
    print(f"召回率 (Recall): {recall:.5f}")
    print(f"F1分数 (F1 Score): {f1:.5f}")
    print(f"AUC: {auc:.5f}")

    # 绘制混淆矩阵
    try:
        from sklearn.metrics import confusion_matrix, classification_report

        cm = confusion_matrix(all_targets, all_predictions)
        print("\n混淆矩阵:")
        print(cm)

        # 计算更详细的分类报告
        print("\n分类报告:")
        print(classification_report(all_targets, all_predictions, target_names=['Negative', 'Positive']))

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

        # 找出错误分类的样本索引
        misclassified = np.where(all_predictions != all_targets)[0]
        print(f"\n错误分类样本数: {len(misclassified)}")

    except Exception as e:
        print(f"无法显示混淆矩阵和分类报告: {e}")

    # 返回评估指标，方便后续分析
    metrics = {
        'loss': avg_test_loss,
        'accuracy': test_accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }

    return metrics


# 预测函数（优化版）
def predict_improved(model, text, word_int, device, seq_len=400, threshold=0.5):
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
        pred, _, attention_weights = model(text_tensor, hs)

    # 应用阈值
    class_pred = 1 if pred.item() > threshold else 0

    # 输出结果
    print("文本:", text)
    print("预测概率值:", pred.item())
    print("预测类别:", "正面" if class_pred == 1 else "负面")
    print("预测置信度:", abs(pred.item() - 0.5) * 2)  # 转换为0-1的置信度分数

    return pred.item(), class_pred, attention_weights


# 交叉验证训练函数
def train_with_kfold(X, y, word_int, device, config, n_splits=5):
    """使用K折交叉验证训练模型，减少过拟合风险"""
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_metrics = []
    best_model = None
    best_f1 = 0

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        print(f"\n{'=' * 20} Fold {fold + 1}/{n_splits} {'=' * 20}")

        # 分割数据
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 创建数据加载器
        train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
                                  num_workers=config['num_workers'], drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False,
                                num_workers=config['num_workers'], drop_last=True)

        # 创建模型
        model = SentimentImproved(
            input_size=config['input_size'],
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim'],
            output_size=config['output_size'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        )
        model = model.to(device)

        # 创建损失函数和优化器
        criterion = FocalLoss(alpha=0.25, gamma=2)
        # Continue from where paste.txt was cut off
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                      weight_decay=config['weight_decay'])

        # Train the model
        history = train_model_improved(
            model=model,
            device=device,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=config['num_epochs'],
            val_loader=val_loader,
            batch_size=config['batch_size'],
            patience=config['patience'],
            fold=fold + 1
        )

        # Evaluate on validation set
        print(f"\nEvaluating fold {fold + 1} model on validation set:")
        val_metrics = test_model_improved(
            model=model,
            data_loader=val_loader,
            device=device,
            criterion=criterion,
            batch_size=config['batch_size']
        )

        fold_metrics.append(val_metrics)

        # Keep track of best model based on F1 score
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_model = {
                'state_dict': {k: v.cpu().detach().clone() for k, v in model.state_dict().items()},
                'fold': fold + 1,
                'metrics': val_metrics
            }

        # Print average metrics across all folds
    print("\n" + "=" * 50)
    print("K-fold Cross Validation Results:")
    print("=" * 50)

    avg_metrics = {metric: np.mean([fold[metric] for fold in fold_metrics]) for metric in fold_metrics[0].keys()}
    std_metrics = {metric: np.std([fold[metric] for fold in fold_metrics]) for metric in fold_metrics[0].keys()}

    for metric in avg_metrics.keys():
        print(f"Average {metric}: {avg_metrics[metric]:.5f} ± {std_metrics[metric]:.5f}")

    print(f"Best model from fold {best_model['fold']} with F1 score: {best_model['metrics']['f1']:.5f}")

    return best_model


# Ensemble prediction function to combine multiple models
def ensemble_predict(models, text, word_int, device, seq_len=400, threshold=0.5):
    """
    Combine predictions from multiple models for more robust results
    """
    predictions = []

    # Preprocess text once
    text_ints = converts_improved(text, word_int)
    new_text_ints = reset_text([text_ints], seq_len=seq_len)
    text_tensor = torch.from_numpy(new_text_ints).to(device)

    # Get predictions from each model
    for model_info in models:
        model = model_info['model']
        model.eval()

        # Initialize hidden state
        batch_size = text_tensor.size(0)
        hs = model.init_hidden(batch_size, device)

        # Predict
        with torch.no_grad():
            pred, _, _ = model(text_tensor, hs)
            predictions.append(pred.item())

    # Average predictions
    avg_pred = np.mean(predictions)

    # Apply threshold
    class_pred = 1 if avg_pred > threshold else 0

    # Output results
    print("Text:", text)
    print("Ensemble prediction probability:", avg_pred)
    print("Predicted class:", "Positive" if class_pred == 1 else "Negative")
    print("Prediction confidence:", abs(avg_pred - 0.5) * 2)

    return avg_pred, class_pred


# Learning rate finder to help select optimal learning rates
def find_optimal_lr(model, train_loader, criterion, device, start_lr=1e-7, end_lr=10, num_iter=100):
    """
    Implements a learning rate finder to identify optimal learning rate
    """
    # Clone model to not affect the original
    model_clone = type(model)(
        model.embedding.num_embeddings,
        model.embedding.embedding_dim,
        model.hidden_dim,
        model.output_size,
        model.num_layers,
        dropout=0.3
    ).to(device)
    model_clone.load_state_dict({k: v.clone() for k, v in model.state_dict().items()})

    # Set up optimizer with very low learning rate
    optimizer = torch.optim.Adam(model_clone.parameters(), lr=start_lr)

    # Calculate learning rate multiplier
    mult = (end_lr / start_lr) ** (1 / num_iter)

    # Storage for learning rates and losses
    lr_history = []
    loss_history = []

    # Training loop
    model_clone.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx >= num_iter:
            break

        # Update learning rate
        lr = start_lr * (mult ** batch_idx)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Record learning rate
        lr_history.append(lr)

        # Move data to device
        data, target = data.to(device), target.to(device)

        # Initialize hidden state
        hs = model_clone.init_hidden(data.size(0), device)

        # Forward pass
        optimizer.zero_grad()
        output, hs, _ = model_clone(data, hs)
        loss = criterion(output, target.float().view(-1, 1))

        # Record loss
        loss_history.append(loss.item())

        # Backward pass
        loss.backward()
        optimizer.step()

        print(f"Iteration {batch_idx + 1}/{num_iter}, LR: {lr:.8f}, Loss: {loss.item():.4f}")

    # Plot loss vs learning rate
    plt.figure(figsize=(10, 6))
    plt.plot(lr_history, loss_history)
    plt.xscale('log')
    plt.xlabel('Learning Rate (log scale)')
    plt.ylabel('Loss')
    plt.title('Loss vs. Learning Rate')
    plt.savefig('lr_finder.png')
    plt.close()

    print("Learning rate finder plot saved as 'lr_finder.png'")

    # Find the optimal learning rate (where loss is decreasing most steeply)
    smoothed_losses = np.array(loss_history)
    smooth_f = 0.05
    for i in range(1, len(smoothed_losses)):
        smoothed_losses[i] = smoothed_losses[i - 1] * (1 - smooth_f) + smoothed_losses[i] * smooth_f

    # Calculate gradients
    gradients = (smoothed_losses[1:] - smoothed_losses[:-1]) / (lr_history[1:] - lr_history[:-1])

    # Find the point of steepest descent
    min_grad_idx = np.argmin(gradients) if len(gradients) > 0 else 0
    optimal_lr = lr_history[min_grad_idx]

    # Divide by 10 to be conservative (the steepest point may be too high)
    recommended_lr = optimal_lr / 10

    print(f"Recommended learning rate: {recommended_lr:.8f}")

    return recommended_lr


def main():
    # Set random seed for reproducibility
    set_seed(42)

    # Download required NLTK packages
    try:
        nltk.download('punkt')
        nltk.download('stopwords')
    except Exception as e:
        print(f"Unable to download NLTK packages: {e}")

    # Read text data
    try:
        with open("reviews.txt", "r", encoding="utf-8") as file:
            text = file.read()
    except Exception as e:
        print(f"Unable to read reviews.txt: {e}")
        return

    print(f"Total text length: {len(text)}")

    try:
        with open('labels.txt', 'r', encoding="utf-8") as file:
            labels = file.read()
    except Exception as e:
        print(f"Unable to read labels.txt: {e}")
        return

    print(f"Labels length: {len(labels)}")

    # Clean and preprocess data
    clean_reviews = []
    for review in text.split('\n'):
        cleaned = clean_text(review)
        # Remove stopwords
        cleaned = remove_stopwords(cleaned)
        if cleaned:  # Ensure it's not an empty string
            clean_reviews.append(cleaned)

    # Display cleaned examples
    print(f"Number of cleaned reviews: {len(clean_reviews)}")
    if clean_reviews:
        print(f"First cleaned review: {clean_reviews[0]}")

    # Process labels
    labels = labels.split('\n')
    label_int = np.array([1 if x == 'positive' else 0 for x in labels[:len(clean_reviews)]])

    # Check class balance
    positive_count = sum(label_int)
    negative_count = len(label_int) - positive_count
    print(f"Positive reviews: {positive_count}, Negative reviews: {negative_count}")

    # Calculate class weights for imbalance handling
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(label_int), y=label_int)
    print(f"Class weights: {class_weights}")

    # Apply stemming
    stemmed_reviews = [stemming(review) for review in clean_reviews]

    # Build vocabulary
    words = []
    for review in stemmed_reviews:
        words.extend(review.split())

    # Create word frequency counter
    word_counts = Counter(words)
    print(f"Total vocabulary: {len(word_counts)}")

    # Filter low-frequency words (appearing less than 3 times)
    min_word_count = 3
    filtered_words = {word for word, count in word_counts.items() if count >= min_word_count}
    print(f"Filtered vocabulary size: {len(filtered_words)}")

    # Create word mappings
    word_int = {word: i + 1 for i, word in enumerate(filtered_words)}
    int_word = {i + 1: word for i, word in enumerate(filtered_words)}

    # Convert reviews to integer sequences
    text_ints = []
    for review in stemmed_reviews:
        sample = []
        for word in review.split():
            if word in word_int:
                sample.append(word_int[word])
            else:
                sample.append(0)  # 0 represents unknown words
        if sample:
            text_ints.append(sample)

    print(f"Number of processed reviews: {len(text_ints)}")

    # Perform data augmentation
    print("Performing data augmentation...")
    augmented_texts, augmented_labels = augment_data(clean_reviews, label_int)

    print(f"Added {len(augmented_texts)} augmented samples")

    # Process augmented texts
    augmented_text_ints = []
    for aug_text in augmented_texts:
        # Apply stemming
        stemmed = stemming(aug_text)

        # Convert to integer sequence
        sample = []
        for word in stemmed.split():
            if word in word_int:
                sample.append(word_int[word])
            else:
                sample.append(0)
        if sample:
            augmented_text_ints.append(sample)

    # Combine original and augmented data
    all_text_ints = text_ints + augmented_text_ints
    all_labels = np.concatenate([label_int, augmented_labels])

    print(f"Final dataset size after augmentation: {len(all_text_ints)}")

    # Set sequence length
    seq_len = 400
    dataset = reset_text(all_text_ints, seq_len=seq_len)
    print(f"Dataset shape: {dataset.shape}")

    # Convert to tensors
    dataset_tensor = torch.from_numpy(dataset)
    label_tensor = torch.from_numpy(all_labels)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Model configuration
    config = {
        'input_size': len(word_int) + 1,  # +1 for unknown token index 0
        'output_size': 1,
        'embedding_dim': 256,
        'hidden_dim': 128, 
        'num_layers': 3,
        'dropout': 0.4,
        'batch_size': 128,
        'num_workers': 0,
        'num_epochs': 30,
        'patience': 7,
        'learning_rate': 0.001,
        'weight_decay': 1e-5
    }

    # Option 1: Simple train/val/test split
    if config.get('use_simple_split', False):
        # Split data
        all_samples = len(dataset_tensor)
        train_size = int(all_samples * 0.8)
        rest_size = all_samples - train_size
        val_size = int(rest_size * 0.5)
        test_size = rest_size - val_size

        # Get train, val, test samples
        train = dataset_tensor[:train_size]
        train_labels = label_tensor[:train_size]
        rest_samples = dataset_tensor[train_size:]
        rest_labels = label_tensor[train_size:]
        val = rest_samples[:val_size]
        val_labels = rest_labels[:val_size]
        test = rest_samples[val_size:]
        test_labels = rest_labels[val_size:]

        # Create datasets
        train_dataset = TensorDataset(train, train_labels)
        val_dataset = TensorDataset(val, val_labels)
        test_dataset = TensorDataset(test, test_labels)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
                                  num_workers=config['num_workers'], pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False,
                                num_workers=config['num_workers'], pin_memory=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False,
                                 num_workers=config['num_workers'], pin_memory=True, drop_last=True)

        # Create model
        model = SentimentImproved(
            input_size=config['input_size'],
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim'],
            output_size=config['output_size'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        )
        model = model.to(device)

        # Use learning rate finder
        print("Finding optimal learning rate...")
        optimal_lr = find_optimal_lr(
            model=model,
            train_loader=train_loader,
            criterion=FocalLoss(alpha=0.25, gamma=2),
            device=device,
            start_lr=1e-7,
            end_lr=1,
            num_iter=100
        )

        # Update learning rate in config
        config['learning_rate'] = optimal_lr

        # Create loss function and optimizer
        criterion = FocalLoss(alpha=0.25, gamma=2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                      weight_decay=config['weight_decay'])

        # Train model
        history = train_model_improved(
            model=model,
            device=device,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=config['num_epochs'],
            val_loader=val_loader,
            batch_size=config['batch_size'],
            patience=config['patience']
        )

        # Plot training history
        plt.figure(figsize=(12, 8))

        # Plot loss
        plt.subplot(2, 2, 1)
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Plot accuracy
        plt.subplot(2, 2, 2)
        plt.plot(history['train_acc'], label='Training Accuracy')
        plt.plot(history['val_acc'], label='Validation Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        # Plot F1 score
        plt.subplot(2, 2, 3)
        plt.plot(history['train_f1'], label='Training F1')
        plt.plot(history['val_f1'], label='Validation F1')
        plt.title('F1 Score')
        plt.xlabel('Epochs')
        plt.ylabel('F1')
        plt.legend()

        # Plot AUC
        plt.subplot(2, 2, 4)
        plt.plot(history['train_auc'], label='Training AUC')
        plt.plot(history['val_auc'], label='Validation AUC')
        plt.title('AUC')
        plt.xlabel('Epochs')
        plt.ylabel('AUC')
        plt.legend()

        plt.tight_layout()
        plt.savefig('training_history_detailed.png')
        print("Training history plots saved as 'training_history_detailed.png'")

        # Test model
        test_metrics = test_model_improved(
            model=model,
            data_loader=test_loader,
            device=device,
            criterion=criterion,
            batch_size=config['batch_size']
        )

    else:
        # Option 2: K-fold cross-validation (recommended for better generalization)
        print("\nPerforming K-fold cross-validation training...")
        best_model_info = train_with_kfold(
            X=dataset,
            y=all_labels,
            word_int=word_int,
            device=device,
            config=config,
            n_splits=5
        )

        # Create and save the best model
        best_model = SentimentImproved(
            input_size=config['input_size'],
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim'],
            output_size=config['output_size'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        )
        best_model.load_state_dict(best_model_info['state_dict'])
        best_model = best_model.to(device)

        # Save the model
        torch.save({
            'model_state_dict': best_model.state_dict(),
            'config': config,
            'vocab': word_int,
            'metrics': best_model_info['metrics'],
            'fold': best_model_info['fold']
        }, 'best_sentiment_model.pth')
        print("Best model saved as 'best_sentiment_model.pth'")

    # Save the final model
    try:
        final_model = model if 'model' in locals() else best_model
        torch.save({
            'model_state_dict': final_model.state_dict(),
            'vocab': word_int,
            'config': config
        }, 'sentiment_model_optimized.pth')
        print("Model saved as 'sentiment_model_optimized.pth'")
    except Exception as e:
        print(f"Error saving model: {e}")

    # Example predictions
    print("\nExample predictions:")

    # Positive example
    positive_example = "This movie is so amazing. The plot is attractive and I really like it."
    predict_improved(final_model, positive_example, word_int, device)

    # Negative example
    negative_example = "The movie was terrible. I hated every minute of it. The acting was poor."
    predict_improved(final_model, negative_example, word_int, device)

    # Neutral example
    neutral_example = "The movie had some good parts and some bad parts. The acting was okay."
    predict_improved(final_model, neutral_example, word_int, device)

    # Additional challenging examples
    challenging_example = "The movie wasn't bad, but I wouldn't call it good either."
    predict_improved(final_model, challenging_example, word_int, device)

    sarcastic_example = "Wow, what a masterpiece... if you enjoy wasting two hours of your life."
    predict_improved(final_model, sarcastic_example, word_int, device)


if __name__ == "__main__":
    # Solve multiprocessing issues on Windows
    freeze_support()
    main()