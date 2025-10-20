# inference_sentiment.py
import re
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# 模型定义（和训练时一致）
# -------------------------
class SentimentImproved(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_dim, output_size, num_layers, dropout=0.3):
        super(SentimentImproved, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_dim)
        nn.init.xavier_uniform_(self.embedding.weight)

        self.embed_dropout = nn.Dropout(0.2)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            dropout=dropout, batch_first=True,
                            bidirectional=True)

        self.projection = nn.Linear(embedding_dim, hidden_dim * 2)
        self.attention_weights = nn.Parameter(torch.rand(hidden_dim * 2))

        self.layer_norm1 = nn.LayerNorm(hidden_dim * 2)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        self.bn = nn.BatchNorm1d(hidden_dim * 2)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        nn.init.kaiming_normal_(self.fc1.weight)

        self.dropout = nn.Dropout(dropout)
        self.elu = nn.ELU(inplace=True)

        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_size)
        nn.init.xavier_normal_(self.fc2.weight)

        self.sigmoid = nn.Sigmoid()

    def attention_net(self, lstm_output):
        batch_size, seq_len, hidden_size = lstm_output.size()
        attn_weights = torch.bmm(
            lstm_output,
            self.attention_weights.unsqueeze(0).unsqueeze(2).expand(batch_size, -1, 1)
        ).squeeze(2)
        temp = 1.0
        soft_attn_weights = F.softmax(attn_weights / temp, dim=1).unsqueeze(2)
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights).squeeze(2)
        return context, soft_attn_weights.squeeze(2)

    def forward(self, x, hidden):
        batch_size = x.size(0)
        x = x.long()
        embeds = self.embedding(x)
        embeds = self.embed_dropout(embeds)
        residual = self.projection(embeds.mean(dim=1))
        lstm_out, hidden = self.lstm(embeds, hidden)
        attn_output, attention_weights = self.attention_net(lstm_out)
        combined = attn_output + residual
        normalized = self.layer_norm1(combined)

        # BatchNorm1d expects (batch, features)
        norm_output = self.bn(normalized)
        fc1_output = self.fc1(norm_output)
        fc1_output = self.dropout(fc1_output)
        fc1_output = self.elu(fc1_output)
        fc1_output = self.layer_norm2(fc1_output)
        fc1_output = self.bn2(fc1_output)
        fc2_output = self.fc2(fc1_output)
        sigmoid_out = self.sigmoid(fc2_output)

        return sigmoid_out, hidden, attention_weights

    def init_hidden(self, batch_size, device):
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(device)
        nn.init.normal_(h0, mean=0, std=0.01)
        nn.init.normal_(c0, mean=0, std=0.01)
        return (h0, c0)


# -------------------------
# 文本处理函数（与训练时一致）
# -------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' URL ', text)
    text = re.sub(
        r':\)|;\)|:-\)|\(-:|:-D|:D|=-\)|=\)|:-\(|:\(|:-o|:o|:-O|:O|:-0|8-\)|8\)|:-\||:\||:P|:-P|:p|:-p|:b|:-b',
        ' EMOJI ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' NUM ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# 这里使用简单词干（原脚本中使用 PorterStemmer；为避免额外依赖，这里保留一个轻量替代）
# 如果你仍希望使用 NLTK 的 PorterStemmer，请自行安装并替换下面的实现。
try:
    from nltk.stem import PorterStemmer
    _ps = PorterStemmer()
    def stemming(text):
        words = text.split()
        return ' '.join([_ps.stem(w) for w in words])
except Exception:
    # 退化：没有 nltk 时仅返回原文（最好提前安装 nltk）
    def stemming(text):
        return text

# stopwords（尝试使用 nltk 的停用词；若不可用则使用一个小集合）
try:
    from nltk.corpus import stopwords as _st
    STOPWORDS = set(_st.words('english'))
except Exception:
    STOPWORDS = set([
        'the','a','an','in','on','and','or','is','are','was','were','be','to','of','for','it','this','that'
    ])

def remove_stopwords(text):
    word_tokens = text.split()
    filtered_text = [word for word in word_tokens if word not in STOPWORDS]
    return ' '.join(filtered_text)

def converts_improved(text, word_int, unknown_token=0):
    clean = clean_text(text)
    clean = remove_stopwords(clean)
    stemmed = stemming(clean)
    text_ints = []
    for word in stemmed.split():
        if word in word_int:
            text_ints.append(word_int[word])
        else:
            text_ints.append(unknown_token)
    return text_ints

def reset_text(text_list, seq_len):
    dataset = np.zeros((len(text_list), seq_len), dtype=np.int64)
    for index, sentence in enumerate(text_list):
        if len(sentence) < seq_len:
            dataset[index, :len(sentence)] = sentence
        else:
            dataset[index, :] = sentence[:seq_len]
    return dataset


# -------------------------
# 加载模型（从 checkpoint / saved dict）
# -------------------------
def load_model_from_checkpoint(checkpoint_path, device=None):
    """
    加载 checkpoint。checkpoint 可以是：
      - 直接保存的 model.state_dict()（仅权重） -> 需要外部提供 config/vocab
      - 一个 dict 包含 keys: 'model_state_dict', 'config', 'vocab'（更理想）
    返回 (model, vocab, config)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(checkpoint_path, map_location=device)

    vocab = None
    config = None

    # 若 ckpt 包含完整信息
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt and 'config' in ckpt:
        config = ckpt.get('config')
        vocab = ckpt.get('vocab', None)
        state_dict = ckpt['model_state_dict']
        # 尝试从 config 创建模型
        if config is None:
            raise ValueError("checkpoint contains model_state_dict but no config")
        model = SentimentImproved(
            input_size=config['input_size'],
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim'],
            output_size=config['output_size'],
            num_layers=config['num_layers'],
            dropout=config.get('dropout', 0.3)
        )
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model, vocab, config

    # 若 ckpt 是直接保存的 state_dict（或模型 dict）
    elif isinstance(ckpt, dict):
        # try to guess config keys in ckpt
        if 'config' in ckpt:
            config = ckpt['config']
        if 'vocab' in ckpt:
            vocab = ckpt['vocab']
        # if state_dict is nested under 'model_state_dict'
        if 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        else:
            # assume ckpt itself is state_dict
            state_dict = ckpt

        if config is None:
            raise ValueError("Config needed to reconstruct model is missing in checkpoint. Provide config or a checkpoint saved with 'config'.")
        model = SentimentImproved(
            input_size=config['input_size'],
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim'],
            output_size=config['output_size'],
            num_layers=config['num_layers'],
            dropout=config.get('dropout', 0.3)
        )
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model, vocab, config

    else:
        raise ValueError("Unsupported checkpoint format")


# -------------------------
# 单模型预测函数
# -------------------------
def predict_improved(model, text, word_int, device=None, seq_len=400, threshold=0.5):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # 文本转索引
    text_ints = converts_improved(text, word_int)
    new_text_ints = reset_text([text_ints], seq_len=seq_len)
    text_tensor = torch.from_numpy(new_text_ints).to(device)

    # 初始化隐藏状态
    batch_size = text_tensor.size(0)
    hs = model.init_hidden(batch_size, device)

    with torch.no_grad():
        pred, _, attention_weights = model(text_tensor, hs)

    prob = float(pred.item())
    class_pred = 1 if prob > threshold else 0
    # 注意力权重仅对单个样本（batch size=1）返回
    attention_weights_out = attention_weights.cpu().numpy() if attention_weights is not None else None
    
    # 调整置信度计算，使其为 0 到 1 之间
    confidence = abs(prob - 0.5) * 2

    return {
        'probability': prob,
        'class': int(class_pred),
        'confidence': float(confidence),
        'attention_weights': attention_weights_out
    }


# -------------------------
# 集成预测函数（接收已加载模型列表）
# models_info: list of dicts {'model': model_obj}
# -------------------------
def ensemble_predict(models_info, text, word_int, device=None, seq_len=400, threshold=0.5):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 文本预处理
    text_ints = converts_improved(text, word_int)
    new_text_ints = reset_text([text_ints], seq_len=seq_len)
    text_tensor = torch.from_numpy(new_text_ints).to(device)

    preds = []
    with torch.no_grad():
        for info in models_info:
            model = info['model']
            model.to(device)
            model.eval()
            hs = model.init_hidden(text_tensor.size(0), device)
            pred, _, _ = model(text_tensor, hs)
            preds.append(float(pred.item()))

    avg_pred = float(np.mean(preds))
    class_pred = 1 if avg_pred > threshold else 0
    confidence = abs(avg_pred - 0.5) * 2

    return {
        'probability': avg_pred,
        'class': int(class_pred),
        'confidence': float(confidence),
        'individual_probs': preds
    }


# -------------------------
# 演示如何使用（多轮预测）
# -------------------------
if __name__ == "__main__":
    # 修改为你保存模型的路径
    CHECKPOINT_PATH = "sentiment_model_optimized.pth"  # 或 'best_sentiment_model.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"尝试加载模型：{CHECKPOINT_PATH}...")
    try:
        final_model, word_int, config = load_model_from_checkpoint(CHECKPOINT_PATH, device=device)
        print("模型加载成功！")
    except Exception as e:
        print(f"⚠️ 加载模型失败: {e}")
        # 在实际部署中，你需要确保模型和词汇表文件存在且格式正确。
        print("请确保提供了正确的 CHECKPOINT_PATH，并且该文件中包含模型权重、词汇表 ('vocab') 和配置 ('config')。")
        raise

    # 检查 vocab
    if word_int is None:
        raise ValueError("vocab (word_int) not found in checkpoint. 请在 checkpoint 中包含 'vocab' 映射。")

    seq_len = config.get('seq_len', 400) if config else 400
    
    # 定义待预测的文本列表
    examples = [
        ("Positive example", "This movie is so amazing. The plot is attractive and I really like it."),
        ("Negative example", "The movie was terrible. I hated every minute of it. The acting was poor."),
        ("Neutral example", "The movie had some good parts and some bad parts. The acting was okay."),
        ("Challenging example", "The movie wasn't bad, but I wouldn't call it good either."),
        ("Sarcastic example", "Wow, what a masterpiece... if you enjoy wasting two hours of your life.")
    ]

    print("\n" + "="*50)
    print("开始多轮情感预测...")
    print("="*50)
    
    # 循环进行预测
    for name, text in examples:
        print(f"\n--- 示例: {name} ---")
        print(f"原始文本: \"{text}\"")

        # 调用预测函数
        result = predict_improved(final_model, text, word_int, device=device, seq_len=seq_len, threshold=0.5)

        # 格式化输出结果
        sentiment = "POSITIVE 👍" if result['class'] == 1 else "NEGATIVE 👎"
        
        print(f"预测情感: {sentiment}")
        print(f"概率 (Positive): {result['probability']:.4f}")
        print(f"置信度: {result['confidence']:.4f} (0=低, 1=高)")
        
        # 预处理后的文本（用于理解模型的输入）
        processed_text_ints = converts_improved(text, word_int)
        processed_words = [k for k, v in word_int.items() if v in processed_text_ints and v != 0] # 仅显示非UNK词汇
        print(f"预处理词汇数: {len(processed_text_ints)}")
        # print(f"预处理词汇: {' '.join(processed_words)}") # 打印所有词汇可能很长

    print("\n" + "="*50)
    print("多轮预测完成。")