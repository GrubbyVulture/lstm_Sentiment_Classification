
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from v3 import SentimentImproved, test_model_improved, reset_text, converts_improved, clean_text, remove_stopwords, stemming

def evaluate_on_training_data(model_path, reviews_file="reviews.txt", labels_file="labels.txt", batch_size=128, seq_len=400):
    # === 1. 加载模型与配置 ===
    checkpoint = torch.load(model_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    config = checkpoint['config']
    word_int = checkpoint['vocab']

    model = SentimentImproved(
        input_size=config['input_size'],
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        output_size=config['output_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    print(f"模型已加载到 {device}")

    # === 2. 加载全部训练文本与标签 ===
    with open(reviews_file, "r", encoding="utf-8") as f:
        texts = f.read().split('\n')
    with open(labels_file, "r", encoding="utf-8") as f:
        labels = f.read().split('\n')

    # 清洗文本并转为索引序列
    processed = []
    for text in texts:
        if text.strip():
            clean = clean_text(text)
            clean = remove_stopwords(clean)
            stemmed = stemming(clean)
            processed.append(stemmed)

    label_int = np.array([1 if x == "positive" else 0 for x in labels[:len(processed)]])
    print(f"加载了 {len(processed)} 条样本")

    # === 3. 将文本映射为整数序列 ===
    all_text_ints = []
    for text in processed:
        seq = []
        for word in text.split():
            seq.append(word_int.get(word, 0))
        all_text_ints.append(seq)

    # 填充到相同长度
    dataset = reset_text(all_text_ints, seq_len=seq_len)
    dataset_tensor = torch.from_numpy(dataset)
    label_tensor = torch.from_numpy(label_int)

    # 创建 DataLoader
    full_dataset = TensorDataset(dataset_tensor, label_tensor)
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)

    # === 4. 定义损失函数并计算性能 ===
    from v3 import FocalLoss
    criterion = FocalLoss(alpha=0.25, gamma=2)

    metrics = test_model_improved(
        model=model,
        data_loader=full_loader,
        device=device,
        criterion=criterion,
        batch_size=batch_size
    )

    print("\n==== 在全部训练数据上的最终评估结果 ====")
    for k, v in metrics.items():
        print(f"{k}: {v:.5f}")


if __name__ == "__main__":
    evaluate_on_training_data("sentiment_model_optimized.pth")
