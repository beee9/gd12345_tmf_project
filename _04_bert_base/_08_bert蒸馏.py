# -*- coding: utf-8 -*-
"""
知识蒸馏 (Knowledge Distillation)
教师: BERT微调版 (F1=82.03%, 409MB)
学生: 2层小BERT (hidden=128, 约15MB)
训练: 蒸馏损失 = alpha*KL(学生,教师软标签) + (1-alpha)*CE(学生,硬标签)
"""
import sys
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _01_config import Config
from _02_dataloader_utils import build_all_dataloader
from _03_bert_classifier_model import MyBertClassifier
from _04_bert_model_eval_utils import model_eval

config = Config()

# ========== 1. 学生模型：2层小BERT ==========
class SmallBertClassifier(nn.Module):
    def __init__(self, hidden=128, layers=2, heads=4, intermediate=512):
        super().__init__()
        student_config = transformers.BertConfig(
            vocab_size=config.bert_config.vocab_size,
            hidden_size=hidden,
            num_hidden_layers=layers,
            num_attention_heads=heads,
            intermediate_size=intermediate,
            max_position_embeddings=config.max_len + 2,
        )
        self.bert = transformers.BertModel(student_config)
        self.linear = nn.Linear(hidden, config.class_num)

    def forward(self, x):
        result = self.bert(**x)
        return self.linear(result['pooler_output'])


import transformers  # 上面类定义用到了


# ========== 2. 数据集：缓存教师软标签 ==========
def prepare_soft_labels(num_samples=3000):
    """用教师BERT对训练集前num_samples条生成软标签并缓存"""
    cache_path = os.path.join(os.path.dirname(config.bert_classifier_model_save_path),
                              'teacher_soft_labels.pt')
    if os.path.exists(cache_path):
        print(f'加载缓存教师软标签: {cache_path}')
        data = torch.load(cache_path, map_location='cpu')
        return data['texts'], data['labels'], data['teacher_logits']

    print(f'教师BERT生成软标签（{num_samples}条, 预计2~3分钟）...')
    # 加载教师
    teacher = MyBertClassifier()
    teacher.load_state_dict(torch.load(config.bert_classifier_model_save_path, map_location=config.device))
    teacher.to(config.device)
    teacher.eval()

    # 读训练集
    df = pd.read_csv(config.train_path, sep='\t', names=['text', 'label'], engine='python')
    df = df.head(num_samples)
    texts = df['text'].tolist()
    labels = df['label'].tolist()

    all_logits = []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            tokens = config.bert_tokenizer(batch_texts, max_length=config.max_len,
                                           padding='max_length', truncation=True,
                                           return_tensors='pt')
            tokens = {k: v.to(config.device) for k, v in tokens.items()}
            logits = teacher(tokens).cpu()
            all_logits.append(logits)
    teacher_logits = torch.cat(all_logits, dim=0)
    print(f'软标签生成完成: {teacher_logits.shape}')

    torch.save({'texts': texts, 'labels': labels, 'teacher_logits': teacher_logits}, cache_path)
    print(f'已缓存到: {cache_path}')
    return texts, labels, teacher_logits


# ========== 3. 蒸馏训练 ==========
def distill_train(teacher_logits, labels, epochs=8, temperature=3.0, alpha=0.5):
    """学生模型学习教师的软标签
    注意: 学生模型随机初始化，需用较大学习率(预训练微调用小lr，随机初始化要大lr)
    """
    print(f'开始蒸馏训练: epochs={epochs}, temperature={temperature}, alpha={alpha}')
    student = SmallBertClassifier()
    student.to(config.device)

    # 数据分批
    n = len(labels)
    logits_t = teacher_logits.to(config.device)
    labels_t = torch.tensor(labels).to(config.device)

    optimizer = torch.optim.AdamW(student.parameters(), lr=5e-4)
    batch_size = 32
    n_batches = (n + batch_size - 1) // batch_size

    for epoch in range(epochs):
        student.train()
        total_loss = 0
        t0 = time.time()
        for i in range(0, n, batch_size):
            end = min(i + batch_size, n)
            texts_batch = texts[i:end]
            # tokenize
            tokens = config.bert_tokenizer(texts_batch, max_length=config.max_len,
                                           padding='max_length', truncation=True,
                                           return_tensors='pt')
            tokens = {k: v.to(config.device) for k, v in tokens.items()}

            # 学生logits
            student_logits = student(tokens)
            # 教师软标签
            teacher_soft = F.softmax(logits_t[i:end] / temperature, dim=-1)
            # 蒸馏损失: KL散度
            student_soft = F.log_softmax(student_logits / temperature, dim=-1)
            kl_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (temperature ** 2)
            # 硬标签损失: 交叉熵
            ce_loss = F.cross_entropy(student_logits, labels_t[i:end])
            # 加权
            loss = alpha * kl_loss + (1 - alpha) * ce_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / n_batches
        print(f'Epoch {epoch+1}/{epochs} 损失:{avg_loss:.4f} 耗时:{time.time()-t0:.1f}秒')

    return student


# ========== 4. 主流程 ==========
if __name__ == '__main__':
    # 生成教师软标签（训练集前3000条）
    texts, labels, teacher_logits = prepare_soft_labels(num_samples=3000)

    # 蒸馏训练学生
    student = distill_train(teacher_logits, labels, epochs=8)

    # ===== 评估对比 =====
    train_dataloader, dev_dataloader, test_dataloader = build_all_dataloader()

    # 学生模型在CPU评估（小模型快，且不用来回切GPU）
    student.eval()
    print('评估学生模型(测试集)...')
    acc_s, pre_s, rec_s, f1_s = model_eval(test_dataloader, student.to('cpu'), device='cpu')

    # 学生模型大小
    os.makedirs(os.path.join(os.path.dirname(config.bert_classifier_model_save_path), 'distill'), exist_ok=True)
    save_path = os.path.join(os.path.dirname(config.bert_classifier_model_save_path), 'distill', 'student_bert_distill.pth')
    torch.save(student.state_dict(), save_path)
    size_mb = os.path.getsize(save_path) / 1024 / 1024

    print('=' * 55)
    print('【知识蒸馏结果】')
    print(f'教师模型(bert-base): F1≈0.8203, 大小≈409MB')
    print(f'学生模型(2层小BERT): F1={f1_s:.4f}, 大小≈{size_mb:.1f}MB')
    print(f'模型缩小: {409/size_mb:.1f}倍 | F1保留: {f1_s/0.8203*100:.1f}%')
    print('=' * 55)
