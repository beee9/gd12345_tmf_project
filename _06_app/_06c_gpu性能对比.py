# -*- coding: utf-8 -*-
"""
GPU性能对比测试：不同batch大小 vs 计算量
判断是显存交换问题 还是 显卡被限速
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '_04_bert_base'))
import torch

print('GPU:', torch.cuda.get_device_name(0), flush=True)

# 测试1：纯矩阵计算速度（对比基准）
print('\n=== 测试1: 纯GPU矩阵计算(1000x1000x100次) ===', flush=True)
a = torch.randn(1000, 1000, device='cuda')
b = torch.randn(1000, 1000, device='cuda')
torch.cuda.synchronize()
t0 = time.time()
for _ in range(100):
    c = a @ b
torch.cuda.synchronize()
print(f'耗时: {time.time()-t0:.2f}秒', flush=True)

# 测试2：加载BERT模型，不同batch对比
print('\n=== 测试2: BERT不同batch大小对比 ===', flush=True)
from _01_config import Config
config = Config()
from _03_bert_classifier_model import MyBertClassifier
model = MyBertClassifier()
model.to(config.device)
model.train()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

for bs in [4, 8, 16, 32]:
    texts = ['小区电梯坏了半个月没人修，老人上下楼很困难'] * bs
    batch = config.bert_tokenizer(texts, max_length=128, padding='max_length',
                                  truncation=True, return_tensors='pt')
    batch = {k: v.to(config.device) for k, v in batch.items()}
    labels = torch.randint(0, 14, (bs,)).to(config.device)

    # 预热
    logits = model(batch)
    loss = loss_fn(logits, labels)

    # 计时3次
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(3):
        logits = model(batch)
        loss = loss_fn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    per = (time.time() - t0) / 3
    print(f'batch={bs:2d}: 单批{per:.3f}秒, 每样本{per/bs*1000:.1f}毫秒', flush=True)
