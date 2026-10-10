# -*- coding: utf-8 -*-
"""
GPU诊断v2：逐步打印，确认BERT是否在GPU上训练
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '_04_bert_base'))

import torch
print('[1] torch版本:', torch.__version__, flush=True)
print('[2] CUDA可用:', torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print('[3] GPU名称:', torch.cuda.get_device_name(0), flush=True)

print('[4] 开始加载BERT配置(约1-2分钟)...', flush=True)
t0 = time.time()
from _01_config import Config
config = Config()
print(f'[5] 配置加载完成, device = {config.device}, 耗时{time.time()-t0:.0f}秒', flush=True)

print('[6] 创建分类模型...', flush=True)
from _03_bert_classifier_model import MyBertClassifier
model = MyBertClassifier()
model.to(config.device)
model.train()

# 检查参数设备
first_param = next(model.parameters())
bert_param = next(model.bert.parameters())
print(f'[7] 模型linear参数设备: {first_param.device}', flush=True)
print(f'[8] bert子模块参数设备: {bert_param.device}', flush=True)

print('[9] 构造测试batch...', flush=True)
tokenizer = config.bert_tokenizer
texts = ['小区电梯坏了半个月没人修，老人上下楼很困难'] * config.batch_size
batch = tokenizer(texts, max_length=config.max_len, padding='max_length', truncation=True, return_tensors='pt')
batch = {k: v.to(config.device) for k, v in batch.items()}
labels = torch.randint(0, config.class_num, (config.batch_size,)).to(config.device)
print(f'[10] batch数据设备: {next(iter(batch.values())).device}', flush=True)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

print('[11] 预热1次...', flush=True)
logits = model(batch)
loss = loss_fn(logits, labels)

print('[12] 正式测5次速度...', flush=True)
start = time.time()
for i in range(5):
    logits = model(batch)
    loss = loss_fn(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
elapsed = (time.time() - start) / 5
print(f'[13] 单batch训练耗时: {elapsed:.3f}秒', flush=True)
print(f'[14] 估算每轮(396批): {elapsed*396/60:.1f}分钟, 3轮总: {elapsed*396*3/60:.1f}分钟', flush=True)
