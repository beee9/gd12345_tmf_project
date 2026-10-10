# -*- coding: utf-8 -*-
"""
BERT最终模型测试集评估 + 羽绒服工单复测
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _01_config import Config
from _02_dataloader_utils import build_all_dataloader
from _03_bert_classifier_model import MyBertClassifier
from _04_bert_model_eval_utils import model_eval
import torch

config = Config()

# 1. 加载最终保存的模型
print('加载最终模型...')
my_bert_model = MyBertClassifier()
my_bert_model.load_state_dict(torch.load(config.bert_classifier_model_save_path, map_location=config.device))
my_bert_model.to(config.device)

# 2. 测试集评估
train_dataloader, dev_dataloader, test_dataloader = build_all_dataloader()
print('评估测试集...')
acc, pre, rec, f1 = model_eval(test_dataloader, my_bert_model)
print('=' * 50)
print('测试集最终成绩:')
print(f'准确率:{acc} 精确率:{pre} 召回率:{rec} F1:{f1}')
print('=' * 50)

# 3. 羽绒服工单复测
text = "我去年冬天在商场买了一件羽绒服，最近拿出来穿发现严重掉毛，去找商家要求退货，商家说已经过了三包期限不给退，我该怎么办"
tokens = config.bert_tokenizer([text], max_length=config.max_len, padding='max_length',
                               truncation=True, return_tensors='pt')
tokens = {k: v.to(config.device) for k, v in tokens.items()}
my_bert_model.eval()
with torch.no_grad():
    logits = my_bert_model(tokens)
    pred_idx = torch.argmax(logits, dim=-1).item()
    print(f'羽绒服工单预测: {config.id2class[pred_idx]} (类别编号:{pred_idx})')
