# -*- coding: utf-8 -*-
"""
用训练好的BERT模型测试"羽绒服退货"工单（RF预测错误的那条），看看BERT能否纠正
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '_04_bert_base'))

from _04_bert_base._05_bert_predict_fun import load_model, predict_fun

# 加载训练好的BERT模型（best F1=75.53%）
print('正在加载BERT模型，请稍候...')
model = load_model()
print('模型加载完成！\n')

# RF当时分错的工单：羽绒服掉毛要求退货（RF预测成了"其他"，正确应为"市场监管"）
text = "我去年冬天在商场买了一件羽绒服，最近拿出来穿发现严重掉毛，去找商家要求退货，商家说已经过了三包期限不给退，我该怎么办"

# 可以在这里换成你想测的任何一条工单：
# text = "..."

data = {"text": text}
result = predict_fun(data, model)

print('=' * 50)
print('工单内容:', text)
print('-' * 50)
print('BERT预测类别:', result['predict_class'], '(类别编号:', result['predict_index'], ')')
print('=' * 50)
