# -*- coding: utf-8 -*-
"""
12345工单打标项目 - FastText 词级别 + 默认参数训练
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fasttext
from _01_data._01_config import Config

# TODO 提前创建配置对象
config = Config()

# TODO 1.模型训练（词级别，默认参数）
model = fasttext.train_supervised(input=config.process_train_path_words, verbose=2)

# TODO 2.模型保存
os.makedirs(os.path.dirname(config.ft_word_default_model_path), exist_ok=True)
model.save_model(config.ft_word_default_model_path)
print('模型保存成功!')
print('===============================================================================')

# TODO 4.模型评估
result = model.test(config.process_test_path_words)
print(f"模型评估结果是:{result}")  # (数据个数, 精确率, 召回率)
print('===============================================================================')

# TODO 5.模型预测示例
text = '你好，我想咨询一下养老保险怎么补缴'
label = model.predict(" ".join(__import__('jieba').lcut(text)))
print(f'模型预测结果是:{label}')
