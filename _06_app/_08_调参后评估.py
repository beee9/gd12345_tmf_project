# -*- coding: utf-8 -*-
"""
调参后模型测试集评估：RF(调参版) vs FastText(调参版) vs 调参前
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '_01_data'))

import pandas as pd
import jieba
import pickle
import fasttext
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from _01_data._01_config import Config
config = Config()

# ========== 加载测试集 ==========
df_test = pd.read_csv(config.test_path, sep='\t', names=['text', 'label'], engine='python')
y_test = df_test['label'].values
# 12345数据含换行符：FastText单行预测需去除
texts_clean = [str(t).replace('\n', '').replace('\r', '') for t in df_test['text']]
# 分词（RF的TFIDF基于分词文本）
texts_jieba = [' '.join(jieba.cut(t)) for t in texts_clean]

print(f'测试集: {len(df_test)} 条')
print('=' * 60)

# ========== 1. RF 调参版评估 ==========
with open(config.tfidf_save_path, 'rb') as f:
    tfidf = pickle.load(f)
with open(config.rf_save_model_path, 'rb') as f:
    rf_model = pickle.load(f)

X_test = tfidf.transform(texts_jieba)
y_pred_rf = rf_model.predict(X_test)
acc = accuracy_score(y_test, y_pred_rf)
pre = precision_score(y_test, y_pred_rf, average='macro', zero_division=0)
rec = recall_score(y_test, y_pred_rf, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred_rf, average='macro', zero_division=0)
print(f'【RF调参版】准确率:{acc:.4f} 精确率:{pre:.4f} 召回率:{rec:.4f} F1:{f1:.4f}')

# ========== 2. FastText 调参版评估 ==========
ft_auto = fasttext.load_model(config.ft_word_auto_model_path)
y_pred_ft = [config.class2id[ft_auto.predict(t)[0][0].replace('__label__', '')] for t in texts_jieba]
acc = accuracy_score(y_test, y_pred_ft)
pre = precision_score(y_test, y_pred_ft, average='macro', zero_division=0)
rec = recall_score(y_test, y_pred_ft, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred_ft, average='macro', zero_division=0)
print(f'【FT调参版】准确率:{acc:.4f} 精确率:{pre:.4f} 召回率:{rec:.4f} F1:{f1:.4f}')

# ========== 3. FastText 旧版（default）评估 ==========
ft_default = fasttext.load_model(config.ft_word_default_model_path)
y_pred_ftd = [config.class2id[ft_default.predict(t)[0][0].replace('__label__', '')] for t in texts_jieba]
acc = accuracy_score(y_test, y_pred_ftd)
pre = precision_score(y_test, y_pred_ftd, average='macro', zero_division=0)
rec = recall_score(y_test, y_pred_ftd, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred_ftd, average='macro', zero_division=0)
print(f'【FT旧版】 准确率:{acc:.4f} 精确率:{pre:.4f} 召回率:{rec:.4f} F1:{f1:.4f}')

print('=' * 60)
print('调参前成绩参考: RF F1=74.57% | FastText F1=75.76% | BERT F1=82.03%')
