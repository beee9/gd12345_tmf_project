# -*- coding: utf-8 -*-
"""
12345工单打标项目 - 随机森林模型评估（在真实测试集上评估）
"""
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from _01_config import Config

# TODO 1.提前创建配置对象
config = Config()

# TODO 2.读取测试集分词后数据,同时获取特征和标签
df_data = pd.read_csv(config.process_test_path, sep='\t')
x_words = df_data['words']
y_label = df_data['label']

# TODO 3.加载tfidf然后把文本数据转为数值特征
with open(config.tfidf_save_path, 'rb') as f:
    tfidf = pickle.load(f)
new_x_words = tfidf.transform(x_words)

# TODO 4.加载rf模型然后预测
with open(config.rf_save_model_path, 'rb') as f:
    model = pickle.load(f)
y_pred = model.predict(new_x_words)

# TODO 5.评估
print('==========真实测试集评估结果==========')
print(f"准确率:{accuracy_score(y_label, y_pred)}")
print(f"精确率:{precision_score(y_label, y_pred, average='macro')}")
print(f"召回率:{recall_score(y_label, y_pred, average='macro')}")
print(f"f1分数:{f1_score(y_label, y_pred, average='macro')}")
