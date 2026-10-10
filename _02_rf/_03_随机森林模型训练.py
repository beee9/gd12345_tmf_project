# -*- coding: utf-8 -*-
"""
12345工单打标项目 - 随机森林模型训练
TFIDF文本特征 -> 随机森林分类器 -> 保存模型
"""
import os
import time
import pickle
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from _01_config import Config

# TODO 1.提前创建配置对象
config = Config()

# TODO 2.读取分词后的数据文件
df_data = pd.read_csv(config.process_train_path, sep='\t')

# TODO 3.单独获取特征和标签
x_words = df_data['words']
y_labels = df_data['label']

# TODO 4.使用TFIDF把文本特征转为数值特征
# 4.1 提前把停用词读取放到列表中
stop_words = [line.strip() for line in open(config.stop_words_path, 'r', encoding='utf8')]
# 4.2 创建tfidf对象并设置停用词
tfidf = sklearn.feature_extraction.text.TfidfVectorizer(stop_words=stop_words)
# 4.3 训练文本特征转数值特征
words_feature = tfidf.fit_transform(x_words)
print(f'Tfidf特征矩阵形状: {words_feature.shape}')

# TODO 5.随机森林模型训练
# 5.1 先切割数据（训练集80%，测试集20%）
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(words_feature, y_labels, test_size=0.2,
                                                                            random_state=666)
# 5.2 创建随机森林模型
# n_estimators=100个决策树, verbose=2打印详细日志, n_jobs=-1使用本地电脑所有的处理器
model = sklearn.ensemble.RandomForestClassifier(n_estimators=100, verbose=2, n_jobs=-1)
# 5.3 模型训练
start_time = time.time()
print('开始训练了...')
model.fit(X_train, y_train)
print(f'训练结束,耗时{time.time() - start_time}秒')

# TODO 6.随机森林模型预测
y_pred = model.predict(X_test)

# TODO 7.随机森林模型评估
print('==========训练集切分测试集评估结果==========')
print(f"准确率:{accuracy_score(y_test, y_pred)}")
print(f"精确率:{precision_score(y_test, y_pred, average='macro')}")
print(f"召回率:{recall_score(y_test, y_pred, average='macro')}")
print(f"f1分数:{f1_score(y_test, y_pred, average='macro')}")

# TODO 8.保存模型
os.makedirs(os.path.dirname(config.tfidf_save_path), exist_ok=True)
with open(config.tfidf_save_path, 'wb') as f:
    pickle.dump(tfidf, f)
with open(config.rf_save_model_path, 'wb') as f:
    pickle.dump(model, f)
print('tfidf和rf模型保存成功!')
