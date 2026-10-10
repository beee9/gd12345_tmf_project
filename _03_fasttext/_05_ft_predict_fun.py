# -*- coding: utf-8 -*-
"""
12345工单打标项目 - FastText 预测函数（供Web服务调用）
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fasttext
import jieba
from _01_data._01_config import Config

# TODO 提前创建配置对象
config = Config()

# 加载调参后的模型（Optuna调参版，测试集F1=81.05%）
model = fasttext.load_model(config.ft_word_auto_model_path)


# TODO 定义API接口函数
def predict_fun(data):
    # todo 1. json数据中获取文本,然后直接做分词
    words = " ".join(jieba.lcut(data['text']))
    # todo 2. fasttext模型直接预测
    y_pred_tuple = model.predict(words)
    print(y_pred_tuple)
    # todo 3. 从结果中获取类别名（去掉__label__前缀）
    y_pred_class = y_pred_tuple[0][0][9:]
    # todo 4. 拼接到data中返回
    data['predict_class'] = y_pred_class
    # 返回
    return data


if __name__ == '__main__':
    text = input('请您输入一条工单内容:')
    data = {"text": text}
    result = predict_fun(data)
    print(result)
