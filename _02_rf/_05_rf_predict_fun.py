# -*- coding: utf-8 -*-
"""
12345工单打标项目 - 随机森林模型预测（供Web服务等调用）
"""
import jieba
import pickle
from _01_config import Config

# TODO 1.提前创建配置对象
config = Config()

# TODO 2.加载tfidf
with open(config.tfidf_save_path, 'rb') as f:
    tfidf = pickle.load(f)
# TODO 3.加载rf模型
with open(config.rf_save_model_path, 'rb') as f:
    model = pickle.load(f)


# TODO 4.定义api接口
def predict_fun(data):
    # todo 4.1 json数据中获取文本,然后直接做分词
    words = " ".join(jieba.lcut(data['text']))
    # todo 4.2 tfidf文本转数值特征
    number_words = tfidf.transform([words])
    # todo 4.3 rf模型预测类别索引
    y_pred_list = model.predict(number_words)  # [3]
    y_pred_idx = y_pred_list[0]
    # todo 4.4 根据索引获取类别名
    id2class = {index: line.strip() for index, line in enumerate(open(config.class_path, 'r', encoding='utf8'))}
    y_pred_class = id2class[y_pred_idx]

    # todo 4.5 拼接到data中并返回
    data['predict_class'] = y_pred_class
    data['predict_index'] = int(y_pred_idx)
    # 返回
    return data


if __name__ == '__main__':
    # 模拟页面传递过来json数据
    text = input('请您输入一条工单内容:')
    data = {"text": text}
    # TODO 模拟调用API
    result = predict_fun(data)
    print(result)
