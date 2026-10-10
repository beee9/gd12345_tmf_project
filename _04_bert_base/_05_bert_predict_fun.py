# -*- coding: utf-8 -*-
"""
12345工单打标项目 - BERT模型预测函数（供Web服务调用）
"""
# 导包
from _01_config import Config
from _03_bert_classifier_model import MyBertClassifier
import torch

# 1个配置文件
config = Config()


# 加载模型
def load_model():
    my_bert_model = MyBertClassifier()
    my_bert_model.load_state_dict(torch.load(config.bert_classifier_model_save_path, map_location=config.device))
    my_bert_model.to(config.device)
    my_bert_model.eval()
    return my_bert_model


# 定义API接口函数
def predict_fun(data, my_bert_model):
    # todo 1. json数据中获取文本,然后使用tokenizer处理为特征张量
    batch_texts_tensor = config.bert_tokenizer(data['text'], max_length=config.max_len, padding='max_length',
                                               truncation=True, return_tensors='pt')
    batch_texts_tensor = {k: v.to(config.device) for k, v in batch_texts_tensor.items()}
    # todo 2. bert模型直接预测
    logits = my_bert_model(batch_texts_tensor)
    y_pred_idx = torch.argmax(logits, dim=-1).item()
    # todo 3. 根据索引获取类别名
    id2class = {index: line.strip() for index, line in enumerate(open(config.class_path, 'r', encoding='utf8'))}
    y_pred_class = id2class[y_pred_idx]
    # todo 4. 拼接到data中返回
    data['predict_class'] = y_pred_class
    data['predict_index'] = int(y_pred_idx)
    # 返回
    return data


if __name__ == '__main__':
    my_bert_model = load_model()
    text = input('请您输入一条工单内容:')
    data = {"text": text}
    result = predict_fun(data, my_bert_model)
    print(result)
