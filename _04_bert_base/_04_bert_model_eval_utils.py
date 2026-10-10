# -*- coding: utf-8 -*-
"""
12345工单打标项目 - BERT模型评估工具
"""
# 导包
import torch
from _01_config import Config
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1个配置文件
config = Config()


def model_eval(dev_dataloader, my_bert_model, device=None):
    # 2个准备
    # 数据和模型传参
    if device is None:
        device = config.device
    # 准备模型 评估模式
    my_bert_model.eval()
    with torch.no_grad():
        # 提前创建两个列表,用于存储所有的样本的预测和真实标签
        all_pred_labels, all_true_labels = [], []
        # 1个遍历
        # 内层循环控制批次
        for index, (batch_texts_tensor, batch_labels_tensor) in enumerate(tqdm(dev_dataloader), start=1):
            # TODO 把数据放到指定设备上
            batch_labels_tensor = batch_labels_tensor.to(device)
            batch_texts_tensor = {k: v.to(device) for k, v in batch_texts_tensor.items()}
            # 2个核心
            # 前向传播
            logits = my_bert_model(batch_texts_tensor)
            # 累加每批的预测标签和真实标签
            all_true_labels.extend(batch_labels_tensor.tolist())  # 累加真实的这批标签
            batch_pred_labels = torch.argmax(logits, dim=-1)  # 最大分数对应的索引就是预测的标签
            all_pred_labels.extend(batch_pred_labels.tolist())  # 累加预测的这批标签
        # 计算准确率,精确率,召回率,f1分数
        acc = accuracy_score(all_true_labels, all_pred_labels)
        pre = precision_score(all_true_labels, all_pred_labels, average='macro')
        rec = recall_score(all_true_labels, all_pred_labels, average='macro')
        f1 = f1_score(all_true_labels, all_pred_labels, average='macro')
        # 返回方便打印日志
        return acc, pre, rec, f1
