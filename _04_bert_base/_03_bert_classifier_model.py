# -*- coding: utf-8 -*-
"""
12345工单打标项目 - BERT分类模型
"""
# 导包
import torch
from _01_config import Config

# TODO 加载配置对象
config = Config()


# TODO 自定义分类模型  1个继承2个重写
class MyBertClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 隐藏层使用bert的原始结构
        self.bert = config.bert_model
        # bert输出768维度不符合当前14分类需求,添加线性层转换,最终输出
        self.linear = torch.nn.Linear(config.bert_config.hidden_size, config.class_num)

    def forward(self, x):
        # todo 预训练好的bert前向传播
        result = self.bert(**x)
        # todo 把bert的768结果转为14维
        logits = self.linear(result['pooler_output'])
        # todo 返回结果
        return logits  # 因为后续要多分类交叉熵损失函数它底层有softmax,此处就不用softmax了


if __name__ == '__main__':
    from _02_dataloader_utils import build_all_dataloader
    # 准备数据
    train_dataloader, dev_dataloader, test_dataloader = build_all_dataloader()
    # 准备模型
    my_bert_model = MyBertClassifier()
    # 遍历dataloader的时候自动调用collate_fn处理数据
    for batch_texts_tensor, batch_labels_tensor in test_dataloader:
        # 前向传播
        logits = my_bert_model(batch_texts_tensor)
        break
