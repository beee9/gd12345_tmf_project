# -*- coding: utf-8 -*-
"""
12345工单打标项目 - BERT模块配置
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import transformers
from _01_data._01_config import Config as GlobalConfig


class Config(GlobalConfig):
    def __init__(self):
        super().__init__()
        # 提前加载tokenizer和bert模型对象
        self.bert_tokenizer = transformers.BertTokenizer.from_pretrained(self.bert_base_chinese_path)
        self.bert_model = transformers.BertModel.from_pretrained(self.bert_base_chinese_path)
        # 提前加载config文件对象
        self.bert_config = transformers.BertConfig.from_pretrained(self.bert_base_chinese_path)
        # 设备：优先cuda，其次mps（mac），最后cpu
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        # bert训练参数（工单文本较长，max_len设置128）
        self.epochs = 3
        # RTX 3060 Laptop 6GB显存：batch=16时最快（0.2秒/批），batch>=32会显存溢出变慢几十倍
        self.batch_size = 16
        self.lr = 5e-5
        self.max_len = 128
        self.class_num = len(self.id2class)  # 14


if __name__ == '__main__':
    c = Config()
    print('bert模型路径:', c.bert_base_chinese_path)
    print('设备:', c.device)
    print('类别数:', c.class_num)
