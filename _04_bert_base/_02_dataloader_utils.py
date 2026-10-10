# -*- coding: utf-8 -*-
"""
12345工单打标项目 - BERT数据加载工具
"""
# 导包
# 注意：必须先 import pandas 再 import torch/transformers！
# (torch 2.4 与 pandas 3.x 存在 DLL 加载顺序冲突, 反序会 0xC0000005 原生崩溃)
import pandas as pd
import torch
from _01_config import Config

# todo 提前创建配置对象
config = Config()


# todo 加载原始数据
def load_data_list(base_path):
    """
    :param base_path: train.txt或者dev.txt或者test.txt的路径
    原始数据:  工单文本\t标签
    注意：工单文本可能含换行符，必须用pandas读取
    :return: [(工单文本,标签),(工单文本,标签)...]
    """
    df = pd.read_csv(base_path, sep='\t', names=['text', 'label'])
    data_list = []
    for text, label in zip(df['text'], df['label']):
        data_list.append((str(text), int(label)))
    return data_list


# todo 自定义dataset类: 1个继承3个重写
class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]


# todo 自定义collate_fn函数
def my_collate_fn(batch):
    """
    :param batch: 一批的[(工单文本,标签),(工单文本,标签)...]
    :return: 这一批的tokenizer处理后的特征数据和标签
    """
    # todo 单独获取这一批数据中的所有texts和labels
    texts, labels = zip(*batch)
    # todo 先把labels转换为张量
    batch_labels_tensor = torch.tensor(labels)
    # todo 再用tokenizer处理texts,最后转换为张量
    batch_texts_tensor = config.bert_tokenizer(texts, max_length=config.max_len, padding='max_length', truncation=True,
                                               return_tensors='pt')
    # todo 返回结果
    return batch_texts_tensor, batch_labels_tensor


# TODO 提前构建训练/验证/测试3个dataloader
def build_all_dataloader():
    # 1.构建train_dataloader
    data_list = load_data_list(config.train_path)
    data_set = MyDataSet(data_list)
    train_dataloader = torch.utils.data.DataLoader(data_set, config.batch_size, shuffle=True, collate_fn=my_collate_fn)
    # 2.构建dev_dataloader
    data_list = load_data_list(config.dev_path)
    data_set = MyDataSet(data_list)
    dev_dataloader = torch.utils.data.DataLoader(data_set, config.batch_size, shuffle=False, collate_fn=my_collate_fn)
    # 3.构建test_dataloader
    data_list = load_data_list(config.test_path)
    data_set = MyDataSet(data_list)
    test_dataloader = torch.utils.data.DataLoader(data_set, config.batch_size, shuffle=False, collate_fn=my_collate_fn)
    # 4.返回
    return train_dataloader, dev_dataloader, test_dataloader


if __name__ == '__main__':
    train_dataloader, dev_dataloader, test_dataloader = build_all_dataloader()
    print(f"训练集按照每批{config.batch_size}条,一共{len(train_dataloader)}批")
    print(f"验证集按照每批{config.batch_size}条,一共{len(dev_dataloader)}批")
    print(f"测试集按照每批{config.batch_size}条,一共{len(test_dataloader)}批")
