# -*- coding: utf-8 -*-
"""
12345工单打标项目 - 数据分词并存储
使用jieba对原始文本分词，供随机森林使用
"""
import os
import pandas as pd
import jieba
from _01_config import Config

# 提前创建Config对象
config = Config()


def process_data(base_path, process_path):
    """读取原始数据 -> jieba分词 -> 保存到processed_data目录"""
    # todo 1.读取数据（txt格式：text\tlabel）
    df_data = pd.read_csv(base_path, sep='\t', names=['text', 'label'])
    # todo 2.apply()对每一行进行分词并存储为一列
    df_data['words'] = df_data['text'].apply(lambda x: " ".join(jieba.lcut(x)))
    # todo 3.保证目录存在
    os.makedirs(os.path.dirname(process_path), exist_ok=True)
    # todo 4.保存数据
    df_data.to_csv(process_path, sep='\t', index=False, header=True)
    print(f'处理完成: {process_path}, 共{len(df_data)}条')


if __name__ == '__main__':
    # 处理训练数据
    process_data(config.train_path, config.process_train_path)
    # 处理验证数据
    process_data(config.dev_path, config.process_dev_path)
    # 处理测试数据
    process_data(config.test_path, config.process_test_path)
