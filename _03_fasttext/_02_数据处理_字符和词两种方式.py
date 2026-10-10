# -*- coding: utf-8 -*-
"""
12345工单打标项目 - FastText 数据处理
生成 fasttext 训练格式：__label__类别名 分词后的文本
注意：工单文本中可能含换行符，必须用 pandas 读取，不能逐行 split
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import jieba
from _01_data._01_config import Config

# 提前创建配置对象
config = Config()


def process_data(base_path, process_path, is_jieba):
    """生成fasttext格式数据：__label__类别名 文本内容"""
    os.makedirs(os.path.dirname(process_path), exist_ok=True)
    df = pd.read_csv(base_path, sep='\t', names=['text', 'label'])
    with open(process_path, 'w', encoding='utf8') as fw:
        for text, label in zip(df['text'], df['label']):
            # 清洗文本中的换行符和制表符
            text = str(text).replace('\n', '').replace('\r', '').replace('\t', '')
            label_name = config.id2class[int(label)]
            if is_jieba:
                words = " ".join(jieba.lcut(text))
                ft_line = '__label__' + label_name + " " + words + '\n'
            else:
                chars = " ".join(list(text))
                ft_line = '__label__' + label_name + " " + chars + '\n'
            fw.write(ft_line)


if __name__ == '__main__':
    print('生成字符级别数据...')
    process_data(config.train_path, config.process_train_path_chars, is_jieba=False)
    process_data(config.test_path, config.process_test_path_chars, is_jieba=False)
    process_data(config.dev_path, config.process_dev_path_chars, is_jieba=False)
    print('生成词级别数据...')
    process_data(config.train_path, config.process_train_path_words, is_jieba=True)
    process_data(config.test_path, config.process_test_path_words, is_jieba=True)
    process_data(config.dev_path, config.process_dev_path_words, is_jieba=True)
    print('全部数据处理完成!')
