# -*- coding: utf-8 -*-
"""
12345工单打标项目 - 数据分析
统计训练/验证/测试集样本数、类别分布、文本长度等
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import matplotlib.pyplot as plt
from _01_data._01_config import Config

plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文乱码
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示

config = Config()


def analyze(path, name):
    df = pd.read_csv(path, sep='\t', names=['text', 'label'])
    print(f'====={name}数据集=====')
    print(f'样本总数: {len(df)}')
    print(f'文本长度分布: 平均{df["text"].str.len().mean():.1f}字, 最大{df["text"].str.len().max()}, 最小{df["text"].str.len().min()}')
    print('类别分布:')
    label_counts = df['label'].value_counts().sort_index()
    for label, count in label_counts.items():
        print(f'  {label}. {config.id2class[label]}: {count}条 ({count/len(df)*100:.2f}%)')
    return df


if __name__ == '__main__':
    df_train = analyze(config.train_path, '训练')
    df_dev = analyze(config.dev_path, '验证')
    df_test = analyze(config.test_path, '测试')

    # 绘制类别分布图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (name, df) in zip(axes, [('训练集', df_train), ('验证集', df_dev), ('测试集', df_test)]):
        counts = df['label'].value_counts().sort_index()
        ax.bar(range(len(counts)), counts.values)
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels([config.id2class[i] for i in counts.index], rotation=45, ha='right', fontsize=8)
        ax.set_title(name)
        ax.set_xlabel('类别')
        ax.set_ylabel('样本数')
    plt.tight_layout()
    os.makedirs(config.root_path + '文档', exist_ok=True)
    plt.savefig(config.root_path + '文档/类别分布图.png', dpi=150)
    print('类别分布图已保存到 文档/类别分布图.png')
