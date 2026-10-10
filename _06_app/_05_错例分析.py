# -*- coding: utf-8 -*-
"""
分析工单分类失败原因：看看"三包/退货/掉毛"等词在各类别中的分布
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import jieba
from _01_data._01_config import Config

config = Config()
df = pd.read_csv(config.train_path, sep='\t', names=['text', 'label'])
df['类别'] = df['label'].map(config.id2class)

# 要调查的关键词（用户这条工单的核心词）
keywords = ['三包', '退货', '掉毛', '羽绒服', '商场', '不给退', '退']

print('=' * 60)
print(f'训练集共 {len(df)} 条，各关键词在各类别中的分布：')
print('=' * 60)
for kw in keywords:
    hit = df[df['text'].str.contains(kw, na=False)]
    if len(hit) == 0:
        print(f'\n【{kw}】训练集中出现 0 次 —— 模型从未见过这个词！')
        continue
    print(f'\n【{kw}】出现 {len(hit)} 次，分布如下：')
    top = hit['类别'].value_counts().head(5)
    for cls, cnt in top.items():
        print(f'   {cls}: {cnt}次 ({cnt/len(hit)*100:.0f}%)')

# 看看"其他"类里都有什么
print('\n' + '=' * 60)
other = df[df['label'] == 13]
print(f'"其他"类共 {len(other)} 条，抽样看5条内容：')
for t in other['text'].head(5):
    print(f'   - {t[:50]}')
