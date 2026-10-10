# -*- coding: utf-8 -*-
"""
LLM few-shot + CoT + 候选打分 联合评估（210条，约5分钟）
在CoT+打分基础上，提示词里加入每类1条真实工单示例
"""
import os
import sys
import re
import random
import pandas as pd
from openai import OpenAI
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '_01_data'))
from _01_data._01_config import Config

config = Config()
client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com")
random.seed(42)

# ========== 1. 从训练集每类抽1条真实样本做few-shot ==========
df_train = pd.read_csv(config.train_path, sep='\t', names=['text', 'label'], engine='python')
df_train['text'] = df_train['text'].str.replace('\n', ' ').str.replace('\r', ' ')
fewshot_lines = []
for cls in config.class2id.keys():
    cls_samples = df_train[df_train['label'] == config.class2id[cls]]['text'].tolist()
    if cls_samples:
        sample = random.choice(cls_samples)
        fewshot_lines.append(f'例："{sample[:80]}..." → {cls}')
fewshot_str = '\n'.join(fewshot_lines)
print(f'few-shot示例: {len(fewshot_lines)} 条（14类 × 1条）')

# ========== 2. 联合提示词（few-shot + CoT + 打分） ==========
COT_FS_PROMPT = """你是12345政务热线工单分类专家。工单必须归入以下14类之一：
科教文体、公卫医疗、城乡建设、自然资源与环境保护、交通运输、农林牧渔、公共安全、人力资源社会保障、民政社区、市场监管、经济财贸、司法行政、政风行风党风、其他

【真实工单参考示例】
""" + fewshot_str + """

【处理步骤】
第一步(推理)：分析工单中的关键信息，判断涉及哪个领域、哪类政府职责；
第二步(打分)：对以下14个类别逐一给出0-100的匹配度分数，只输出"类别=分数"格式，每行一个；
第三步(结论)：输出匹配度最高的类别名称（参考上面的示例判断标准）。

输出格式必须严格为：
推理：<简要推理，不超过50字>
打分：<类别=分数>（14行）
结论：<类别名称>

工单："""


def parse_response(resp):
    """解析模型输出：优先取'结论'后的类别，取不到则从打分中找最高分"""
    m = re.search(r'结论[：:]\s*([\u4e00-\u9fa5]+)', resp)
    if m and m.group(1) in config.class2id:
        return m.group(1)
    best_cls, best_score = None, -1
    for line in resp.splitlines():
        mm = re.match(r'([\u4e00-\u9fa5]+)\s*=\s*(\d+)', line.strip())
        if mm and mm.group(1) in config.class2id:
            s = int(mm.group(2))
            if s > best_score:
                best_cls, best_score = mm.group(1), s
    if best_cls:
        return best_cls
    return '其他'


# ========== 3. 分层抽210条（与实验2同规模，直接可比） ==========
df_test = pd.read_csv(config.test_path, sep='\t', names=['text', 'label'], engine='python')
df_test['text'] = df_test['text'].str.replace('\n', ' ').str.replace('\r', ' ')

sample_per_class = 15
sampled = []
for cls in config.class2id.keys():
    cls_test = df_test[df_test['label'] == config.class2id[cls]]
    n = min(sample_per_class, len(cls_test))
    sampled.append(cls_test.sample(n, random_state=42))
df_sample = pd.concat(sampled).sample(frac=1, random_state=42)
y_test = df_sample['label'].values
print(f'评估集: {len(df_sample)} 条')

# ========== 4. 预测 ==========
y_pred = []
for i, text in enumerate(df_sample['text'].tolist()):
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": COT_FS_PROMPT},
                {"role": "user", "content": text},
            ],
            stream=False,
            temperature=0.1,
            max_tokens=400,
        )
        resp = response.choices[0].message.content.strip()
        pred_cls = parse_response(resp)
        y_pred.append(config.class2id.get(pred_cls, config.class2id['其他']))
    except Exception as e:
        print(f'[{i}] 错误: {e}')
        y_pred.append(config.class2id['其他'])
    if (i + 1) % 50 == 0:
        print(f'进度: {i+1}/{len(df_sample)}')

# ========== 5. 评估 ==========
acc = accuracy_score(y_test, y_pred)
pre = precision_score(y_test, y_pred, average='macro', zero_division=0)
rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
print('=' * 55)
print(f'【LLM few-shot+CoT+打分】准确率:{acc:.4f} 精确率:{pre:.4f} 召回率:{rec:.4f} F1:{f1:.4f}')
print(f'评估条数: {len(df_sample)}')
print('=' * 55)
print('对照: zero-shot F1=0.4763 | few-shot F1=0.4879 | CoT+打分 F1=0.5109')
