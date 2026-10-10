# -*- coding: utf-8 -*-
"""
12345工单打标项目 - LLM配合提示词模型评估（在测试集上评估）
"""
import os
import pandas as pd
from openai import OpenAI
from _01_config import Config
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

config = Config()

# 初始化客户端
client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com")

# 读取提示词
SYSTEM_PROMPT = open(config.root_path + '_05_llm_prompt/提示词.txt', 'r', encoding='utf8').read()


# todo 加载原始数据
def load_data(base_path):
    df = pd.read_csv(base_path, sep='\t', names=['text', 'label'])
    texts = [str(t) for t in df['text']]
    labels = [int(l) for l in df['label']]
    return texts, labels


def predict_fun(q):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q}
        ],
        temperature=0.1
    )
    y_pred_class = response.choices[0].message.content.strip()
    # 返回对应索引（找不到类别时默认返回"其他"）
    return config.class2id.get(y_pred_class, config.class2id['其他'])


def llm_prompt_eval(path):
    # 获取所有的工单文本和真实标签
    texts, all_true_labels = load_data(path)
    # 提前创建pred_labels列表,存储预测结果
    all_pred_labels = []
    for q in tqdm(texts):
        y_pred_label = predict_fun(q)
        all_pred_labels.append(y_pred_label)
    # todo 计算准确率,精确率,召回率,f1分数
    acc = accuracy_score(all_true_labels, all_pred_labels)
    pre = precision_score(all_true_labels, all_pred_labels, average='macro')
    rec = recall_score(all_true_labels, all_pred_labels, average='macro')
    f1 = f1_score(all_true_labels, all_pred_labels, average='macro')
    print(f'准确率:{acc},精确率:{pre},召回率:{rec},f1:{f1}')


if __name__ == '__main__':
    # 先用一个小文件快速验证，再评估全量测试集
    llm_prompt_eval(config.test_path)
