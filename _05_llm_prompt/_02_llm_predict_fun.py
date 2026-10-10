# -*- coding: utf-8 -*-
"""
12345工单打标项目 - LLM预测函数（DeepSeek）
"""
import os
from openai import OpenAI
from _01_config import Config

# 初始化客户对象
config = Config()
client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),  # 通过环境变量读取API Key
    base_url="https://api.deepseek.com")

# 读取提示词
SYSTEM_PROMPT = open(config.root_path + '_05_llm_prompt/提示词.txt', 'r', encoding='utf8').read()


def predict_fun(data):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": data['text']},
        ],
        stream=False,
        temperature=0.1
    )
    y_pred_class = response.choices[0].message.content.strip()

    # 拼接到data中并返回
    data['predict_class'] = y_pred_class

    return data


if __name__ == '__main__':
    # 模拟页面传递过来json数据
    text = input('请您输入一条工单内容:')
    data = {"text": text}
    # 模拟调用API
    result = predict_fun(data)
    print(result)
