# -*- coding: utf-8 -*-
"""
LLM few-shot 轻量评估（方案B：只跑few-shot，300条，约3.5分钟）
- few-shot示例：从训练集每类抽2条真实样本嵌入提示词
- 评估集：测试集抽300条（每类约21条，保持分布）
"""
import os
import sys
import random
import pandas as pd
import numpy as np
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

# ========== 1. 从训练集每类抽2条真实样本做few-shot示例 ==========
df_train = pd.read_csv(config.train_path, sep='\t', names=['text', 'label'], engine='python')
df_train['text'] = df_train['text'].str.replace('\n', ' ').str.replace('\r', ' ')
fewshot_examples = []
for cls in config.class2id.keys():
    cls_samples = df_train[df_train['label'] == config.class2id[cls]]['text'].tolist()
    if len(cls_samples) >= 2:
        picked = random.sample(cls_samples, 2)
        fewshot_examples.append(f'输入："{picked[0]}" -> 输出：{cls}')
        fewshot_examples.append(f'输入："{picked[1]}" -> 输出：{cls}')

fewshot_str = '\n'.join(fewshot_examples)
print(f'few-shot示例: {len(fewshot_examples)} 条（14类 × 2条）')

# ========== 2. 基础提示词（复用原提示词，但Few-shot部分替换为真实样本） ==========
base_prompt = """Role: 你是一名12345政务服务便民热线的工单分类专家，你的核心任务是根据工单内容，精准判断其所属的唯一类别。
Constraints & Output Format: 输出要求：你必须仅且仅返回以下14个中文类别之一，必须有一个类别，如果实在找不到合适的，可以默认返回"其他"，不得包含任何解释、标点符号或换行符。
可选类别：科教文体、公卫医疗、城乡建设、自然资源与环境保护、交通运输、农林牧渔、公共安全、人力资源社会保障、民政社区、市场监管、经济财贸、司法行政、政风行风党风、其他

Category Definition Guidelines (分类判定标准):
科教文体 (科学技术/教育/文化/体育)：涉及学校教学、考试招生、科技活动、文化场馆、文艺演出、体育赛事、健身设施等。
公卫医疗 (公共卫生/医疗卫生)：涉及医院就医、医保报销、药品价格、传染病防控、公共卫生事件、疫苗接种、医患纠纷等。
城乡建设 (城乡建设/规划/住房)：涉及市政道路建设、老旧小区改造、房屋质量、物业管理、违章建筑、拆迁安置、城市规划、供水供气等。
自然资源与环境保护 (自然资源/生态环境)：涉及环境污染、噪音扰民、大气水体土壤污染、垃圾处理、耕地林地保护、自然资源开发等。
交通运输 (交通运输)：涉及公共交通、地铁公交运行、出租车网约车、道路拥堵、交通违章、货车超载、桥梁隧道养护、停车管理等。
农林牧渔 (农业/林业/畜牧业/渔业)：涉及农业生产补贴、农村土地流转、农产品价格、畜禽养殖污染、渔业资源、森林防火等。
公共安全 (公共安全/应急管理)：涉及火灾隐患、危化品安全、烟花爆竹、社会治安、治安案件、消防安全、安全生产事故等。
人力资源社会保障 (人社/就业/社保)：涉及劳动纠纷、工资拖欠、养老保险、医疗保险、失业保险、就业创业扶持、职称评定等。
民政社区 (民政/社区服务)：涉及低保救助、残疾人福利、养老服务、婚姻登记、社区服务、优抚安置、社会组织登记等。
市场监管 (市场秩序/消费者权益)：涉及消费维权、食品安全、药品监督、价格欺诈、假冒伪劣、无证经营、虚假宣传等。
经济财贸 (经济/财政/商贸)：涉及招商引资、企业服务、税收政策、商业活动、展会经济、对外贸易、金融信贷等。
司法行政 (司法/法制/行政)：涉及法律咨询、司法援助、行政复议、信访投诉、仲裁调解、监狱矫正、公证服务等。
政风行风党风 (党风廉政/作风问题)：涉及干部作风、吃拿卡要、不作为慢作为、腐败举报、政务窗口服务态度等。
其他 (不属于以上类别的其他事项)：涉及以上13类都无法归入的其他咨询、建议、投诉等。

Few-shot Examples (真实工单参考，请模仿这些输入-输出的对应关系):
""" + fewshot_str + """

Your Response (只输出一个类别):"""

# ========== 3. 从测试集抽300条（每类约21条） ==========
df_test = pd.read_csv(config.test_path, sep='\t', names=['text', 'label'], engine='python')
df_test['text'] = df_test['text'].str.replace('\n', ' ').str.replace('\r', ' ')

# 分层抽样：每类抽约 ceil(300/14)=22 条
sample_per_class = 22
sampled = []
for cls in config.class2id.keys():
    cls_test = df_test[df_test['label'] == config.class2id[cls]]
    n = min(sample_per_class, len(cls_test))
    sampled.append(cls_test.sample(n, random_state=42))
df_sample = pd.concat(sampled).sample(frac=1, random_state=42)
y_test = df_sample['label'].values
print(f'评估集: {len(df_sample)} 条（分层抽样）')

# ========== 4. few-shot 预测 ==========
y_pred = []
from tqdm import tqdm
for text in tqdm(df_sample['text'].tolist()):
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": base_prompt},
                {"role": "user", "content": text},
            ],
            stream=False,
            temperature=0.1
        )
        pred = response.choices[0].message.content.strip()
        # 映射到类别编号（找不到映射就按"其他"）
        y_pred.append(config.class2id.get(pred, config.class2id['其他']))
    except Exception as e:
        print(f'错误: {e}')
        y_pred.append(config.class2id['其他'])

# ========== 5. 评估 ==========
acc = accuracy_score(y_test, y_pred)
pre = precision_score(y_test, y_pred, average='macro', zero_division=0)
rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
print('=' * 50)
print(f'【LLM few-shot】准确率:{acc:.4f} 精确率:{pre:.4f} 召回率:{rec:.4f} F1:{f1:.4f}')
print(f'评估条数: {len(df_sample)}')
print(f'few-shot示例: {len(fewshot_examples)} 条')
print('=' * 50)
