# -*- coding: utf-8 -*-
"""
12345工单打标项目 - 统一模型预测调度（供Gradio等界面调用）
支持: RF / FastText / BERT / LLM 四种模型
"""
import os
import sys

# 把项目根目录加入系统路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _01_data._01_config import Config

config = Config()


class ModelService:
    """模型服务：懒加载各模型，统一predict接口"""

    def __init__(self):
        self.rf_tfidf = None
        self.rf_model = None
        self.ft_model = None
        self.bert_model = None

    # ============ 随机森林 ============
    def _load_rf(self):
        import pickle
        with open(config.tfidf_save_path, 'rb') as f:
            self.rf_tfidf = pickle.load(f)
        with open(config.rf_save_model_path, 'rb') as f:
            self.rf_model = pickle.load(f)

    def predict_rf(self, text):
        import jieba
        if self.rf_model is None:
            self._load_rf()
        words = " ".join(jieba.lcut(text))
        number_words = self.rf_tfidf.transform([words])
        idx = int(self.rf_model.predict(number_words)[0])
        return config.id2class[idx], idx

    # ============ FastText ============
    def _load_ft(self):
        import fasttext
        self.ft_model = fasttext.load_model(config.ft_word_auto_model_path)

    def predict_ft(self, text):
        import jieba
        if self.ft_model is None:
            self._load_ft()
        words = " ".join(jieba.lcut(text))
        y_pred_tuple = self.ft_model.predict(words)
        label = y_pred_tuple[0][0][9:]  # 去掉__label__前缀
        idx = config.class2id.get(label, config.class2id['其他'])
        return label, idx

    # ============ BERT ============
    def _load_bert(self):
        import torch
        sys.path.insert(0, os.path.join(config.root_path, '_04_bert_base'))
        from _03_bert_classifier_model import MyBertClassifier
        from _01_config import Config as BertConfig
        self.bert_cfg = BertConfig()
        self.bert_model = MyBertClassifier()
        self.bert_model.load_state_dict(
            torch.load(config.bert_classifier_model_save_path, map_location=self.bert_cfg.device))
        self.bert_model.to(self.bert_cfg.device)
        self.bert_model.eval()

    def predict_bert(self, text):
        import torch
        if self.bert_model is None:
            self._load_bert()
        batch_texts_tensor = self.bert_cfg.bert_tokenizer(
            text, max_length=self.bert_cfg.max_len, padding='max_length',
            truncation=True, return_tensors='pt')
        batch_texts_tensor = {k: v.to(self.bert_cfg.device) for k, v in batch_texts_tensor.items()}
        logits = self.bert_model(batch_texts_tensor)
        idx = int(torch.argmax(logits, dim=-1).item())
        return config.id2class[idx], idx

    # ============ LLM(DeepSeek) ============
    def _get_deepseek_key(self):
        """优先环境变量，其次本地文件"""
        key = os.environ.get('DEEPSEEK_API_KEY')
        if key:
            return key
        key_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'deepseek_api_key.txt')
        if os.path.exists(key_file):
            key = open(key_file, 'r', encoding='utf8').read().strip()
            if key:
                return key
        return None

    def save_deepseek_key(self, key):
        """保存API Key到本地文件，供界面调用"""
        key = key.strip()
        if not key:
            return False
        key_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'deepseek_api_key.txt')
        with open(key_file, 'w', encoding='utf8') as f:
            f.write(key)
        return True

    def predict_llm(self, text):
        from openai import OpenAI
        api_key = self._get_deepseek_key()
        if not api_key:
            return '未设置DeepSeek API Key，请先在界面"LLM设置"里填写', -1
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com")
        system_prompt = open(config.root_path + '_05_llm_prompt/提示词.txt', 'r', encoding='utf8').read()
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=0.1)
        label = response.choices[0].message.content.strip()
        idx = config.class2id.get(label, config.class2id['其他'])
        return label, idx

    # ============ 统一预测接口 ============
    def predict(self, text, model_name='随机森林'):
        try:
            if model_name == '随机森林':
                label, idx = self.predict_rf(text)
            elif model_name == 'FastText':
                label, idx = self.predict_ft(text)
            elif model_name == 'BERT':
                label, idx = self.predict_bert(text)
            elif model_name == 'LLM(DeepSeek)':
                label, idx = self.predict_llm(text)
            else:
                return f'未知模型: {model_name}', -1
            return label, idx
        except Exception as e:
            return f'预测失败: {e}', -1


if __name__ == '__main__':
    svc = ModelService()
    test_text = '我家小区门口的路灯坏了三个月没人修，晚上出行非常危险'
    print('模型列表: 随机森林 / FastText / BERT / LLM(DeepSeek)')
    for name in ['随机森林', 'FastText', 'BERT', 'LLM(DeepSeek)']:
        label, idx = svc.predict(test_text, name)
        print(f'{name}: 类别={label} (id={idx})')
