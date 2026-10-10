# -*- coding: utf-8 -*-
"""
BERT 量化实验：动态量化(INT8) vs 原版(FP32)
对比：模型大小、推理速度、测试集F1
注意：必须先 import pandas 再 import torch/transformers！
      (torch 2.4 与 pandas 3.x DLL 加载顺序冲突, 反序会导致 0xC0000005 原生崩溃)
"""
import pandas as pd  # 必须放在最前
import sys
import os
import time
import copy
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _01_config import Config
from _02_dataloader_utils import build_all_dataloader
from _03_bert_classifier_model import MyBertClassifier
from _04_bert_model_eval_utils import model_eval

config = Config()

# ========== 1. 加载原版模型 ==========
print('加载原版模型...')
my_bert_model = MyBertClassifier()
my_bert_model.load_state_dict(torch.load(config.bert_classifier_model_save_path, map_location=config.device))
my_bert_model.to(config.device)
my_bert_model.eval()

# ========== 2. 模型大小对比 ==========
fp32_size = os.path.getsize(config.bert_classifier_model_save_path) / 1024 / 1024
print(f'原版FP32模型大小: {fp32_size:.1f} MB')

# ========== 3. 加载数据 + 评估原版(FP32, GPU) ==========
# 注意: 必须在量化之前完成 GPU 评估!
# 因为 config.bert_model 是全局共享实例, 量化前需要把它从 GPU 移到 CPU
train_dataloader, dev_dataloader, test_dataloader = build_all_dataloader()

print('评估原版(FP32, GPU)...')
acc_fp32, pre_fp32, rec_fp32, f1_fp32 = model_eval(test_dataloader, my_bert_model)
print(f'FP32: 准确率:{acc_fp32:.4f} 精确率:{pre_fp32:.4f} 召回率:{rec_fp32:.4f} F1:{f1_fp32:.4f}')

# ========== 4. 动态量化（INT8，必须在CPU上执行） ==========
print('动态量化(INT8)...')
try:
    # 先释放 GPU 模型, 把共享的 config.bert_model 移回 CPU
    del my_bert_model
    torch.cuda.empty_cache()
    config.bert_model.to('cpu')

    # 注意: config.bert_model 是全局共享实例, quantize_dynamic 是原地操作
    # 必须 deepcopy 隔离, 否则会污染 config.bert_model(其Linear被替换为量化版)
    model_cpu = MyBertClassifier()
    model_cpu.bert = copy.deepcopy(config.bert_model)  # 隔离共享BERT
    model_cpu.load_state_dict(torch.load(config.bert_classifier_model_save_path, map_location='cpu'))
    model_cpu.to('cpu')
    model_cpu.eval()

    # torch <=2.4 动态量化兼容 transformers BERT 完整结构
    # 直接量化全模型所有 Linear 层（BERT 主体 + 分类头），权重 INT8
    quantized_model = torch.quantization.quantize_dynamic(
        model_cpu, {torch.nn.Linear}, dtype=torch.qint8
    )
    print('量化成功(全量INT8, BERT主体+分类头)')
except Exception as e:
    print(f'量化失败: {e}')
    quantized_model = None

# ========== 5. 评估量化版（CPU） ==========
if quantized_model is not None:
    quantized_model = quantized_model.to('cpu')
    print('评估量化版(INT8, CPU)...')
    acc_int8, pre_int8, rec_int8, f1_int8 = model_eval(test_dataloader, quantized_model, device='cpu')
    print(f'INT8: 准确率:{acc_int8:.4f} 精确率:{pre_int8:.4f} 召回率:{rec_int8:.4f} F1:{f1_int8:.4f}')

# ========== 6. 推理速度对比（都在CPU上测，公平对比） ==========
print('推理速度对比（单条, CPU）...')
sample_text = "我家小区门口的路灯坏了三个月没人修，晚上出行非常危险"
tokens = config.bert_tokenizer([sample_text], max_length=config.max_len, padding='max_length',
                               truncation=True, return_tensors='pt')
tokens_cpu = {k: v.to('cpu') for k, v in tokens.items()}

# FP32速度（CPU）— 同样隔离共享BERT, 避免被量化污染
fp32_cpu = MyBertClassifier()
fp32_cpu.bert = copy.deepcopy(config.bert_model)  # 隔离: 此时的config.bert_model是干净FP32
fp32_cpu.load_state_dict(torch.load(config.bert_classifier_model_save_path, map_location='cpu'))
fp32_cpu.to('cpu')
fp32_cpu.eval()
with torch.no_grad():
    for _ in range(5):
        fp32_cpu(tokens_cpu)
    t0 = time.time()
    for _ in range(20):
        fp32_cpu(tokens_cpu)
    t_fp32 = (time.time() - t0) / 20 * 1000

# INT8速度（CPU）
if quantized_model is not None:
    with torch.no_grad():
        for _ in range(5):
            quantized_model(tokens_cpu)
        t0 = time.time()
        for _ in range(20):
            quantized_model(tokens_cpu)
        t_int8 = (time.time() - t0) / 20 * 1000
    print(f'FP32(CPU): {t_fp32:.1f} ms/条 | INT8(CPU): {t_int8:.1f} ms/条 | 加速 {t_fp32/t_int8:.1f}x')
else:
    print(f'FP32(CPU): {t_fp32:.1f} ms/条')

# ========== 7. 保存量化模型 ==========
if quantized_model is not None:
    save_dir = os.path.join(os.path.dirname(config.bert_classifier_model_save_path))
    save_path = os.path.join(save_dir, 'bert_classifier_int8.pth')
    torch.save(quantized_model.state_dict(), save_path)
    int8_size = os.path.getsize(save_path) / 1024 / 1024
    print(f'量化模型已保存: {save_path}')
    print(f'INT8模型大小: {int8_size:.1f} MB (压缩比 {fp32_size/int8_size:.2f}x)')
    print('=' * 55)
    print(f'F1对比: FP32={f1_fp32:.4f} vs INT8={f1_int8:.4f} (损失 {f1_fp32-f1_int8:.4f})')
