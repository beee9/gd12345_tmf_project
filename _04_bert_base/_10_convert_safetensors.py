# -*- coding: utf-8 -*-
"""
把 BERT 预训练权重 pytorch_model.bin 转为 model.safetensors
原因：transformers 4.57+ 因 CVE-2025-32434 禁止 torch<2.6 用 torch.load 加载 .bin
     但 safetensors 格式不受此限制，转换后可配合 torch 2.4.1 正常加载
"""
import os
import torch
from safetensors.torch import save_file

base_dir = r'E:\Binge project\gd12345_tmf_project\_04_bert_base\bert-base-chinese'
bin_path = os.path.join(base_dir, 'pytorch_model.bin')
safetensors_path = os.path.join(base_dir, 'model.safetensors')

print('加载 pytorch_model.bin ...')
sd = torch.load(bin_path, map_location='cpu', weights_only=False)
print(f'state_dict keys: {len(sd)}')

print('保存 model.safetensors ...')
# clone + contiguous 解除共享内存与非连续存储，否则 safetensors 拒绝保存
sd = {k: v.clone().contiguous() for k, v in sd.items()}
save_file(sd, safetensors_path)
print(f'完成: {os.path.getsize(safetensors_path)/1024/1024:.1f} MB')

# 备份原文件，避免 transformers 优先选择 .bin 触发 CVE 检查
backup = bin_path.replace('pytorch_model.bin', 'pytorch_model.bin.bak')
os.rename(bin_path, backup)
print(f'原 bin 已备份为: {backup}')
print('全部完成!')
