# -*- coding: utf-8 -*-
"""
剪枝 (Pruning) 实验
方法: 幅度剪枝 (Magnitude Pruning) - 对BERT分类头的linear层和注意力层剪掉低幅值权重
对比: 剪枝前后 模型大小 / 测试集F1 / 推理速度
"""
import sys
import os
import time
import torch
import torch.nn.utils.prune as prune

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _01_config import Config
from _02_dataloader_utils import build_all_dataloader
from _03_bert_classifier_model import MyBertClassifier
from _04_bert_model_eval_utils import model_eval

config = Config()

# ========== 1. 加载原版模型 ==========
print('加载原版模型...')
model = MyBertClassifier()
model.load_state_dict(torch.load(config.bert_classifier_model_save_path, map_location=config.device))
model.to(config.device)
model.eval()

# ========== 2. 评估原版 ==========
train_dataloader, dev_dataloader, test_dataloader = build_all_dataloader()
print('评估原版模型(测试集)...')
acc0, pre0, rec0, f10 = model_eval(test_dataloader, model)

# 统计原版参数量
total_params0 = sum(p.numel() for p in model.parameters())
print(f'原版参数量: {total_params0/1e6:.1f}M')

# ========== 3. 幅度剪枝 ==========
prune_amount = 0.3  # 剪掉30%低幅值权重
print(f'幅度剪枝: 剪掉 {prune_amount*100:.0f}% 低幅值权重...')

pruned_modules = []
for name, module in model.named_modules():
    # 对Linear层剪枝（BERT的attention和FFN全是Linear）
    if isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=prune_amount)
        pruned_modules.append((name, module))

print(f'已对 {len(pruned_modules)} 个Linear层剪枝')

# 计算稀疏度（被剪掉的比例）
total_zeros = 0
total_params = 0
for name, module in pruned_modules:
    weight = module.weight.detach()
    total_zeros += (weight == 0).sum().item()
    total_params += weight.numel()
sparsity = total_zeros / total_params
print(f'整体稀疏度: {sparsity*100:.1f}%')

# ========== 4. 评估剪枝后 ==========
print('评估剪枝后模型(测试集)...')
acc1, pre1, rec1, f11 = model_eval(test_dataloader, model)

# ========== 5. 推理速度对比 ==========
sample_text = "我家小区门口的路灯坏了三个月没人修，晚上出行非常危险"
tokens = config.bert_tokenizer([sample_text], max_length=config.max_len, padding='max_length',
                               truncation=True, return_tensors='pt')
tokens = {k: v.to(config.device) for k, v in tokens.items()}

with torch.no_grad():
    for _ in range(5):
        model(tokens)
    t0 = time.time()
    for _ in range(20):
        model(tokens)
    t1 = (time.time() - t0) / 20 * 1000

# ========== 6. 输出对比 ==========
print('=' * 55)
print('【剪枝实验结果】')
print(f'F1对比: 原版={f10:.4f} → 剪枝后={f11:.4f} (损失 {f10-f11:.4f})')
print(f'稀疏度: {sparsity*100:.1f}% (剪掉了{total_zeros/1e6:.1f}M个参数)')
print(f'推理速度: {t1:.1f} ms/条')
print('=' * 55)

# ========== 7. 保存剪枝模型（移除剪枝掩码，固化稀疏权重） ==========
os.makedirs(os.path.join(os.path.dirname(config.bert_classifier_model_save_path), 'prune'), exist_ok=True)
save_path = os.path.join(os.path.dirname(config.bert_classifier_model_save_path), 'prune', 'bert_pruned.pth')
for name, module in pruned_modules:
    prune.remove(module, 'weight')  # 固化掩码
torch.save(model.state_dict(), save_path)
size_mb = os.path.getsize(save_path) / 1024 / 1024
print(f'剪枝模型已保存: {save_path} ({size_mb:.1f}MB)')
