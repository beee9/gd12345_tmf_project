# -*- coding: utf-8 -*-
"""
极简GPU诊断：只检查设备状态，不跑训练
"""
import torch
print('torch版本:', torch.__version__)
print('CUDA可用:', torch.cuda.is_available())
print('GPU数量:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('GPU名称:', torch.cuda.get_device_name(0))
    # 创建一个小张量测试GPU计算
    x = torch.randn(1000, 1000).cuda()
    y = (x @ x).sum().item()
    print('GPU计算测试成功:', y)
