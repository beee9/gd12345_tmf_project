# -*- coding: utf-8 -*-
"""
12345工单打标项目 - 随机森林模块配置
复用全局配置 _01_data/_01_config.py
"""
import sys
import os

# 把项目根目录加入系统路径，保证任意位置运行都能导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _01_data._01_config import Config

if __name__ == '__main__':
    c = Config()
    print(c.train_path)
    print(c.process_train_path)
    print(c.rf_save_model_path)
