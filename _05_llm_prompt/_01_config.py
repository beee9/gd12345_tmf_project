# -*- coding: utf-8 -*-
"""
12345工单打标项目 - LLM模块配置
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _01_data._01_config import Config as GlobalConfig


class Config(GlobalConfig):
    def __init__(self):
        super().__init__()


if __name__ == '__main__':
    c = Config()
    print('类别数:', c.class_num)
    print(c.id2class)
