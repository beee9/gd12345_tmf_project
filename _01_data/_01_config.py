# -*- coding: utf-8 -*-
"""
12345工单打标项目 - 全局配置文件
参考 szai6_tmf_project 架构，适用于工单分类（14类）
"""
import os

# 项目根路径（绝对路径，避免相对路径出错）
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'


class Config():
    def __init__(self):
        print('正在初始化配置文件....')
        # ============ 1.原始数据路径 ============
        self.root_path = ROOT_PATH
        self.train_path = self.root_path + '_01_data/train.txt'
        self.test_path = self.root_path + '_01_data/test.txt'
        self.dev_path = self.root_path + '_01_data/dev.txt'
        # 停用词和类别路径
        self.stop_words_path = self.root_path + '_01_data/stopwords.txt'
        self.class_path = self.root_path + '_01_data/class.txt'
        # 分类词表: {0: '科教文体', 1: '公卫医疗', ...}
        self.id2class = {index: line.strip() for index, line in enumerate(open(self.class_path, 'r', encoding='utf8'))}
        self.class2id = {line.strip(): index for index, line in enumerate(open(self.class_path, 'r', encoding='utf8'))}

        # ============ 2.随机森林相关路径 ============
        # 随机森林处理后数据存放路径
        self.process_train_path = self.root_path + '_02_rf/processed_data/train_process.txt'
        self.process_test_path = self.root_path + '_02_rf/processed_data/test_process.txt'
        self.process_dev_path = self.root_path + '_02_rf/processed_data/dev_process.txt'
        # 随机森林模型保存路径
        self.rf_save_model_path = self.root_path + '_02_rf/model/rf.pkl'
        self.tfidf_save_path = self.root_path + '_02_rf/model/tfidf.pkl'

        # ============ 3.fasttext相关路径 ============
        # fasttext处理后数据存放路径（字符级别）
        self.process_train_path_chars = self.root_path + '_03_fasttext/processed_data/train_process_chars.txt'
        self.process_test_path_chars = self.root_path + '_03_fasttext/processed_data/test_process_chars.txt'
        self.process_dev_path_chars = self.root_path + '_03_fasttext/processed_data/dev_process_chars.txt'
        # fasttext处理后数据存放路径（词级别）
        self.process_train_path_words = self.root_path + '_03_fasttext/processed_data/train_process_words.txt'
        self.process_test_path_words = self.root_path + '_03_fasttext/processed_data/test_process_words.txt'
        self.process_dev_path_words = self.root_path + '_03_fasttext/processed_data/dev_process_words.txt'
        # fasttext模型保存路径
        self.ft_char_default_model_path = self.root_path + '_03_fasttext/model/ft_char_default.bin'
        self.ft_char_auto_model_path = self.root_path + '_03_fasttext/model/ft_char_auto.bin'
        self.ft_word_default_model_path = self.root_path + '_03_fasttext/model/ft_word_default.bin'
        self.ft_word_auto_model_path = self.root_path + '_03_fasttext/model/ft_word_auto.bin'

        # ============ 4.bert相关路径和参数 ============
        # bert预训练模型路径
        self.bert_base_chinese_path = self.root_path + '_04_bert_base/bert-base-chinese'
        # bert模型保存路径
        self.bert_classifier_model_save_path = self.root_path + '_04_bert_base/model/bert_classifier_model.pt'
        # todo 参数
        # 前面适合window   如果你是mac且支持mps,直接设置'mps'也行
        self.device = None  # 延迟到bert模块初始化
        self.epochs = 3
        self.batch_size = 64
        self.lr = 5e-5
        self.max_len = 128
        self.class_num = len(self.id2class)  # 14

        print('配置文件初始化动作完成!')


if __name__ == '__main__':
    c = Config()
    print('root_path:', c.root_path)
    print('类别数:', c.class_num)
    print('类别映射:', c.id2class)
