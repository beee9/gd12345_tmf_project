# -*- coding: utf-8 -*-
"""
12345工单打标项目 - BERT模型边训练边评估
"""
# 导包
import os
from _01_config import Config
from _02_dataloader_utils import build_all_dataloader
from _03_bert_classifier_model import MyBertClassifier
from _04_bert_model_eval_utils import model_eval
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1个配置文件
config = Config()


# 模型训练API
def model_train():
    # 4个准备
    # 准备数据
    train_dataloader, dev_dataloader, test_dataloader = build_all_dataloader()
    # 准备模型
    my_bert_model = MyBertClassifier()
    my_bert_model.to(config.device)
    my_bert_model.train()
    # 准备损失函数
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    # 准备优化器
    optimizer = torch.optim.AdamW(my_bert_model.parameters(), lr=config.lr, betas=(0.9, 0.999))
    # 提前定义best_score,初始0
    best_f1score = 0
    # 2个遍历
    # 外层循环控制轮次
    for epoch in range(1, config.epochs + 1):
        # 额外添加日志变量  每10批打印损失,准确率,精确率,召回率,f1分数
        total_loss, batch_cnt = 0, 0
        all_pred_labels, all_true_labels = [], []
        # 内层循环控制批次
        for index, (batch_texts_tensor, batch_labels_tensor) in enumerate(tqdm(train_dataloader), start=1):
            # 把数据放到指定设备上
            batch_labels_tensor = batch_labels_tensor.to(config.device)
            batch_texts_tensor = {k: v.to(config.device) for k, v in batch_texts_tensor.items()}
            # 5个核心
            # 前向传播
            logits = my_bert_model(batch_texts_tensor)
            # 计算损失
            loss = loss_fn(logits, batch_labels_tensor)
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 参数更新
            optimizer.step()
            # 额外计算日志变量
            total_loss += loss.item()
            batch_cnt += 1
            all_true_labels.extend(batch_labels_tensor.tolist())
            batch_pred_labels = torch.argmax(logits, dim=-1)
            all_pred_labels.extend(batch_pred_labels.tolist())
            # 额外打印日志  每10批打印损失,准确率,精确率,召回率,f1分数
            if index % 10 == 0 or index == len(train_dataloader):
                # 计算准确率,精确率,召回率,f1分数
                acc = accuracy_score(all_true_labels, all_pred_labels)
                pre = precision_score(all_true_labels, all_pred_labels, average='macro')
                rec = recall_score(all_true_labels, all_pred_labels, average='macro')
                f1 = f1_score(all_true_labels, all_pred_labels, average='macro')
                # 计算损失
                avg_loss = total_loss / batch_cnt
                # 打印日志
                print(f"训练日志: 轮次:{epoch},当前批次:{index},损失:{avg_loss},准确率:{acc},精确率:{pre},召回率:{rec},f1:{f1}")
                # 清空日志变量
                total_loss, batch_cnt = 0, 0
                all_pred_labels, all_true_labels = [], []
            # 边训练边评估  每100批,评估一次,如果当前分数高于历史最高分,记录并保存
            if index % 100 == 0 or index == len(train_dataloader):
                # 调用model_eval()进行评估
                acc, pre, rec, f1 = model_eval(dev_dataloader, my_bert_model)
                print(f"评估日志: 轮次:{epoch},当前批次:{index},准确率:{acc},精确率:{pre},召回率:{rec},f1:{f1}")
                # 一定记得把模型再切换到训练模式
                my_bert_model.train()
                # 判断当前评估分数是否是最优分数,如果是就记录并保存
                if f1 > best_f1score:
                    # 记录当前最优分数
                    best_f1score = f1
                    # 1个保存（先确保目录存在）
                    os.makedirs(os.path.dirname(config.bert_classifier_model_save_path), exist_ok=True)
                    torch.save(my_bert_model.state_dict(), config.bert_classifier_model_save_path)
                    print(f"保存当前最优分数:{f1}的模型")


if __name__ == '__main__':
    model_train()
