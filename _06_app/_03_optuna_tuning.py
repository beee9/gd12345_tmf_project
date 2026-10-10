# -*- coding: utf-8 -*-
"""
12345工单打标项目 - Optuna 自动调参
功能:
  1. 随机森林自动调参（搜索 n_estimators/max_depth/min_samples_split 等）
  2. FastText 自动调参（搜索 lr/dim/epoch/minCount 等）
  3. 调参后保存最优模型，并输出调参报告

运行: python _06_app/_03_optuna_tuning.py
"""
import os
import sys
import time
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _01_data._01_config import Config

config = Config()


# ==================== 随机森林调参 ====================
def tune_random_forest(n_trials=30):
    """使用Optuna对随机森林进行贝叶斯搜索调参"""
    import optuna
    import pandas as pd
    import sklearn
    from sklearn.metrics import f1_score
    from sklearn.ensemble import RandomForestClassifier

    print('=' * 60)
    print('开始随机森林自动调参...')

    # 1.读取分词后的训练数据
    df_data = pd.read_csv(config.process_train_path, sep='\t')
    x_words = df_data['words']
    y_labels = df_data['label']

    # 2.TFIDF转换（复用训练脚本的停用词逻辑）
    stop_words = [line.strip() for line in open(config.stop_words_path, 'r', encoding='utf8')]
    tfidf = sklearn.feature_extraction.text.TfidfVectorizer(stop_words=stop_words)
    words_feature = tfidf.fit_transform(x_words)
    print(f'Tfidf特征矩阵形状: {words_feature.shape}')

    # 3.切分数据（训练80% / 验证20%）
    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(
        words_feature, y_labels, test_size=0.2, random_state=666)

    # 4.定义目标函数（Optuna最大化验证集F1）
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=50),
            'max_depth': trial.suggest_int('max_depth', 10, 100, step=10),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'n_jobs': -1,
            'verbose': 0,
            'random_state': 666,
        }
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        return f1_score(y_val, y_pred, average='macro')

    # 5.贝叶斯搜索
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=666))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # 6.输出最优参数并用全量训练数据重训
    best_params = study.best_params
    print(f'最优F1: {study.best_value:.4f}')
    print(f'最优参数: {best_params}')
    best_params.update({'n_jobs': -1, 'verbose': 0, 'random_state': 666})
    best_model = RandomForestClassifier(**best_params)
    best_model.fit(words_feature, y_labels)

    # 7.保存最优模型和TFIDF
    os.makedirs(os.path.dirname(config.tfidf_save_path), exist_ok=True)
    with open(config.tfidf_save_path, 'wb') as f:
        pickle.dump(tfidf, f)
    with open(config.rf_save_model_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f'最优随机森林模型已保存: {config.rf_save_model_path}')

    return study.best_params, study.best_value


# ==================== FastText 调参 ====================
def tune_fasttext(n_trials=20):
    """使用Optuna对FastText进行自动调参"""
    import optuna
    import fasttext

    print('=' * 60)
    print('开始FastText自动调参...')

    # 1.定义目标函数
    def objective(trial):
        params = {
            'lr': trial.suggest_float('lr', 0.05, 1.0, log=True),
            'dim': trial.suggest_int('dim', 50, 300, step=50),
            'epoch': trial.suggest_int('epoch', 5, 50, step=5),
            'minCount': trial.suggest_int('minCount', 1, 5),
            'wordNgrams': trial.suggest_int('wordNgrams', 1, 4),
            'verbose': 0,
        }
        model = fasttext.train_supervised(
            input=config.process_train_path_words,
            **params)
        result = model.test(config.process_dev_path_words)
        return result[1]  # 精确率

    # 2.贝叶斯搜索
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=666))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # 3.输出最优参数并重训
    best_params = study.best_params
    print(f'最优精确率: {study.best_value:.4f}')
    print(f'最优参数: {best_params}')
    best_params.update({'verbose': 2})
    best_model = fasttext.train_supervised(input=config.process_train_path_words, **best_params)

    # 4.保存模型
    os.makedirs(os.path.dirname(config.ft_word_auto_model_path), exist_ok=True)
    best_model.save_model(config.ft_word_auto_model_path)
    print(f'最优FastText模型已保存: {config.ft_word_auto_model_path}')

    return study.best_params, study.best_value


if __name__ == '__main__':
    start = time.time()
    try:
        import optuna
    except ImportError:
        print('❌ 未安装optuna，请先安装: pip install optuna')
        sys.exit(1)

    # 随机森林调参
    try:
        rf_params, rf_f1 = tune_random_forest(n_trials=30)
    except Exception as e:
        print(f'随机森林调参失败: {e}')
        rf_params, rf_f1 = None, None

    # FastText调参
    try:
        ft_params, ft_pre = tune_fasttext(n_trials=20)
    except Exception as e:
        print(f'FastText调参失败: {e}')
        ft_params, ft_pre = None, None

    print('=' * 60)
    print(f'调参完成，总耗时: {time.time()-start:.1f}秒')
    if rf_params:
        print(f'随机森林最优F1: {rf_f1:.4f}, 参数: {rf_params}')
    if ft_params:
        print(f'FastText最优精确率: {ft_pre:.4f}, 参数: {ft_params}')
