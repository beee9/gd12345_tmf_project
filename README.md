# 12345工单打标项目（gd12345_tmf_project）

基于 **12345政务服务便民热线工单数据** 的文本分类项目，参考 `szai6_tmf_project`（AI大模型）架构搭建。

## 一、项目结构

```
gd12345_tmf_project/
├── index.html                         # Voxel Hero 风格前端落地页（浏览器访问入口）
├── .streamlit/config.toml             # Streamlit 大屏 Voxel Hero 主题配置
├── _01_data/                          # 数据准备模块
│   ├── train.txt                      # 训练集（工单文本\t类别id）
│   ├── dev.txt                        # 验证集
│   ├── test.txt                       # 测试集
│   ├── class.txt                      # 14个类别名称（每行一个）
│   ├── stopwords.txt                  # 停用词表
│   ├── _01_config.py                  # 全局配置文件
│   └── _03_数据分析.py                 # 数据分析脚本
├── _02_rf/                            # 随机森林基线模型
│   ├── _01_config.py                  # 配置（复用全局配置）
│   ├── _02_数据分词并存储.py           # jieba分词
│   ├── _03_随机森林模型训练.py         # TFIDF + 随机森林训练
│   ├── _04_随机森林模型评估.py         # 测试集评估
│   └── _05_rf_predict_fun.py          # 预测函数（供Web调用）
├── _03_fasttext/                      # FastText 模型
│   ├── _02_数据处理_字符和词两种方式.py
│   ├── _03_字符级别数据和默认参数方式训练模型.py
│   ├── _03_词级别数据和默认参数方式训练模型.py
│   ├── _04_词级别的数据和自动调参数方式训练模型.py
│   └── _05_ft_predict_fun.py          # 预测函数
├── _04_bert_base/                     # BERT 微调模型
│   ├── bert-base-chinese/             # 预训练模型（已拷贝）
│   ├── _01_config.py
│   ├── _02_dataloader_utils.py        # 数据加载器
│   ├── _03_bert_classifier_model.py   # BERT分类模型
│   ├── _04_bert_model_eval_utils.py   # 评估工具
│   ├── _04_bert模型边训练边评估.py     # 训练脚本
│   └── _05_bert_predict_fun.py        # 预测函数
├── _05_llm_prompt/                    # LLM 大模型分类
│   ├── _01_config.py
│   ├── 提示词.txt                      # 14类工单分类提示词
│   ├── _02_llm_predict_fun.py         # DeepSeek 预测函数
│   └── _03_llm配合提示词_模型评估.py   # LLM 评估脚本
├── _06_app/                           # 可视化应用层
│   ├── _01_model_service.py           # 统一模型调度（RF/FastText/BERT/LLM）
│   ├── _02_gradio_app.py              # Gradio 网页交互界面
│   ├── _03_optuna_tuning.py           # Optuna 自动调参（RF + FastText）
│   ├── _04_streamlit_dashboard.py     # Streamlit 数据大屏
│   └── _05_web_app.py                 # Voxel Hero 风格网页服务器（纯标准库）
└── 文档/                              # 分析报告等
```

## 二、数据说明

| 数据集 | 文件 | 样本数 |
|--------|------|--------|
| 训练集 | train.txt | 25323 |
| 验证集 | dev.txt   | 3163  |
| 测试集 | test.txt  | 3164  |

**14个类别**（class.txt）：
科教文体、公卫医疗、城乡建设、自然资源与环境保护、交通运输、农林牧渔、公共安全、人力资源社会保障、民政社区、市场监管、经济财贸、司法行政、政风行风党风、其他

## 三、运行流程

### 1. 安装依赖
```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple jieba scikit-learn pandas numpy fasttext tqdm
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch transformers matplotlib openai
```

### 2. 随机森林（最快，无需GPU）
```bash
cd _02_rf
python _02_数据分词并存储.py
python _03_随机森林模型训练.py
python _04_随机森林模型评估.py
python _05_rf_predict_fun.py
```

### 3. FastText（速度快）
```bash
cd _03_fasttext
python _02_数据处理_字符和词两种方式.py
python _03_字符级别数据和默认参数方式训练模型.py
python _03_词级别数据和默认参数方式训练模型.py
python _04_词级别的数据和自动调参数方式训练模型.py
```

### 4. BERT（效果最好，需要transformers）
```bash
cd _04_bert_base
python _04_bert模型边训练边评估.py
python _05_bert_predict_fun.py
```

### 5. LLM（需要设置 DeepSeek API Key）
```bash
set DEEPSEEK_API_KEY=你的key
cd _05_llm_prompt
python _03_llm配合提示词_模型评估.py
```

### 6. Gradio 可视化界面（推荐）
训练好任意模型后，启动网页界面，浏览器打开 http://127.0.0.1:7860
```bash
# 需要安装 gradio
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple gradio

# 启动界面（在项目根目录下运行）
cd E:\Binge project\gd12345_tmf_project
python _06_app\_02_gradio_app.py
```
界面上支持：输入工单内容 → 选择模型（随机森林/FastText/BERT/LLM）→ 一键打标。

### 7. Optuna 自动调参（提升模型效果）
```bash
# 需要安装 optuna
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple optuna

cd E:\Binge project\gd12345_tmf_project
python _06_app\_03_optuna_tuning.py
```
自动搜索随机森林和 FastText 的最优超参数（30次搜索），调参结果自动保存为最优模型。

### 8. Streamlit 数据大屏
```bash
# 需要安装 streamlit
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple streamlit

cd E:\Binge project\gd12345_tmf_project
streamlit run _06_app\_04_streamlit_dashboard.py
```
功能：数据集总览（类别分布/文本长度）、在线打标、多模型效果对比表。
已适配 Voxel Hero 主题（深黑底 + 品牌橙 + Bebas Neue），主题配置见 `.streamlit/config.toml`。

### 9. 浏览器前端（政务风格，推荐）
落地页为政务风格设计（深黑 + 政务蓝 #2563EB、Bebas Neue 标题、视频背景、玻璃拟态），
将 `index.html` 落地页与后端模型打通，浏览器打开 http://127.0.0.1:8000 即可使用。
页面内置：在线打标（单模型 / 四模型投票）、模型支撑、14 类工单类别体系。
移动端自动切换为汉堡菜单导航。
注：如需换回橙色品牌风，将 index.html 与 _04_streamlit_dashboard.py 中的
`#2563EB` / `#1D4ED8` 全局替换为 `#FF5733` / `#E04C2C` 即可（其余无需改动）。

**一键启动（推荐，双击即可）**：
```text
启动-前端落地页.bat     → 前端 http://127.0.0.1:8000
启动-数据大屏.bat       → 大屏 http://127.0.0.1:8501
启动-全部服务.bat       → 两个一起启动并打开页面
```

手动启动：
```bash
cd E:\Binge project\gd12345_tmf_project
E:\anaconda3\envs\pytorch-gpu\python.exe _06_app\_05_web_app.py
```
- 服务器本体纯 Python 标准库实现，零额外依赖；模型依赖使用 conda 环境
  `pytorch-gpu`（已装 sklearn/fasttext/torch/transformers/openai）。
- 模型懒加载，首次调用 BERT 需等待权重加载（约 10 秒），属正常现象。
- LLM(DeepSeek) 需先设置环境变量 `DEEPSEEK_API_KEY`。

**常见问题**：
- 页面能打开但"开始打标"报 `Failed to fetch`：说明后端服务没在运行，双击启动脚本即可；
  注意要访问 http://127.0.0.1:8000，不要直接双击 index.html 打开。
- 报 `No module named 'xxx'`：启动时没有用对 Python 环境，请改用
  `E:\anaconda3\envs\pytorch-gpu\python.exe`（三个启动脚本已默认使用该环境）。
- 开了代理/VPN（梯子）时访问 127.0.0.1 失败：请把 `127.0.0.1`、`localhost` 加入
  代理软件（Clash/v2ray 等）的直连/绕过列表，或临时关闭系统代理。

## 10. 分享给他人使用（局域网 / 外网）

**场景一：同一网络（连同一个 WiFi / 同一个路由器）**
1. 双击 `放行防火墙-以管理员运行.bat`（需点一次"是"授权）放行 8000/8501 端口
2. 查看本机局域网 IP：`ipconfig` 里"WLAN"一栏的 IPv4 地址（当前是 `192.168.1.6`）
3. 其他电脑/手机浏览器访问：`http://192.168.1.6:8000`（前端）、`http://192.168.1.6:8501`（大屏）
4. 注意：路由器若开启"AP 隔离/客户端隔离"，需关闭，否则设备间无法互访

**场景二：不同网络 / 外网（内网穿透，最简单免费）**
以 cpolar 为例：
```bash
# 1. 到 https://www.cpolar.com 注册并下载安装
cpolar authtoken <你的认证token>
# 2. 为前端开隧道（8000 端口）
cpolar http 8000
# 3. 得到 https://xxxx.cpolar.top 公网地址，发给任何地方的人即可访问
```
- 同类工具：natapp（国内）、ngrok（海外）、frp（自建，需有公网服务器）
- 免费版隧道地址每次启动会变，正式使用建议升级固定域名
- 大屏同理：另开一条 `cpolar http 8501`

**场景三：稳定正式（云服务器）**
阿里云/腾讯云轻量服务器（装 Ubuntu）→ 安装 Python 3.11 + 项目依赖 → 上传项目 → 运行两个服务 → 在云控制台"安全组"放行 8000/8501 → 用公网 IP 访问。

**安全提醒**：暴露到公网后任何人都能调用模型接口（预测消耗 CPU/GPU，LLM 消耗 API 额度）。
建议只做演示用；正式对外请加访问口令/鉴权，或部署在内网仅对单位开放。

## 四、说明
- 所有模块的 `_01_config.py` 均复用 `_01_data/_01_config.py` 全局配置，修改数据路径只需改一处。
- 随机森林与 FastText 为 CPU 基线模型，BERT 需要较长时间训练（有GPU更快）。
- LLM 模块使用 DeepSeek API，需自行申请 Key 并设置环境变量。
- `_06_app/_01_model_service.py` 为统一模型调度，新模型接入只需实现 `predict_xxx(text)` 并注册到 `predict()`。
