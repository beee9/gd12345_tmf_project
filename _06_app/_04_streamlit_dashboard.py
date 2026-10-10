# -*- coding: utf-8 -*-
"""
12345工单打标项目 - Streamlit 数据大屏（政务风格）
功能:
  1. 数据集总览（样本数/类别分布/文本长度）
  2. 模型在线打标（RF/FastText/BERT/LLM）
  3. 多模型效果对比（需先运行各模型评估脚本）

运行: streamlit run _06_app/_04_streamlit_dashboard.py
设计: 深黑 #050505 + 政务蓝 #2563EB + 玻璃拟态
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from _01_data._01_config import Config
from _06_app._01_model_service import ModelService

# ==================== 暗色图表风格 ====================
plt.rcParams.update({
    'font.sans-serif': ['Microsoft YaHei', 'SimHei'],
    'axes.unicode_minus': False,
    'figure.facecolor': '#050505',
    'axes.facecolor': '#050505',
    'axes.edgecolor': '#333333',
    'axes.labelcolor': '#D1D5DB',
    'xtick.color': '#D1D5DB',
    'ytick.color': '#D1D5DB',
    'text.color': '#FFFFFF',
    'grid.color': '#1F2937',
    'grid.alpha': 0.5,
})

config = Config()
service = ModelService()

# ==================== Voxel Hero 主题 CSS ====================
VX_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600;700&display=swap');

/* 背景 */
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main {
    background: #050505;
}
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
[data-testid="stToolbar"] { visibility: hidden; }

/* 字体 */
html, body, .stApp {
    font-family: 'Inter', 'PingFang SC', 'Microsoft YaHei', sans-serif;
    -webkit-font-smoothing: antialiased;
}
h1, h2, h3 {
    font-family: 'Bebas Neue', 'PingFang SC', 'Microsoft YaHei', sans-serif;
    letter-spacing: 0.03em;
}
::selection { background: #2563EB; color: #FFFFFF; }

/* 指标卡片（玻璃拟态） */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 1rem;
    padding: 1.1rem 1.25rem;
    transition: background 0.3s, border-color 0.3s;
}
[data-testid="stMetric"]:hover { background: rgba(255,255,255,0.1); border-color: rgba(37,99,235,0.5); }
[data-testid="stMetricLabel"] { color: #9CA3AF; letter-spacing: 0.05em; }
[data-testid="stMetricValue"] { color: #2563EB; font-family: 'Bebas Neue', 'Microsoft YaHei', sans-serif; font-size: 2.2rem; }

/* 按钮 */
.stButton > button, .stDownloadButton > button {
    background: #2563EB;
    color: #FFFFFF;
    border: none;
    border-radius: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    font-size: 0.8rem;
    padding: 0.55rem 1.6rem;
    transition: all 0.3s;
}
.stButton > button:hover { background: #1D4ED8; color: #FFFFFF; border: none; }
.stButton > button:focus, .stButton > button:active { color: #FFFFFF; border: none; }

/* 文本区 */
[data-testid="stTextArea"] textarea {
    background: rgba(10,10,10,0.8);
    color: #E5E7EB;
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 0.6rem;
    transition: border-color 0.3s, box-shadow 0.3s;
}
[data-testid="stTextArea"] textarea:focus {
    border-color: #2563EB;
    box-shadow: 0 0 0 1px rgba(37,99,235,0.4);
}

/* 下拉选择框 */
[data-baseweb="select"] > div {
    background: rgba(10,10,10,0.8);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 0.6rem;
    color: #E5E7EB;
}
[data-baseweb="popover"] [data-baseweb="menu"],
[data-baseweb="popover"] [data-baseweb="listbox"] { background: #0F0F0F; }
[data-baseweb="popover"] li { color: #E5E7EB; }
[data-baseweb="popover"] li:hover { background: rgba(37,99,235,0.15); }

/* 侧边栏 */
[data-testid="stSidebar"] {
    background: rgba(10,10,10,0.92);
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #D1D5DB; }
[data-testid="stSidebar"] [data-testid="stRadio"] label p { color: #D1D5DB; }
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover p { color: #2563EB; }

/* Tabs */
[data-baseweb="tab-list"] { gap: 0.75rem; border-bottom: 1px solid rgba(255,255,255,0.08); }
[data-baseweb="tab"] { color: #6B7280; font-weight: 600; letter-spacing: 0.05em; font-size: 0.85rem; }
[data-baseweb="tab-highlight"] { background-color: #2563EB; }

/* 表格圆角 */
[data-testid="stDataFrame"] { border-radius: 0.75rem; overflow: hidden; }

/* 提示框圆角 */
[data-testid="stSuccess"], [data-testid="stError"],
[data-testid="stInfo"], [data-testid="stWarning"] { border-radius: 0.75rem; }

/* 分割线 */
hr { border-color: rgba(255,255,255,0.08) !important; }

/* 自定义组件 */
.vx-section-label {
    color: #2563EB;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.35em;
    margin: 0;
}
.vx-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 1rem;
    padding: 1.5rem 1.75rem;
    transition: background 0.3s, border-color 0.3s;
}
.vx-label {
    font-family: 'Bebas Neue', 'Microsoft YaHei', sans-serif;
    font-size: 3rem;
    color: #2563EB;
    line-height: 1;
    letter-spacing: 0.03em;
}
.vx-chip {
    display: inline-block;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 9999px;
    padding: 0.35rem 0.9rem;
    margin: 0.25rem 0.25rem 0 0;
    font-size: 0.8rem;
    color: #D1D5DB;
    transition: all 0.3s;
}
.vx-chip:hover {
    border-color: #2563EB;
    color: #FFFFFF;
    background: rgba(37,99,235,0.1);
}
"""

# 页面配置
st.set_page_config(page_title='12345政务服务便民热线 · 工单数据大屏', layout='wide')
st.markdown(f'<style>{VX_CSS}</style>', unsafe_allow_html=True)


# ==================== 顶部品牌栏 ====================
def render_header():
    st.markdown("""
    <div style="display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:0.5rem;padding:1.2rem 0 0.8rem;">
      <div style="font-family:'Bebas Neue','PingFang SC','Microsoft YaHei',sans-serif;font-size:1.9rem;letter-spacing:0.3em;line-height:1;">
        <span style="color:#FFFFFF;">12</span><span style="color:#2563EB;">3</span><span style="color:#FFFFFF;">45</span>
        <span style="color:#9CA3AF;font-size:0.95rem;letter-spacing:0.15em;margin-left:0.6rem;">工单数据大屏</span>
      </div>
      <div style="color:#6B7280;font-size:10px;font-weight:600;letter-spacing:0.25em;">政务服务 · RF / FASTTEXT / BERT / DEEPSEEK</div>
    </div>
    <div style="border-top:1px solid rgba(255,255,255,0.08);margin-bottom:1rem;"></div>
    """, unsafe_allow_html=True)


# ==================== 加载数据 ====================
@st.cache_data
def load_all_data():
    train = pd.read_csv(config.train_path, sep='\t', names=['text', 'label'])
    dev = pd.read_csv(config.dev_path, sep='\t', names=['text', 'label'])
    test = pd.read_csv(config.test_path, sep='\t', names=['text', 'label'])
    return train, dev, test


@st.cache_data
def load_class_map():
    return {i: name for i, name in enumerate(
        open(config.class_path, 'r', encoding='utf8').read().strip().split('\n'))}


# ==================== 侧边栏 ====================
st.sidebar.markdown(
    '<div style="font-family:\'Bebas Neue\',\'PingFang SC\',\'Microsoft YaHei\',sans-serif;'
    'font-size:1.7rem;letter-spacing:0.3em;padding:0.5rem 0 0.2rem;">'
    '<span style="color:#FFFFFF;">12</span><span style="color:#2563EB;">3</span>'
    '<span style="color:#FFFFFF;">45</span></div>',
    unsafe_allow_html=True)
st.sidebar.markdown('<p class="vx-section-label">// CONTROL</p>', unsafe_allow_html=True)
page = st.sidebar.radio('选择功能', ['📊 数据总览', '🤖 在线打标', '📈 模型对比'])
st.sidebar.markdown(
    '<p style="color:#6B7280;font-size:10px;letter-spacing:0.2em;margin-top:1.2rem;">'
    '工单智能分类系统<br>&copy; 2026 学习演示项目</p>',
    unsafe_allow_html=True)

train, dev, test = load_all_data()
class_map = load_class_map()

render_header()


# ==================== 页面1: 数据总览 ====================
if page == '📊 数据总览':
    st.markdown('<p class="vx-section-label">// DATA CENTER</p>', unsafe_allow_html=True)
    st.title('📊 工单数据集总览')

    # 指标卡片
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('训练集样本', f'{len(train):,}')
    col2.metric('验证集样本', f'{len(dev):,}')
    col3.metric('测试集样本', f'{len(test):,}')
    col4.metric('类别数量', f'{len(class_map)}')

    st.markdown('---')

    # 类别分布
    st.subheader('📌 类别分布对比')
    tab1, tab2, tab3 = st.tabs(['训练集', '验证集', '测试集'])
    for tab, df in [(tab1, train), (tab2, dev), (tab3, test)]:
        with tab:
            counts = df['label'].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(12, 5))
            bars = ax.bar([class_map[i] for i in counts.index], counts.values,
                          color='#2563EB', alpha=0.9, edgecolor='#050505')
            ax.set_ylabel('样本数', fontsize=11)
            ax.set_title('各类别样本数', fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            for bar, v in zip(bars, counts.values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts.values) * 0.008,
                        f'{v:,}', ha='center', fontsize=8, color='#D1D5DB')
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
            st.dataframe(pd.DataFrame({'类别': [class_map[i] for i in counts.index],
                                       '样本数': counts.values,
                                       '占比%': (counts.values / len(df) * 100).round(2)}))

    # 文本长度分布
    st.subheader('📏 文本长度分布')
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.hist(train['text'].str.len(), bins=50, color='#2563EB', alpha=0.85, edgecolor='#050505')
    ax.set_xlabel('文本长度（字符）', fontsize=11)
    ax.set_ylabel('频数', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    st.pyplot(fig)

    # 数据预览
    st.subheader('👀 数据预览')
    preview = train.head(10).copy()
    preview['类别'] = preview['label'].map(class_map)
    st.dataframe(preview[['text', '类别']])


# ==================== 页面2: 在线打标 ====================
elif page == '🤖 在线打标':
    st.markdown('<p class="vx-section-label">// ONLINE TAGGING</p>', unsafe_allow_html=True)
    st.title('🤖 工单在线打标')

    col1, col2 = st.columns([3, 1])
    with col2:
        model_name = st.selectbox(
            '选择模型',
            ['随机森林', 'FastText', 'BERT', 'LLM(DeepSeek)'],
            help='随机森林最快(CPU)；BERT效果最好；LLM需配置DeepSeek API Key')
    with col1:
        text_input = st.text_area('工单内容', height=120,
                                  placeholder='请输入工单内容，例如：我家小区门口的路灯坏了三个月没人修，晚上出行非常危险')

    if st.button('🚀 开始打标', type='primary', use_container_width=True):
        if not text_input.strip():
            st.warning('请输入工单内容')
        else:
            with st.spinner('正在预测...'):
                label, idx = service.predict(text_input.strip(), model_name)
            if idx == -1:
                st.error(label)
            else:
                st.markdown(f"""
                <div class="vx-card" style="text-align:center;">
                  <p class="vx-section-label" style="color:#6B7280;">预测类别</p>
                  <p class="vx-label">{label}</p>
                  <p style="color:#9CA3AF;font-size:0.85rem;margin:0.4rem 0 0;">id = {idx}</p>
                </div>""", unsafe_allow_html=True)
                st.markdown(
                    f'<p style="color:#D1D5DB;font-size:0.85rem;margin-top:0.6rem;">'
                    f'工单内容：{text_input[:200]}</p>', unsafe_allow_html=True)

    st.markdown('---')
    st.markdown('<p class="vx-section-label">// 14 CLASSES</p>', unsafe_allow_html=True)
    chips = ' '.join(f'<span class="vx-chip">{i}. {class_map[i]}</span>' for i in range(len(class_map)))
    st.markdown(chips, unsafe_allow_html=True)


# ==================== 页面3: 模型对比 ====================
elif page == '📈 模型对比':
    st.markdown('<p class="vx-section-label">// BENCHMARK</p>', unsafe_allow_html=True)
    st.title('📈 多模型效果对比')
    st.markdown('> 数据来源：各模型在**测试集**上的真实评估结果（随机森林/调参版FastText/BERT 均已实测）')

    # 真实测试集评估数据（来自 _06_app/_08_调参后评估.py 与 _04_bert_base/_06_bert_test_eval.py 与 _05_llm_prompt评估）
    default_data = {
        '模型': ['随机森林', 'FastText(词+调参)', 'BERT', 'LLM(DeepSeek)'],
        '准确率': [0.7797, 0.8274, 0.8420, 0.5038],
        '精确率': [0.7517, 0.8109, 0.8282, 0.5433],
        '召回率': [0.7453, 0.8112, 0.8173, 0.4733],
        'F1分数': [0.7457, 0.8105, 0.8203, 0.4763],
        '备注': ['TF-IDF+RF', 'Optuna调参', 'bert-base微调', 'DeepSeek-chat 零样本'],
    }
    df_result = pd.DataFrame(default_data)
    edited = st.data_editor(df_result, num_rows='dynamic')

    # 指标卡片
    st.markdown('---')
    col1, col2, col3 = st.columns(3)
    best = edited[edited['F1分数'] > 0].sort_values('F1分数', ascending=False)
    if len(best) > 0:
        col1.metric('🏆 最优模型', best.iloc[0]['模型'])
        col2.metric('最高 F1 分数', f"{best.iloc[0]['F1分数']:.4f}")
        col3.metric('平均 F1 分数', f"{best['F1分数'].mean():.4f}")

    if st.button('📊 生成对比图', type='primary'):
        if 'F1分数' in edited.columns and len(edited) > 0:
            plot_data = edited[edited['F1分数'] > 0]
            if len(plot_data) == 0:
                st.warning('暂无已评估的模型数据，请先填写F1分数')
            else:
                fig, ax = plt.subplots(figsize=(10, 5))
                bars = ax.bar(plot_data['模型'], plot_data['F1分数'], color='#2563EB', alpha=0.9,
                              edgecolor='#050505', width=0.5)
                ax.set_ylabel('F1分数', fontsize=11)
                ax.set_title('各模型F1分数对比', fontsize=14, fontweight='bold')
                ax.grid(axis='y', alpha=0.3)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                for bar, v in zip(bars, plot_data['F1分数']):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                            f'{v:.4f}', ha='center', fontsize=11, color='#D1D5DB')
                st.pyplot(fig)

                # 四指标对比图
                st.markdown('---')
                st.subheader('📊 四指标综合对比')
                fig2, ax2 = plt.subplots(figsize=(10, 5))
                metrics = ['准确率', '精确率', '召回率', 'F1分数']
                x = range(len(metrics))
                width = 0.2
                colors = ['#2563EB', '#10B981', '#F59E0B', '#EF4444']
                for i, model in enumerate(plot_data['模型']):
                    vals = [plot_data[m].iloc[i] for m in metrics]
                    ax2.bar([xi + (i - len(plot_data) / 2) * width for xi in x], vals,
                            width=width, label=model, color=colors[i % len(colors)], alpha=0.9)
                ax2.set_xticks(list(x))
                ax2.set_xticklabels(metrics, fontsize=11)
                ax2.set_ylabel('分数', fontsize=11)
                ax2.set_ylim(0, 1.05)
                ax2.legend()
                ax2.grid(axis='y', alpha=0.3)
                ax2.spines['top'].set_visible(False)
                ax2.spines['right'].set_visible(False)
                st.pyplot(fig2)
