# -*- coding: utf-8 -*-
"""
12345工单打标项目 - Gradio 可视化交互界面
运行: python _06_app/_02_gradio_app.py  然后浏览器打开 http://127.0.0.1:7860
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _06_app._01_model_service import ModelService
from _01_data._01_config import Config

config = Config()

# 模型服务单例
service = ModelService()

# 可选模型列表
MODEL_CHOICES = ['随机森林', 'FastText', 'BERT', 'LLM(DeepSeek)']


def predict_compare(text):
    """四模型同时投票：RF / FastText / BERT / LLM"""
    if not text or not text.strip():
        return '请输入工单内容'
    rows = []
    for name in ['随机森林', 'FastText', 'BERT', 'LLM(DeepSeek)']:
        label, idx = service.predict(text.strip(), name)
        if idx == -1:
            rows.append(f'| {name} | ❌ {label} |')
        else:
            rows.append(f'| {name} | **{label}** (id={idx}) |')
    table = '| 模型 | 预测类别 |\n|---|---|\n' + '\n'.join(rows)
    return table


def save_llm_key(key):
    """保存DeepSeek API Key"""
    if not key or not key.strip():
        return '❌ Key不能为空'
    if service.save_deepseek_key(key):
        return '✅ DeepSeek API Key 已保存！可以直接用 LLM 模型了'
    return '❌ 保存失败'


def predict_interface(text, model_name, show_detail):
    """Gradio 回调函数：输入工单文本 + 模型选择，返回打标结果"""
    if not text or not text.strip():
        return '请输入工单内容', '', ''

    label, idx = service.predict(text.strip(), model_name)

    if idx == -1:
        return f'❌ {label}', '', ''

    # 详细信息
    detail = f"""
### 分类结果
- **工单内容**: {text[:100]}{'...' if len(text) > 100 else ''}
- **模型**: {model_name}
- **预测类别**: **{label}** (id={idx})
- **类别说明**: {get_class_desc(label)}
"""
    if show_detail:
        return f'**{label}**', detail, label
    else:
        return f'**{label}**', '', label


def get_class_desc(label):
    desc_map = {
        '科教文体': '科学技术/教育/文化/体育类事项',
        '公卫医疗': '公共卫生/医疗卫生/医保类事项',
        '城乡建设': '市政建设/住房/规划/物业类事项',
        '自然资源与环境保护': '环境污染/生态/自然资源类事项',
        '交通运输': '公共交通/道路/停车类事项',
        '农林牧渔': '农业/林业/畜牧业/渔业类事项',
        '公共安全': '治安/消防/安全生产类事项',
        '人力资源社会保障': '劳动/就业/社保类事项',
        '民政社区': '低保/养老/社区服务类事项',
        '市场监管': '消费维权/食品安全/价格类事项',
        '经济财贸': '招商引资/企业/商贸类事项',
        '司法行政': '法律/信访/仲裁类事项',
        '政风行风党风': '干部作风/党风廉政类事项',
        '其他': '其他咨询/建议/投诉',
    }
    return desc_map.get(label, '')


def build_app():
    import gradio as gr

    with gr.Blocks(title='12345工单自动打标系统') as demo:
        gr.Markdown('# 🏷️ 12345工单自动打标系统')
        gr.Markdown('输入工单内容，选择模型，自动判断所属类别（共14类）')

        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label='工单内容',
                    placeholder='请输入工单内容，例如：我家小区门口的路灯坏了三个月没人修，晚上出行非常危险',
                    lines=5)
                gr.Markdown('**📝 点示例一键填入**（点击下面任意一条，自动填到输入框）')
                gr.Examples(
                    examples=[
                        ['我家小区门口的路灯坏了三个月没人修，晚上出行非常危险，请尽快处理。'],
                        ['我去年冬天在商场买了一件羽绒服，最近拿出来穿发现严重掉毛，去找商家要求退货，商家说已经过了三包期限不给退，我该怎么办。'],
                        ['在XX医院住院三天，被要求自费购买了2000多元的进口药，医生说不用这个药就不给安排手术，这合理吗？'],
                        ['XX建筑公司拖欠我三个月工资共18000元，老板电话不接，人也找不到，希望政府帮忙讨薪。'],
                        ['楼下烧烤店每晚营业到凌晨两三点，油烟味熏得整栋楼都开不了窗，还经常有客人半夜大声喧哗，严重扰民。'],
                        ['XX路公交车站牌显示末班车是21:30，但上周连续三天21点就等不到车了，等了40分钟没来，请核实。'],
                        ['XX化工厂夜间偷偷排放黑色污水到河里，河水已经变黑发臭，鱼都死了，请环保部门来查。'],
                        ['我在网上报名的培训班突然关门跑路，交了5800元学费只上了一半的课，联系负责人电话已停机。'],
                    ],
                    inputs=[text_input],
                    label=None,
                    examples_per_page=4,
                )
                model_choice = gr.Dropdown(
                    choices=MODEL_CHOICES, value='随机森林',
                    label='选择模型',
                    info='随机森林最快(CPU)；BERT效果最好；LLM需配置DeepSeek API Key')
                show_detail = gr.Checkbox(label='显示详细信息', value=True)
                submit_btn = gr.Button('🚀 开始打标', variant='primary')
                compare_btn = gr.Button('🚦 四模型同时投票对比', variant='secondary')

                with gr.Accordion('🤖 LLM 设置（DeepSeek API Key）', open=False):
                    llm_key_input = gr.Textbox(
                        label='DeepSeek API Key',
                        placeholder='输入你的 DeepSeek API Key（sk-开头），填一次即可，保存在本地',
                        type='password')
                    save_key_btn = gr.Button('💾 保存 Key', size='sm')
                    key_status = gr.Markdown()

            with gr.Column(scale=1):
                output_label = gr.Markdown('**等待输入...**')
                output_detail = gr.Markdown()
                output_text = gr.Textbox(label='预测类别(复制用)', visible=False)
                compare_output = gr.Markdown()

        gr.Markdown('---')
        gr.Markdown('### 📊 类别列表')
        gr.Markdown('、'.join(f'**{i+1}.** {name}' for i, name in enumerate(config.id2class.values())))

        submit_btn.click(
            fn=predict_interface,
            inputs=[text_input, model_choice, show_detail],
            outputs=[output_label, output_detail, output_text])

        text_input.submit(
            fn=predict_interface,
            inputs=[text_input, model_choice, show_detail],
            outputs=[output_label, output_detail, output_text])

        compare_btn.click(
            fn=predict_compare,
            inputs=[text_input],
            outputs=[compare_output])

        save_key_btn.click(
            fn=save_llm_key,
            inputs=[llm_key_input],
            outputs=[key_status])

    return demo


if __name__ == '__main__':
    import gradio as gr
    demo = build_app()
    print('=' * 60)
    print('工单打标系统已启动！')
    print('本机访问: http://127.0.0.1:7860')
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        print(f'局域网访问(手机需连同一WiFi): http://{ip}:7860')
    except Exception:
        pass
    print('=' * 60)
    demo.launch(server_name='0.0.0.0', server_port=7860, inbrowser=True, theme=gr.themes.Soft())
