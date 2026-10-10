# -*- coding: utf-8 -*-
"""
12345工单打标项目 - Voxel Hero 风格浏览器前端服务器
（纯 Python 标准库实现，无需安装额外依赖）

功能:
  1. 托管项目根目录 index.html（Voxel Hero 风格落地页）
  2. 提供 REST API: 单模型预测 / 四模型投票 / 类别列表 / 模型列表

运行:
  python _06_app/_05_web_app.py
  然后浏览器打开 http://127.0.0.1:8000

说明:
  - 模型懒加载，首次调用某模型时需要等待其加载（BERT 较慢）
  - LLM(DeepSeek) 需先在"在线打标"页面下方配置 API Key 或设置环境变量 DEEPSEEK_API_KEY
"""
import os
import sys
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

# Windows 控制台默认 GBK，强制 UTF-8 输出，避免打印 emoji/中文报错
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# 把项目根目录加入系统路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _06_app._01_model_service import ModelService

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_HTML_PATH = os.path.join(PROJECT_ROOT, 'index.html')

PORT = 8000

service = ModelService()

MODELS = ['随机森林', 'FastText', 'BERT', 'LLM(DeepSeek)']


def get_classes():
    """读取14个类别名称"""
    from _01_data._01_config import Config
    return list(Config().id2class.values())


class Handler(BaseHTTPRequestHandler):
    """HTTP 处理器"""

    def _send(self, code, body, ctype='application/json; charset=utf-8'):
        if isinstance(body, str):
            body = body.encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get('Content-Length', 0))
        raw = self.rfile.read(length).decode('utf-8')
        return json.loads(raw) if raw else {}

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False))

    # ---------- GET ----------
    def do_GET(self):
        path = urlparse(self.path).path

        if path in ('/', '/index.html'):
            if not os.path.exists(INDEX_HTML_PATH):
                self._send(404, 'index.html 不存在，请将 index.html 放在项目根目录')
                return
            with open(INDEX_HTML_PATH, 'rb') as f:
                self._send(200, f.read(), 'text/html; charset=utf-8')

        elif path == '/api/classes':
            self._json({'ok': True, 'classes': get_classes()})

        elif path == '/api/models':
            self._json({'ok': True, 'models': MODELS})

        elif path == '/api/health':
            self._json({'ok': True, 'name': '12345工单智能打标系统'})

        else:
            self._json({'ok': False, 'error': 'Not Found: ' + path}, 404)

    # ---------- POST ----------
    def do_POST(self):
        path = urlparse(self.path).path
        try:
            data = self._read_json()
        except Exception:
            self._json({'ok': False, 'error': '请求体格式错误'}, 400)
            return

        # 单模型预测
        if path == '/api/predict':
            text = (data.get('text') or '').strip()
            model = data.get('model') or '随机森林'
            if not text:
                self._json({'ok': False, 'error': '请输入工单内容'})
                return
            label, idx = service.predict(text, model)
            self._json({'ok': idx != -1, 'model': model, 'label': label, 'idx': idx})

        # 四模型同时投票
        elif path == '/api/compare':
            text = (data.get('text') or '').strip()
            if not text:
                self._json({'ok': False, 'error': '请输入工单内容'})
                return
            results = []
            for name in MODELS:
                label, idx = service.predict(text, name)
                results.append({'model': name, 'label': label, 'idx': idx, 'ok': idx != -1})
            self._json({'ok': True, 'results': results})

        else:
            self._json({'ok': False, 'error': 'Not Found: ' + path}, 404)

    def do_OPTIONS(self):
        self._send(204, '')

    def log_message(self, fmt, *args):
        sys.stdout.write('[server] ' + fmt % args + '\n')


def main():
    try:
        server = ThreadingHTTPServer(('0.0.0.0', PORT), Handler)
    except OSError as e:
        print(f'❌ 端口 {PORT} 被占用，请关闭占用程序或修改 _05_web_app.py 中的 PORT 后重试: {e}')
        sys.exit(1)

    print('=' * 60)
    print('🏷️  12345工单智能打标系统 - Voxel Hero 前端已启动!')
    print('本机访问: http://127.0.0.1:' + str(PORT))
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        print(f'局域网访问(手机需连同一WiFi): http://{ip}:{PORT}')
    except Exception:
        pass
    print('=' * 60)
    print('提示: 首次调用 BERT 模型需加载权重，等待较久属正常现象')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n服务器已停止')


if __name__ == '__main__':
    main()
