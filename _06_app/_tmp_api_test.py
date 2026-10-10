# -*- coding: utf-8 -*-
"""临时测试脚本：验证 /api/predict 与 /api/compare 的 UTF-8 链路"""
import json
import urllib.request

BASE = 'http://127.0.0.1:8000'
TEXT = '我家小区门口的路灯坏了三个月没人修，晚上出行非常危险，请尽快处理。'


def post(path, payload):
    data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode('utf-8'))


if __name__ == '__main__':
    r1 = post('/api/predict', {'text': TEXT, 'model': '随机森林'})
    print('PREDICT:', json.dumps(r1, ensure_ascii=False))

    r2 = post('/api/compare', {'text': TEXT})
    print('COMPARE:', json.dumps(r2, ensure_ascii=False))

    with open(__file__.replace('_tmp_api_test.py', '_tmp_api_out.txt'), 'w', encoding='utf-8') as f:
        f.write(json.dumps({'predict': r1, 'compare': r2}, ensure_ascii=False, indent=2))
