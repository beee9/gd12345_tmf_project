@echo off
title 12345工单打标-前端落地页(8000)
set PY=E:\anaconda3\envs\pytorch-gpu\python.exe
cd /d "E:\Binge project\gd12345_tmf_project"
echo ============================================
echo   12345工单智能打标系统 - 前端落地页
echo   环境: pytorch-gpu
echo   正在启动服务，请稍候...
echo ============================================
start "" /min "%PY%" _06_app\_05_web_app.py
timeout /t 5 /nobreak >nul
start http://127.0.0.1:8000
echo.
echo 已打开浏览器。若页面未加载，请手动访问:
echo   http://127.0.0.1:8000
echo.
echo 关闭本窗口不影响服务运行;如需停止服务,
echo 请在任务管理器中结束 python.exe 进程。
pause
