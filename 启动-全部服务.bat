@echo off
title 12345工单打标系统-全部服务
set PY=E:\anaconda3\envs\pytorch-gpu\python.exe
cd /d "E:\Binge project\gd12345_tmf_project"
echo ============================================
echo   12345工单智能打标系统 - 一键启动全部服务
echo   环境: pytorch-gpu
echo   前端: http://127.0.0.1:8000
echo   大屏: http://127.0.0.1:8501
echo ============================================
start "" /min "%PY%" _06_app\_05_web_app.py
start "" /min "%PY%" -m streamlit run _06_app\_04_streamlit_dashboard.py --server.port 8501 --server.address 0.0.0.0
timeout /t 6 /nobreak >nul
start http://127.0.0.1:8000
start http://127.0.0.1:8501
echo.
echo 两个页面已打开。若未加载，请手动访问上述地址。
pause
