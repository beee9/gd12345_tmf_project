@echo off
title 12345工单打标-数据大屏(8501)
set PY=E:\anaconda3\envs\pytorch-gpu\python.exe
cd /d "E:\Binge project\gd12345_tmf_project"
echo ============================================
echo   12345工单智能打标系统 - 数据大屏
echo   环境: pytorch-gpu
echo   正在启动 Streamlit，请稍候...
echo ============================================
"%PY%" -m streamlit run _06_app\_04_streamlit_dashboard.py --server.port 8501 --server.address 0.0.0.0
pause
