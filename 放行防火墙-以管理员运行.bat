@echo off
title 12345工单-防火墙放行
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo 需要管理员权限，正在请求授权，请在弹出的窗口点"是"...
    powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)
echo 正在添加防火墙放行规则(端口 8000/8501)...
netsh advfirewall firewall delete rule name="VoxelWeb8000" >nul 2>&1
netsh advfirewall firewall delete rule name="VoxelDash8501" >nul 2>&1
netsh advfirewall firewall add rule name="VoxelWeb8000" dir=in action=allow protocol=TCP localport=8000
netsh advfirewall firewall add rule name="VoxelDash8501" dir=in action=allow protocol=TCP localport=8501
echo.
echo 完成! 已放行 TCP 8000(前端落地页) 和 8501(数据大屏)
echo.
echo 同一网络下的其他电脑现在可访问:
echo   http://192.168.1.6:8000   (前端)
echo   http://192.168.1.6:8501   (数据大屏)
echo.
echo 外网访问请配合内网穿透工具(cpolar / natapp / ngrok)
pause
