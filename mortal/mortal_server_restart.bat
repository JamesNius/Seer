@echo off
REM 安全重启 Mortal 复盘服务: 先杀掉所有旧的 mortal_review_server 进程, 再启动新进程。
REM 解决"重启了却跑的旧代码"——旧进程没被杀导致一直吐旧数据。
chcp 65001 >nul

echo [restart] 停止旧的 mortal_review_server 进程...
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -like '*mortal_review_server*' } | ForEach-Object { Write-Host ('  kill PID ' + $_.ProcessId); Stop-Process -Id $_.ProcessId -Force }"

REM 等待端口释放
timeout /t 2 /nobreak >nul

echo [restart] 启动新的 Mortal 复盘服务 (端口 %1, 默认 9876)...
set PORT=%1
if "%PORT%"=="" set PORT=9876

set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set MORTAL_CFG=E:\Program Files\JetBrains\IdeaProjects\Mortal\mortal\config.toml
call "E:\ProgramData\miniconda3\Scripts\activate.bat"
call conda activate mortal
cd /d "E:\Program Files\JetBrains\IdeaProjects\Mortal\mortal"
python mortal_review_server.py %PORT%
