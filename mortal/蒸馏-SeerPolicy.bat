@echo off
title SeerPolicy distill (teacher = seer.1.0.收敛版)
chcp 65001 >nul

REM ============================================================
REM  蒸馏: value-based 教师(seer.1.0.收敛版) → policy 学生(GN)
REM  步数由 config.toml 的 [distill] max_steps 决定 (默认 30000 ≈ 10h)
REM  产出: checkpoints/SeerPolicy.pth  (可被 train_offline_phase1.py resume)
REM ============================================================

cd /d "E:\Program Files\JetBrains\IdeaProjects\Mortal\mortal"
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set MORTAL_CFG=E:\Program Files\JetBrains\IdeaProjects\Mortal\mortal\config.toml

if not exist logs mkdir logs
set LOG=logs\distill_%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%.log
set LOG=%LOG: =0%

echo ============================================================
echo  蒸馏开始 (3 万步 ≈ 10 小时)
echo  日志: mortal\%LOG%
echo  看进度: 另开一个终端  powershell -c "Get-Content mortal\%LOG% -Wait -Tail 20"
echo  中途可 Ctrl+C 停止, checkpoints\SeerPolicy.pth 是当前进度
echo ============================================================
echo.

"E:\ProgramData\miniconda3\envs\mortal\python.exe" train_distill.py > "%LOG%" 2>&1

echo.
echo ================== 蒸馏结束 ==================
echo 日志: mortal\%LOG%
echo 产出: mortal\checkpoints\SeerPolicy.pth
pause
