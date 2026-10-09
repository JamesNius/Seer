@echo off
title SeerPolicy distill (teacher = seer.1.0 final)
chcp 65001 >nul
cd /d "E:\Program Files\JetBrains\IdeaProjects\Mortal\mortal"
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set MORTAL_CFG=E:\Program Files\JetBrains\IdeaProjects\Mortal\mortal\config.toml

if not exist logs mkdir logs

echo ============================================================
echo  Distill: value-based teacher -^> policy student (GroupNorm)
echo  Teacher: checkpoints\seer.1.0.*.pth   (1,872,615 steps)
echo  Steps  : config.toml [distill] max_steps  (default 100000)
echo  Output : checkpoints\SeerPolicy.pth    (resumable by AWR)
echo ------------------------------------------------------------
echo  Progress prints in THIS window every 20 steps, e.g.:
echo    [DISTILL] step 1,234/100,000 (1.2%%) ^| loss=... ^| teacher-match=... ^| ETA ...h
echo  Full log : mortal\logs\distill.log
echo  Ctrl+C anytime - checkpoints\SeerPolicy.pth = current progress
echo ============================================================
echo.

"E:\ProgramData\miniconda3\envs\mortal\python.exe" train_distill.py

echo.
echo ================== DISTILL FINISHED ==================
echo Log    : mortal\logs\distill.log
echo Output : mortal\checkpoints\SeerPolicy.pth
pause
