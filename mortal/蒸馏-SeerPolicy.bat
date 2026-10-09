@echo off
title SeerPolicy distill (teacher = seer.1.0 final)
cd /d "E:\Program Files\JetBrains\IdeaProjects\Mortal\mortal"
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set MORTAL_CFG=E:\Program Files\JetBrains\IdeaProjects\Mortal\mortal\config.toml

if not exist logs mkdir logs

echo ============================================================
echo  Distill: value-based teacher -^> policy student (GroupNorm)
echo  Teacher: checkpoints\seer.1.0.*.pth   (1,872,615 steps)
echo  Steps  : config.toml [distill] max_steps  (default 30000, ~10h)
echo  Output : checkpoints\SeerPolicy.pth    (resumable by AWR)
echo ------------------------------------------------------------
echo  Log file : mortal\logs\distill.log
echo  Watch it : powershell -c "Get-Content 'E:\Program Files\JetBrains\IdeaProjects\Mortal\mortal\logs\distill.log' -Wait -Tail 20"
echo  Ctrl+C anytime - checkpoints\SeerPolicy.pth = current progress
echo ============================================================
echo.

"E:\ProgramData\miniconda3\envs\mortal\python.exe" train_distill.py > logs\distill.log 2>&1

echo.
echo ================== DISTILL FINISHED ==================
echo Log    : mortal\logs\distill.log
echo Output : mortal\checkpoints\SeerPolicy.pth
pause
